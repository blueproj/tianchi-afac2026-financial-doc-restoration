"""
FinixDoc-VL API 调用模块

【合规说明】本文件是全工程**唯一**发起模型调用的地方，调用的是主办方指定的
FinixDoc-VL 官方接口（configs/config.py 中的 API_URL，call_with_file）；
接口只接收图片文件，无文本 prompt 参数，因此本方案不存在 Prompt 工程。
除此之外工程内无任何其他模型调用（无第三方大模型 API、无本地模型权重）。

功能：
- 鲁棒的API调用（重试、超时、错误处理）
- 调用日志记录（耗时、状态、返回内容长度）
- 结果缓存（避免重复调用；对 chunk 图按图片**内容 MD5** 键控，
  这也是在 VLM 输出带随机性的前提下实现 B 榜结果 100% 复现的机制，见 README）
- 多userId轮转（支持后续并行调用）
"""
import os
import re
import time
import json
import hashlib
import logging
import threading
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
import io

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    API_URL, API_KEY, USER_IDS,
    API_TIMEOUT, API_MAX_RETRIES, API_RETRY_DELAY,
    CACHE_DIR, LOG_DIR, ENABLE_CACHE, INTER_CALL_DELAY
)

# ==================== 日志配置 ====================
logger = logging.getLogger("api_client")
logger.setLevel(logging.DEBUG)

# 文件handler - 记录详细日志
file_handler = logging.FileHandler(
    os.path.join(LOG_DIR, f"api_call_{datetime.now().strftime('%Y%m%d')}.log"),
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s'
))

# 控制台handler - 只显示INFO以上
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(message)s'
))

logger.addHandler(file_handler)
logger.addHandler(console_handler)


class FinixDocClient:
    """FinixDoc-VL API 客户端"""

    def __init__(self, user_ids=None, api_key=None, enable_cache=None, inter_call_delay=INTER_CALL_DELAY):
        """
        Args:
            user_ids: 可用的userId列表，默认使用配置中的全部5个
            api_key: API密钥
            enable_cache: 是否启用缓存
            inter_call_delay: 正常调用之间的间隔（秒），避免触发限流
        """
        self.user_ids = user_ids or USER_IDS
        self.api_key = api_key or API_KEY
        self.enable_cache = ENABLE_CACHE if enable_cache is None else enable_cache
        self.inter_call_delay = inter_call_delay
        self._user_id_idx = 0  # userId轮转计数器
        self._last_call_time = 0  # 上次调用时间
        self._lock = threading.Lock()  # 线程安全锁

        # 调用统计
        self.stats = {
            'total_calls': 0,
            'success_calls': 0,
            'fail_calls': 0,
            'total_time': 0.0,
            'total_response_length': 0,
        }

    def _throttle(self):
        """限流：确保两次调用之间至少有inter_call_delay秒的间隔（线程安全）"""
        if self.inter_call_delay <= 0:
            return
        with self._lock:
            if self._last_call_time > 0:
                elapsed = time.time() - self._last_call_time
                if elapsed < self.inter_call_delay:
                    time.sleep(self.inter_call_delay - elapsed)
            self._last_call_time = time.time()

    def _get_next_user_id(self):
        """轮转获取下一个userId（线程安全）"""
        with self._lock:
            uid = self.user_ids[self._user_id_idx % len(self.user_ids)]
            self._user_id_idx += 1
            return uid

    def _stats_inc(self, key, amount=1):
        """线程安全地更新统计"""
        with self._lock:
            self.stats[key] += amount

    def _get_cache_key(self, image_path, user_id=None):
        """根据文件内容与元信息生成稳定缓存key（默认不绑定userId）"""
        file_stat = os.stat(image_path)
        key_str = f"{image_path}_{file_stat.st_size}_{file_stat.st_mtime}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self, cache_key):
        """从缓存加载结果"""
        if not self.enable_cache:
            return None
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            logger.debug(f"命中缓存: {cache_key}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _save_cache(self, cache_key, content):
        """保存结果到缓存"""
        if not self.enable_cache:
            return
        cache_file = os.path.join(CACHE_DIR, f"{cache_key}.txt")
        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _call_api_once(self, image_path, user_id):
        """单次API调用"""
        url = API_URL
        data = {
            'userId': user_id,
            'apiKey': self.api_key,
            'fileName': os.path.basename(image_path),
        }
        headers = {"Expect": ""}
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}
            response = requests.post(url, data=data, files=files, headers=headers, timeout=API_TIMEOUT)

        resp_json = response.json()

        # 解析返回结构: success -> result -> result(json string) -> choices[0].message.content
        if resp_json.get('success') and resp_json.get('result', {}).get('result'):
            result_str = resp_json['result']['result']
            result_data = json.loads(result_str)
            content = result_data['choices'][0]['message']['content']
            return content
        else:
            error_msg = resp_json.get('message', '未知错误')
            raise Exception(f"API返回错误: {error_msg}, 完整响应: {json.dumps(resp_json, ensure_ascii=False)[:500]}")

    def call_api(self, image_path, user_id=None, image_data=None):
        """
        调用FinixDoc-VL API（带重试和缓存）

        Args:
            image_path: 图片文件路径
            user_id: 指定userId，不指定则自动轮转
            image_data: 如果提供，则直接上传bytes数据（用于上传切块后的临时图片）

        Returns:
            str: Markdown解析结果
        """
        # 如果是内存中的图片数据（切块后），需要特殊处理
        if image_data is not None:
            return self._call_api_with_data(image_data, user_id)

        # 检查缓存
        uid = user_id or self._get_next_user_id()
        cache_key = self._get_cache_key(image_path, uid)
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._stats_inc('total_calls')
            self._stats_inc('success_calls')
            return cached

        # 带重试的API调用（指数退避）
        file_name = os.path.basename(image_path)
        self._stats_inc('total_calls')

        for attempt in range(1, API_MAX_RETRIES + 1):
            self._throttle()
            start_time = time.time()
            try:
                logger.info(f"调用API: file={file_name}, userId={uid}, attempt={attempt}/{API_MAX_RETRIES}")
                content = self._call_api_once(image_path, uid)
                elapsed = time.time() - start_time

                self._stats_inc('success_calls')
                self._stats_inc('total_time', elapsed)
                self._stats_inc('total_response_length', len(content))

                logger.info(
                    f"API成功: file={file_name}, userId={uid}, "
                    f"耗时={elapsed:.2f}s, 返回长度={len(content)}字符"
                )

                # 保存缓存
                self._save_cache(cache_key, content)
                return content

            except Exception as e:
                elapsed = time.time() - start_time
                logger.warning(
                    f"API失败(attempt {attempt}/{API_MAX_RETRIES}): "
                    f"file={file_name}, userId={uid}, 耗时={elapsed:.2f}s, "
                    f"错误={str(e)[:200]}"
                )
                if attempt < API_MAX_RETRIES:
                    # 指数退避：10s, 20s, 40s, 80s...
                    delay = API_RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"  等待 {delay}s 后重试...")
                    time.sleep(delay)
                else:
                    self._stats_inc('fail_calls')
                    logger.error(f"API最终失败: file={file_name}, 错误={str(e)[:300]}")
                    return ""

    def _call_api_with_data(self, image_bytes, user_id=None):
        """用内存中的图片bytes数据调用API（用于切块后的临时图片）"""
        uid = user_id or self._get_next_user_id()

        # 基于图片内容生成缓存key
        img_hash = hashlib.md5(image_bytes).hexdigest()
        cache_key = f"chunk_{img_hash}"
        cached = self._load_cache(cache_key)
        if cached is not None:
            self._stats_inc('total_calls')
            self._stats_inc('success_calls')
            return cached

        self._stats_inc('total_calls')

        for attempt in range(1, API_MAX_RETRIES + 1):
            self._throttle()
            start_time = time.time()
            try:
                url = API_URL
                data = {
                    'userId': uid,
                    'apiKey': self.api_key,
                    'fileName': f'chunk_{img_hash[:8]}.jpg',
                }
                headers = {"Expect": ""}
                files = {'file': (f'chunk_{img_hash[:8]}.jpg', image_bytes, 'image/jpeg')}
                response = requests.post(url, data=data, files=files, headers=headers, timeout=API_TIMEOUT)

                resp_json = response.json()
                if resp_json.get('success') and resp_json.get('result', {}).get('result'):
                    result_str = resp_json['result']['result']
                    result_data = json.loads(result_str)
                    content = result_data['choices'][0]['message']['content']

                    elapsed = time.time() - start_time
                    self._stats_inc('success_calls')
                    self._stats_inc('total_time', elapsed)
                    self._stats_inc('total_response_length', len(content))

                    logger.info(
                        f"API成功(chunk): hash={img_hash[:8]}, userId={uid}, "
                        f"耗时={elapsed:.2f}s, 返回长度={len(content)}字符"
                    )
                    self._save_cache(cache_key, content)
                    return content
                else:
                    error_msg = resp_json.get('message', '未知错误')
                    raise Exception(f"API返回错误: {error_msg}")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.warning(
                    f"API失败(chunk, attempt {attempt}/{API_MAX_RETRIES}): "
                    f"hash={img_hash[:8]}, userId={uid}, 耗时={elapsed:.2f}s, "
                    f"错误={str(e)[:200]}"
                )
                if attempt < API_MAX_RETRIES:
                    # 指数退避
                    delay = API_RETRY_DELAY * (2 ** (attempt - 1))
                    logger.info(f"  等待 {delay}s 后重试...")
                    time.sleep(delay)
                else:
                    self._stats_inc('fail_calls')
                    logger.error(f"API最终失败(chunk): hash={img_hash[:8]}, 错误={str(e)[:300]}")
                    return ""

    def _is_empty_result(self, content):
        """检查API返回是否为空结果（只有markdown代码块包裹，无实际内容）"""
        if not content or len(content.strip()) < 10:
            return True
        # 去除markdown代码块包裹后的实际内容
        cleaned = re.sub(r'^```(?:markdown|md)?\s*\n', '', content.strip())
        cleaned = re.sub(r'\n```\s*$', '', cleaned.strip())
        return len(cleaned.strip()) < 5

    def call_pil_image(self, pil_image, user_id=None):
        """
        直接传入PIL Image对象调用API（用于切块后的图片）
        如果API返回空结果，自动缩小图片重试。

        Args:
            pil_image: PIL.Image对象
            user_id: 指定userId

        Returns:
            str: Markdown解析结果
        """
        original_img = pil_image
        current_img = pil_image

        # 最多尝试3种尺寸：原图、缩小一半、缩小到1/4
        for scale_idx in range(3):
            # 将PIL Image转为bytes
            buf = io.BytesIO()
            if current_img.mode != 'RGB':
                current_img = current_img.convert('RGB')
            current_img.save(buf, format='JPEG', quality=95)
            image_bytes = buf.getvalue()

            content = self._call_api_with_data(image_bytes, user_id)

            # 检查返回是否为空
            if not self._is_empty_result(content):
                return content

            if scale_idx < 2:
                w, h = current_img.size
                logger.info(f"  API返回空结果(长度={len(content)}), 尝试缩小图片: {w}x{h} -> {w//2}x{h//2}")
                current_img = current_img.resize((w // 2, h // 2), Image.LANCZOS)
            else:
                logger.warning(f"  API返回空结果，已尝试3种尺寸，返回原始结果")

        return content


    def get_stats(self):
        """获取调用统计"""
        s = self.stats.copy()
        if s['total_calls'] > 0:
            s['success_rate'] = s['success_calls'] / s['total_calls']
            s['avg_time'] = s['total_time'] / s['success_calls'] if s['success_calls'] > 0 else 0
        else:
            s['success_rate'] = 0
            s['avg_time'] = 0
        return s
