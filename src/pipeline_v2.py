"""
新版端到端 Pipeline：Level 1 (RegionSplitter) + Level 2 (TableChunker) + API + Merger

支持两种文档类型：
    - **table**（大表格图）：Level 1 分区域 + Level 2 切块 + merge_chunk_results
    - **long**（长文档面条图）：ImageChunker.chunk_long_document + merge_vertical + 后处理
    - **auto**：根据宽高比判断（默认）

流程（table 类型）：
    输入图片
      ↓
    Level 1 (RegionSplitter): 按空白/边框切区域 → List[Region]
      ↓
    对每个 Region：
      - text 区域：直接调 API 拿 markdown
      - table 区域：
          Level 2 (TableChunker.chunk_table)：结构分析 → 无边框/半边框画线
              ↓ 内含 Level 1.5 子表切分（SubTableSplitter，见 table_chunker.py
                 chunk_table 内部：画线之后、切 chunk 之前，检测同一 region 内
                 多个阶梯子表的边界，使行分组不跨子表）
              ↓ 按"行带 × 列组"切成 chunks（含 header 拼接；行列范围写入文件名）
          并发调 API（每完成一个立即写盘）
          反应式纠错重调 / banner 表头分离重建（本文件，见下方导读）
          merge_chunk_results 合并 → <table> HTML
      ↓
    按 order 顺序拼接 + 文档级后处理 → 完整 markdown

流程（long 类型）：
    输入图片
      ↓
    ImageChunker.chunk_long_document: 按空白带切成垂直 chunks
      ↓
    并发调 API（每完成一个立即写盘）
      ↓
    merge_vertical → post_process → post_process_headings → 最终 markdown

**默认所有中间产物实时保存**（save_dir 指向具体目录时生效）。

【审核导读】方法描述文档中的关键技术 → 代码位置对照：
    输入调制（方法文档 §4.1）：
        三级切分与画线在其他文件：
            src/region_splitter.py    Level 1 区域切分
            src/subtable_splitter.py  Level 1.5 子表切分（由 table_chunker 调用）
            src/table_chunker.py      Level 2 结构分析 / 画线 / 行带×列组切块
        本文件内的输入调制：
            _upscale_if_small            小图 2x 放大防幻觉
            _is_empty_chunk_debug        纯空 chunk 跳过不调 API
    反应式纠错重调（方法文档 §4.2）：
        _is_ragged_response          行列不齐/过读/均匀漏列判据
        _is_underrow_response        均匀漏行判据
        _is_degenerate_text_response 纯文本退化判据（列坍塌/溢出/重复循环）
        _resplit_and_recall          左右切分重调（沿"按列数居中"的真实竖线切）
        _resplit_and_recall_vertical 上下切分重调
        _ragged_score / _row_score   采纳门控（严格更优才采纳，零回退风险）
    banner 表头分离重建（方法文档 §4.3）：
        _detect_banner_from_chunks   快路检测（首列组 row0 含 colspan banner）
        _detect_banner_via_strip     全宽表头条兜底检测（banner 居中/靠右时）
        _reconstruct_banner_header   程序化重建 rowspan/colspan 表头
        _scheme_b_reconstruct        表头/主体分离的端到端重建流程
    文档级后处理（方法文档 §4.4）：
        _postprocess_chunk_response  chunk 级：非HTML重建/行列补齐/上方文字并入
        _postprocess_table_markdown  文档级：幻影空行列剔除/每列去重"-"/小表转文本/
                                     空表删除/文本段限长/表尾空行删除 等
        _normalize_table_tags        表格开标签归一化（对齐 GT 格式）
"""
import os
import re
import json
import time
import html as _html
import logging
import traceback
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from src.region_splitter import RegionSplitter, Region
from src.table_chunker import TableChunker, merge_chunk_results
from src.image_chunker import ImageChunker
from src.merger import ResultMerger
from src.heading_normalizer import post_process_headings
from src.api_client import FinixDocClient

Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger("pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)


class Pipeline:
    """
    端到端图像解析 pipeline，默认实时保存所有中间产物。

    支持文档类型：
        - 'table'：大表格图（走 Level 1 + Level 2）
        - 'long'：长文档面条图（走 ImageChunker + merge_vertical）
        - 'auto'：根据宽高比自动判断

    Usage:
        p = Pipeline(chunk_workers=8)
        markdown = p.process_image('/path/to/img.jpg', save_dir='out/xxx/', doc_type='table')
    """

    def __init__(self,
                 api_client: Optional[FinixDocClient] = None,
                 chunk_workers: int = 8,
                 splitter: Optional[RegionSplitter] = None,
                 image_chunker: Optional[ImageChunker] = None,
                 merger: Optional[ResultMerger] = None,
                 enable_em_fill: bool = True):
        self.api = api_client or FinixDocClient()
        self.chunk_workers = chunk_workers
        # EM 空单元格占位符填充开关（False 时跳过填充，仍保留纯空跳过+密集放大）
        self.enable_em_fill = enable_em_fill
        # 【问题4】原生结果行列不齐(API 误读)时，把 chunk 沿竖线中点左右切分重调再合并
        self.enable_resplit = True
        # 【问题4-行】原生"均匀但少行"(漏行)时，把 chunk 沿横线中点上下切分重调再堆叠
        self.enable_resplit_row = True
        # 【方案B】banner 表头（跨列 banner + 左侧标签列）：表头带分离重建 + 主体正常切
        self.enable_scheme_b = True
        # table 流程组件
        self.splitter = splitter or RegionSplitter()
        # long 流程组件
        self.image_chunker = image_chunker or ImageChunker()
        # 长文档使用强空白带零重叠切分，overlap_dedup 必须关闭：
        # 否则 merger 会把「相似的非重叠行」（尤其表格行、类似条款）当成重叠去掉，
        # 导致 4-10% 的内容被误删（详见 docs/长文档识别现状诊断_20260719.md）。
        self.merger = merger or ResultMerger(overlap_dedup=False)

    # ==================== 主流程 ====================

    def process_image(self, img_path: str,
                       save_dir: Optional[str] = None,
                       doc_type: str = 'auto') -> str:
        """
        处理单张图片 → 完整 markdown。

        Args:
            img_path: 图片路径
            save_dir: 中间产物保存目录；None 时不保存
            doc_type: 'table' / 'long' / 'auto'（自动按宽高比判断）

        Returns:
            完整 markdown 字符串
        """
        t_start = time.time()
        img_name = os.path.basename(img_path)

        # 自动判断文档类型
        if doc_type == 'auto':
            doc_type = self._detect_doc_type(img_path)
        logger.info(f"处理: {img_name} [doc_type={doc_type}]")

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        # 分流
        if doc_type == 'long':
            markdown = self._process_long_document(img_path, save_dir, t_start)
        else:
            markdown = self._process_table_document(img_path, save_dir, t_start)
            # 表格开标签归一化已在 _process_table_document 内（保存 final.md 前）完成

        elapsed = time.time() - t_start
        logger.info(f"  完成 [{doc_type}]: {len(markdown)} 字符, 耗时 {elapsed:.1f}s")
        return markdown

    @staticmethod
    def _detect_doc_type(img_path: str) -> str:
        """按宽高比 + 高度判断：宽高比<0.1 或 高>10000 → long；否则 → table"""
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(img_path) as img:
            w, h = img.size
        if (w / max(1, h)) < 0.1 or h > 10000:
            return 'long'
        return 'table'

    @staticmethod
    def _normalize_table_tags(markdown: str) -> str:
        """将所有表格开标签统一为 GT 格式 <table border="1" cellpadding="8" cellspacing="0">。
        仅替换开标签（</table> 不受影响）；不改单元格内容与结构。"""
        if not markdown or '<table' not in markdown:
            return markdown
        return re.sub(r'<table[^>]*>',
                      '<table border="1" cellpadding="8" cellspacing="0">',
                      markdown)

    # ==================== 长文档流程 ====================

    def _process_long_document(self, img_path: str,
                                 save_dir: Optional[str],
                                 t_start: float) -> str:
        """
        长文档处理：切块 → 并发 API → merge_vertical → post_process → post_process_headings
        """
        img_name = os.path.basename(img_path)

        # Step 1: 切块（用 ImageChunker.chunk_long_document）
        chunks = self.image_chunker.chunk_long_document(img_path)
        logger.info(f"  长文档切块: {len(chunks)} chunks")

        # Step 2: 保存 chunk 图 + meta（立即）
        long_dir = None
        if save_dir:
            long_dir = os.path.join(save_dir, 'long_chunks')
            os.makedirs(long_dir, exist_ok=True)
            self._save_long_chunks_before_api(chunks, long_dir)

        # Step 3: 并发调 API，每完成一个立即保存
        responses = self._call_long_chunks_realtime(chunks, long_dir)

        # Step 4: 合并 + 后处理（三步都保留中间结果）
        merged_raw = self.merger.merge_vertical(responses)
        merged_post = self.merger.post_process(merged_raw)
        merged_final = post_process_headings(merged_post)

        # Step 5: 实时保存
        if save_dir:
            self._save_text(os.path.join(save_dir, 'merged_raw.md'), merged_raw)
            self._save_text(os.path.join(save_dir, 'merged_post.md'), merged_post)
            self._save_text(os.path.join(save_dir, 'final.md'), merged_final)

            # 收集详情供 HTML report
            chunk_details = []
            for i, ((chunk_img, pos), resp) in enumerate(zip(chunks, responses)):
                cw, ch = chunk_img.size
                base = f"chunk_{i:03d}"
                chunk_details.append({
                    'order': i,
                    'position': self._pos_to_text(pos),
                    'size': [cw, ch],
                    'image_file': f"long_chunks/{base}.jpg",
                    'response': resp or '',
                })
            summary = {
                'image': img_name,
                'doc_type': 'long',
                'markdown_len': len(merged_final),
                'elapsed_s': round(time.time() - t_start, 1),
                'n_chunks': len(chunks),
                'merged_raw_len': len(merged_raw),
                'merged_post_len': len(merged_post),
            }
            self._save_text(os.path.join(save_dir, 'summary.json'),
                            json.dumps(summary, ensure_ascii=False, indent=2))
            self._save_long_html_report(save_dir, img_path, chunk_details,
                                          merged_raw, merged_post, merged_final)

        return merged_final

    def _save_long_chunks_before_api(self, chunks: List[Tuple[Image.Image, tuple]],
                                       long_dir: str):
        """长文档 chunk 图 + meta 立即保存"""
        metas = []
        for i, (chunk_img, pos) in enumerate(chunks):
            cw, ch = chunk_img.size
            base = f"chunk_{i:03d}"
            chunk_img.save(os.path.join(long_dir, f"{base}.jpg"), quality=90)
            metas.append({
                'order': i, 'position': list(pos) if pos else None,
                'size': [cw, ch], 'file': f"{base}.jpg",
            })
        self._save_text(os.path.join(long_dir, 'chunks_meta.json'),
                        json.dumps(metas, ensure_ascii=False, indent=2))

    def _call_long_chunks_realtime(self,
                                     chunks: List[Tuple[Image.Image, tuple]],
                                     save_dir: Optional[str]) -> List[str]:
        """
        长文档 chunk 的 API 并发调用，每完成一个立即保存对应 `_api.txt`。
        """
        n = len(chunks)
        if n == 0:
            return []
        results: List[str] = [''] * n

        if n == 1:
            resp = self.api.call_pil_image(chunks[0][0]) or ''
            results[0] = resp
            if save_dir:
                self._save_text(os.path.join(save_dir, 'chunk_000_api.txt'), resp)
            return results

        workers = min(self.chunk_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(self.api.call_pil_image, chunks[idx][0]): idx
                for idx in range(n)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    resp = future.result()
                except Exception as e:
                    logger.error(f"long chunk#{idx} API 失败: {e}")
                    resp = ''
                results[idx] = resp or ''
                if save_dir:
                    self._save_text(
                        os.path.join(save_dir, f"chunk_{idx:03d}_api.txt"),
                        results[idx])
        return results

    @staticmethod
    def _pos_to_text(pos) -> str:
        """把 position 元组转为可读文本"""
        if pos is None:
            return ''
        if len(pos) == 4:
            x0, y0, x1, y1 = pos
            return f"x={x0}:{x1}, y={y0}:{y1}"
        if len(pos) == 2:
            return f"y={pos[0]}:{pos[1]}"
        return str(pos)

    # ==================== 表格文档流程 ====================

    def _process_table_document(self, img_path: str,
                                  save_dir: Optional[str],
                                  t_start: float) -> str:
        """
        表格文档处理：Level 1 分区域 + Level 2 切块 + 合并
        """
        img_name = os.path.basename(img_path)

        # ----- Level 1 -----
        regions = self.splitter.split(img_path)
        n_tables = sum(1 for r in regions if r.region_type == 'table')
        n_texts = sum(1 for r in regions if r.region_type == 'text')
        logger.info(f"  Level 1: {len(regions)} regions ({n_texts} text + {n_tables} table)")

        if save_dir:
            self._save_level1(regions, img_name, save_dir)

        # ----- 按 order 处理每个 region -----
        parts: List[str] = []
        region_details: List[Dict] = []
        for r in regions:
            if r.region_type == 'text':
                md, raw = self._process_text_region(r, save_dir)
                region_details.append({
                    'order': r.order, 'type': 'text',
                    'bbox': list(r.bbox), 'size': [r.width, r.height],
                    'image_file': f"r{r.order:02d}_{r.region_type}_{r.width}x{r.height}.jpg",
                    'raw': raw or '', 'processed': md, 'chunks': [],
                })
            else:
                md, chunk_details = self._process_table_region(r, save_dir)
                region_details.append({
                    'order': r.order, 'type': 'table',
                    'bbox': list(r.bbox), 'size': [r.width, r.height],
                    'image_file': f"r{r.order:02d}_{r.region_type}_{r.width}x{r.height}.jpg",
                    'raw': '', 'processed': md, 'chunks': chunk_details,
                })
            if md:
                parts.append(md)

        # ----- 拼接 -----
        markdown = '\n\n'.join(parts).strip()

        # ----- 后处理 -----
        markdown = self._postprocess_table_markdown(markdown)
        # 统一表格开标签为 GT 格式（在保存 final.md 前做，确保中间产物与 CSV 一致）
        markdown = self._normalize_table_tags(markdown)

        # 让 report.html 的每个 region 展示"与 final.md 一致"的后处理结果：
        # 此前 region 详情渲染的是 merge 原样（= merged.md），与 final.md 的
        # 剔除幻影空列/行、每列去重"-"、<th>→<td>、开标签归一化 对不上。
        # 这些后处理均按 <table> 独立作用，逐 region 应用与整篇一致（不影响 parts/final.md/CSV）。
        for d in region_details:
            d['processed'] = self._normalize_table_tags(
                self._postprocess_table_markdown(d['processed']))

        # 实时保存 final.md + summary.json + HTML report
        if save_dir:
            self._save_text(os.path.join(save_dir, 'final.md'), markdown)
            summary = {
                'image': img_name,
                'doc_type': 'table',
                'markdown_len': len(markdown),
                'elapsed_s': round(time.time() - t_start, 1),
                'n_regions': len(regions),
                'n_texts': n_texts,
                'n_tables': n_tables,
                'regions': [{
                    'order': d['order'], 'type': d['type'],
                    'bbox': d['bbox'], 'size': d['size'],
                    'n_chunks': len(d['chunks']),
                    'md_len': len(d['processed']),
                } for d in region_details],
            }
            self._save_text(os.path.join(save_dir, 'summary.json'),
                            json.dumps(summary, ensure_ascii=False, indent=2))
            self._save_html_report(save_dir, img_path, region_details, markdown)

        return markdown

    # ==================== Level 1 保存 ====================

    def _save_level1(self, regions: List[Region], img_name: str, save_dir: str):
        """Level 1 完成后立即保存 regions.json + 各 region 子图"""
        for r in regions:
            fname = f"r{r.order:02d}_{r.region_type}_{r.width}x{r.height}.jpg"
            r.image.save(os.path.join(save_dir, fname), quality=85)

        info = {
            'image': img_name,
            'num_regions': len(regions),
            'regions': [{
                'order': r.order,
                'type': r.region_type,
                'bbox': list(r.bbox),
                'size': [r.width, r.height],
            } for r in regions],
        }
        with open(os.path.join(save_dir, 'regions.json'), 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

    # ==================== text 区域 ====================

    @staticmethod
    def _pad_white(img: Image.Image, padding: int = None) -> Image.Image:
        """
        四周加白色 padding，避免 API 对小图/边缘文字识别不准。

        Args:
            img: 原始图片
            padding: 四周填充像素数；None 时自适应（短边的 30%，上限 30px）

        Returns:
            加了白色边框的新图片
        """
        w, h = img.size
        if padding is None:
            padding = min(30, max(10, min(w, h) * 3 // 10))
        new_w = w + padding * 2
        new_h = h + padding * 2
        padded = Image.new('RGB', (new_w, new_h), (255, 255, 255))
        padded.paste(img, (padding, padding))
        return padded

    @staticmethod
    def _upscale_if_small(img: Image.Image, min_side: int = 80) -> Image.Image:
        """小图（短边 < min_side）2x 放大，防止 VLM 对极小文字图片幻觉
        （实测 89×26 "投保年龄" 1x 幻觉为随机人名/乱码/提示词，2x 后 100% 正确）。
        全量扫描 122 个小 text region：观测到的幻觉短边均 ≤41px，≥80px 的全部正确。"""
        w, h = img.size
        if min(w, h) < min_side:
            return img.resize((w * 2, h * 2), Image.LANCZOS)
        return img

    def _process_text_region(self, region: Region,
                              save_dir: Optional[str]) -> Tuple[str, str]:
        """
        text 区域直接调 API。
        Returns: (cleaned_md, raw_response)
        """
        # 小图放大（短边<80px → 2x）+ 四周白色 padding，提高小文字识别、防幻觉
        padded_img = self._pad_white(self._upscale_if_small(region.image))
        resp = self.api.call_pil_image(padded_img) or ''
        cleaned = self._strip_html_tags(self._clean_response(resp))

        if save_dir:
            # 保存 padded 后的图片供调试
            pw, ph = padded_img.size
            fname_padded = f"r{region.order:02d}_text_padded_{pw}x{ph}.jpg"
            padded_img.save(os.path.join(save_dir, fname_padded), quality=85)
            fname = f"r{region.order:02d}_text_api.txt"
            with open(os.path.join(save_dir, fname), 'w', encoding='utf-8') as f:
                f.write(resp)

        return cleaned, resp

    # ==================== table 区域 ====================

    def _process_table_region(self, region: Region,
                               save_dir: Optional[str]) -> Tuple[str, List[Dict]]:
        """
        table 区域：Level 2 切块 + 并发 API（实时保存）+ 合并
        Returns: (merged_markdown, chunk_details)
            chunk_details: list of {chunk_meta 字段 + 'image_file': str, 'response': str}
        """
        # 假表格过滤：高度 < 110px 的 table 直接当 text 处理
        # （110 经 B 榜验证：标题框如 34821e6c r0=100px 被过滤，最近真表 128px 不受影响）
        # 注：曾试过 200，但导致 34727fa6(132/133) 和 ca08e055(128) 的真表被误判→已回退
        if region.height < 110:
            logger.info(f"    假表格过滤: order={region.order} "
                        f"{region.width}x{region.height} → 当 text 处理")
            # 小图放大 + 四周加白色 padding 提高识别准确率
            padded_img = self._pad_white(self._upscale_if_small(region.image))
            resp = self.api.call_pil_image(padded_img) or ''
            cleaned = self._strip_html_tags(self._clean_response(resp))
            if save_dir:
                pw, ph = padded_img.size
                fname_padded = f"r{region.order:02d}_fake_table_padded_{pw}x{ph}.jpg"
                padded_img.save(os.path.join(save_dir, fname_padded), quality=85)
                fname = f"r{region.order:02d}_text_api.txt"
                with open(os.path.join(save_dir, fname), 'w', encoding='utf-8') as f:
                    f.write(resp)
            return cleaned, []

        # Level 2 切块（内部含 Level 1.5 子表切分：多阶梯子表按边界分行组）+ API + 合并
        table_dir = (os.path.join(save_dir, f'table_{region.order:02d}')
                     if save_dir else None)
        logger.info(f"    table order={region.order} "
                    f"{region.width}x{region.height}")
        return self._process_one_table_image(
            region.image, table_dir, f"table_{region.order:02d}")

    def _process_one_table_image(self, table_img: Image.Image,
                                  table_dir: Optional[str],
                                  rel_prefix: str) -> Tuple[str, List[Dict]]:
        """处理单个表格图（整表或子表）：Level 2 切块 + 并发 API + merge_chunk_results。

        Args:
            table_img: 待处理表格图
            table_dir: chunk/中间产物保存目录（None 不保存）
            rel_prefix: chunk_details['image_file'] 的相对路径前缀
        Returns:
            (merged_markdown, chunk_details)
        """
        chunker = TableChunker(auto_granularity=True)
        chunks = chunker.chunk_table(table_img)

        if table_dir:
            os.makedirs(table_dir, exist_ok=True)
            self._save_chunks_before_api(chunks, table_dir, source_img=table_img)

        # 并发调 API，每完成一个立即保存
        responses = self._call_chunks_realtime(chunks, table_dir)

        # 【方案B】banner 表头分离重建（门控 + 命中 banner 才触发 + 失败回退正常合并）
        if self.enable_scheme_b:
            try:
                banner = self._detect_banner_from_chunks(chunks, responses)
                if not banner:
                    # 兜底：banner 文字不在首列组（居中/靠右）→ 全宽表头 strip 重读检测
                    banner = self._detect_banner_via_strip(table_img, chunker, chunks)
                if banner:
                    mb = self._scheme_b_reconstruct(table_img, chunker, banner, table_dir)
                    if mb:
                        logger.info(f"    [方案B] {rel_prefix} 命中 banner 表头 "
                                    f"(N_LABEL={banner[0]}, N_HEADER={banner[1]}, banner='{banner[2][:8]}') → 分离重建")
                        if table_dir:
                            with open(os.path.join(table_dir, 'merged.md'), 'w', encoding='utf-8') as f:
                                f.write(mb)
                        return mb, self._build_chunk_details(chunks, responses, rel_prefix)
            except Exception as e:
                logger.warning(f"    [方案B] {rel_prefix} 重建失败，回退正常合并: {e}")

        # 合并
        pairs = list(zip(chunks, responses))
        merged = merge_chunk_results(pairs)

        # 实时保存 merged.md
        if table_dir:
            with open(os.path.join(table_dir, 'merged.md'), 'w', encoding='utf-8') as f:
                f.write(merged)

        return merged, self._build_chunk_details(chunks, responses, rel_prefix)

    def _build_chunk_details(self, chunks: List[Dict], responses: List[str],
                              rel_prefix: str) -> List[Dict]:
        """收集 chunk 详情供 HTML report 使用。"""
        chunk_details = []
        for c, resp in zip(chunks, responses):
            cw, ch = c['image'].size
            base = self._chunk_basename(c)
            chunk_details.append({
                'order': c['order'],
                'kind': c.get('kind', 'body'),
                'row_range': list(c['row_range']),
                'col_range': list(c['col_range']),
                'bbox': list(c['bbox']),
                'size': [cw, ch],
                'n_header_rows': c.get('n_header_rows'),
                'n_first_cols': c.get('n_first_cols'),
                'image_file': f"{rel_prefix}/{base}_{cw}x{ch}.jpg",
                'response': resp or '',
            })
        return chunk_details

    # ==================== 方案B：banner 表头分离重建 ====================

    @staticmethod
    def _row0_cells_with_span(resp: str):
        """解析响应首行各单元格 (colspan, rowspan, text)。"""
        clean = Pipeline._clean_response(resp or '')
        m = re.search(r'<tr[^>]*>(.*?)</tr>', clean, re.S | re.I)
        if not m:
            return []
        out = []
        for cm in re.finditer(r'<t[dh]([^>]*)>(.*?)</t[dh]>', m.group(1), re.S | re.I):
            a = cm.group(1)
            txt = re.sub(r'<[^>]+>', '', cm.group(2)).strip()
            cs = re.search(r'colspan\s*=\s*"?(\d+)', a, re.I)
            rs = re.search(r'rowspan\s*=\s*"?(\d+)', a, re.I)
            out.append((int(cs.group(1)) if cs else 1, int(rs.group(1)) if rs else 1, txt))
        return out

    def _detect_banner_from_chunks(self, chunks: List[Dict], responses: List[str]):
        """从首行组各列组 API 结果检测 banner 表头模式（高精度：STRONG+SYMPTOM 双信号）。
        触发条件：≥2 列组 且 锚点(首列组) row0 含 colspan>=2 banner(其前有标签列)
                  且 存在其他列组行数 < 锚点行数（顶部 banner 行被丢）。
        返回 (n_label, n_header, banner_text) 或 None。"""
        from collections import defaultdict
        band = defaultdict(list)
        for c, r in zip(chunks, responses):
            if c.get('kind') == 'top_extra':
                continue
            band[c['row_range'][0]].append((c['col_range'][0], r))
        if not band:
            return None
        segs = sorted(band[min(band)], key=lambda x: x[0])
        if len(segs) < 2:
            return None
        anchor = segs[0][1]
        cells0 = self._row0_cells_with_span(anchor)
        banner_idx = next((i for i, (cs, rs, tx) in enumerate(cells0) if cs >= 2), None)
        if banner_idx is None or banner_idx < 1:
            return None
        # 【防误触发】真 banner 为纯文字标题（保单年度末…）不含数字；“年龄当列头”表(如 88a0acf1)
        #   会把含数字的年龄值(如“出生滩30日…”)误标 colspan 当 banner → 排除。
        if re.search(r'\d', cells0[banner_idx][2] or ''):
            return None

        # 症状：其余列组 row0 丢失 banner（首行无 colspan>=2）——顶部 banner 行被丢。
        # 【关键】不能用"行数 < 锤点"判断：传入的 responses 是后处理结果，漏行已被
        # 补空到 n_rows（如 186ee68f 其余组 raw=1 行被补到 12 行），会掩盖丢行症状；
        # 改判 row0 是否仍含 banner（补行只在尾部加空行，row0 不变，不受影响）。
        other_lost = any(
            not any(cs >= 2 for cs, _rs, _tx in self._row0_cells_with_span(r))
            for _, r in segs[1:]
        )
        if not other_lost:                            # 其余组都保留了 banner → 无需重建
            return None
        rss = [cells0[i][1] for i in range(banner_idx)]
        n_header = max(rss) if rss and max(rss) >= 2 else 2
        return (banner_idx, n_header, cells0[banner_idx][2], None)

    def _detect_banner_via_strip(self, table_img: Image.Image, chunker, chunks: List[Dict]):
        """兜底检测：banner 文字居中/靠右、不在首列组时（首列组 API 无 colspan），
        裁\"全宽 × 顶部 n_header 行\"表头 strip 单独重读，从中提取 标签列 + banner。
        返回 (n_label, n_header, banner_text, label_cells) 或 None。"""
        from collections import defaultdict
        band = defaultdict(list)
        for c in chunks:
            if c.get('kind') == 'top_extra':
                continue
            band[c['row_range'][0]].append(c['col_range'][0])
        if not band or len(band[min(band)]) < 2:      # 仅多列组表触发
            return None
        st = chunker.analyze_structure(table_img)
        rb = st.get('row_bounds') or []
        cb = st.get('col_bounds') or []
        n_header = 2                                   # banner 表头均为 2 行（banner 行 + 子表头行）
        if len(rb) <= n_header + 2 or len(cb) < 4:
            return None
        W, H = table_img.size
        hy1 = int(rb[n_header][0])
        if hy1 < 40 or hy1 > H * 0.3:                  # 太薄(非2行表头)/太厚(含数据行) → 跳过
            return None
        if W / max(1, hy1) > 180:                      # 条形 strip 过宽（长宽比>180）→ API 会报 aspect ratio 错，跳过
            return None
        resp = self._call_single_image(table_img.crop((0, 0, W, hy1)))
        rows = Pipeline._extract_tr_cells(resp)
        if len(rows) < 2:
            return None

        def _texts(row):
            return [re.sub(r'<[^>]+>', '', c).strip() for c in row]
        row0 = _texts(rows[0])
        # 结构判据（不依赖 VLM 是否标 colspan / 是否把子表头拆成多格，二者都不稳定）：
        #   banner 表头 row0 = [标签列… , banner 标题]，单元格少且全为非数字文本，
        #   而总列数很多（banner/标签各自横跨多列）。banner = row0 最后一个。
        if len(row0) < 2 or len(row0) >= len(cb) * 0.5:
            return None
        # 【防误触发】真 banner/标签均为纯文字标题（保单年度末/交费期间/被保险人投保年龄…），
        #   不含数字；“年龄当列头”表(如 88a0acf1)会把含数字的年龄值(如“出生滩30日…”)误当 banner → 排除。
        if any(re.search(r'\d', t) for t in row0 if t):
            return None
        banner = row0[-1]
        if not banner or banner.isdigit():
            return None
        label_cells = row0[:-1]
        return (len(label_cells), n_header, banner, label_cells)

    def _reconstruct_banner_header(self, img: Image.Image, rb, cb,
                                    n_label: int, n_header: int, group: int = 10,
                                    banner_text: Optional[str] = None,
                                    label_cells: Optional[List[str]] = None):
        """重建 banner 表头：首组拼标签取 labels+banner；其余组不拼标签只取子表头数字。
        label_cells 提供时（strip 兜底路径）直接用它 + banner_text（banner 居中、首组读不到）。
        返回 (header_tr_html, n_data) 或 None。"""
        n_data = len(cb) - n_label
        if n_data < 2 or n_header < 1 or n_header >= len(rb):
            return None
        hy0, hy1 = int(rb[0][0]), int(rb[n_header][0])
        lx0, lx1 = int(cb[0][0]), int(cb[n_label][0])
        label = img.crop((lx0, hy0, lx1, hy1))
        data_idx = list(range(n_label, len(cb)))
        groups = [data_idx[i:i + group] for i in range(0, len(data_idx), group)]

        def nums(resp):
            txt = re.sub(r'<[^>]+>', ' ', Pipeline._clean_response(resp))
            return re.findall(r'(?<!\d)(\d{1,3})(?!\d)', txt)

        gx0, gx1 = int(cb[groups[0][0]][0]), int(cb[groups[0][-1]][1])
        comp = Image.new('RGB', (label.width + (gx1 - gx0), hy1 - hy0), 'white')
        comp.paste(label, (0, 0))
        comp.paste(img.crop((gx0, hy0, gx1, hy1)), (label.width, 0))
        r0 = self._call_single_image(comp)
        cells = Pipeline._extract_tr_cells(r0)
        row0 = [re.sub(r'<[^>]+>', '', c).strip() for c in cells[0]] if cells else []
        if label_cells is not None:
            # strip 兜底路径：标签 + banner 来自全宽 strip（首列组读不到居中 banner）
            lc = (list(label_cells) + [''] * n_label)[:n_label]
            banner = banner_text or ''
        else:
            # 快路：标签 + banner 来自首组读取（banner 在首列组，原行为不变）
            lc = (row0[:n_label] + [''] * n_label)[:n_label]
            banner = row0[n_label] if len(row0) > n_label else ''
        sub = nums(r0)
        for g in groups[1:]:
            gx0, gx1 = int(cb[g[0]][0]), int(cb[g[-1]][1])
            sub += nums(self._call_single_image(img.crop((gx0, hy0, gx1, hy1))))
        sub = sub[:n_data]
        if len(sub) < n_data * 0.6:                  # 子表头数字太少 → 重建不可靠
            return None
        lab = ''.join(f'<td rowspan="{n_header}">{c}</td>' for c in lc)
        row0h = f'<tr>{lab}<td colspan="{len(sub)}">{banner}</td></tr>'
        row1h = '<tr>' + ''.join(f'<td>{n}</td>' for n in sub) + '</tr>'
        return row0h + row1h, len(sub)

    def _scheme_b_reconstruct(self, table_img: Image.Image, chunker,
                               banner, table_dir: Optional[str]):
        """方案B：表头带分离重建 + 主体正常切 + 拼接。成功返回 markdown，否则 None。"""
        n_label, n_header, banner_text, label_cells = banner
        st = chunker.analyze_structure(table_img)
        rb = st.get('row_bounds') or []
        cb = st.get('col_bounds') or []
        if len(rb) <= n_header + 2 or len(cb) <= n_label + 2:
            return None
        rec = self._reconstruct_banner_header(table_img, rb, cb, n_label, n_header,
                                              banner_text=banner_text, label_cells=label_cells)
        if not rec:
            return None
        header_html, _ = rec
        W, H = table_img.size
        hy1 = int(rb[n_header][0])
        body_img = table_img.crop((0, hy1, W, H))
        body_chunker = TableChunker(auto_granularity=True)
        body_chunks = body_chunker.chunk_table(body_img)
        for c in body_chunks:                        # 主体无表头 → 禁止 merge 剥离表头行
            c['has_header'] = False
            c['n_header_rows'] = 0
        body_dir = os.path.join(table_dir, 'scheme_b_body') if table_dir else None
        if body_dir:
            os.makedirs(body_dir, exist_ok=True)
            self._save_chunks_before_api(body_chunks, body_dir, source_img=body_img)
        body_resps = self._call_chunks_realtime(body_chunks, body_dir)
        body_merged = merge_chunk_results(list(zip(body_chunks, body_resps)))
        body_trs = re.findall(r'<tr[^>]*>.*?</tr>', body_merged, re.S | re.I)
        if not body_trs:
            return None
        final = '<table border="1">' + header_html + ''.join(body_trs) + '</table>'
        return self._normalize_table_tags(final)

    # ==================== chunk 保存 & API 实时调用 ====================

    def _save_chunks_before_api(self, chunks: List[Dict], table_dir: str,
                                  source_img: Optional[Image.Image] = None):
        """API 调用前立即保存所有 chunk 图 + source.jpg + chunk_meta.json"""
        if source_img is not None:
            try:
                source_img.save(os.path.join(table_dir, 'source.jpg'), quality=90)
            except Exception as e:
                logger.warning(f"保存 source.jpg 失败: {e}")

        metas = []
        for c in chunks:
            cw, ch = c['image'].size
            base = self._chunk_basename(c)
            fname = f"{base}_{cw}x{ch}.jpg"
            c['image'].save(os.path.join(table_dir, fname), quality=85)

            m = {k: v for k, v in c.items() if k != 'image'}
            m['image_size'] = [cw, ch]
            m['image_file'] = fname
            metas.append(m)

        meta = {'chunks': metas}
        if chunks:
            c0 = chunks[0]
            meta['table_type'] = c0.get('table_type')
            meta['structure_rows'] = c0.get('structure_rows')
            meta['structure_cols'] = c0.get('structure_cols')
            meta['n_header_rows'] = c0.get('n_header_rows')
            meta['n_first_cols'] = c0.get('n_first_cols')

        with open(os.path.join(table_dir, 'chunk_meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def _call_chunks_realtime(self, chunks: List[Dict],
                                save_dir: Optional[str]) -> List[str]:
        """并发调 API，每完成一个 chunk 立即保存对应的 `_api.txt`。
        
        含智能处理：
        1. 纯空 chunk 直接生成空单元格，不调 API
        2. API 返回后检查行列数，不足时用空单元格补全
        3. API 返回截断（table 不闭合）时，兜底生成空行
        """
        n = len(chunks)
        if n == 0:
            return []
        results: List[str] = [''] * n

        def _resp_path(chunk: Dict) -> str:
            base = self._chunk_basename(chunk)
            return os.path.join(save_dir, f"{base}_api.txt")

        def _process_chunk(idx: int) -> str:
            chunk = chunks[idx]
            n_rows = chunk['row_range'][1] - chunk['row_range'][0] + 1
            n_cols = chunk['col_range'][1] - chunk['col_range'][0] + 1
            chunk_tag = f"chunk#{chunk['order']:03d} r{chunk['row_range'][0]}-{chunk['row_range'][1]} c{chunk['col_range'][0]}-{chunk['col_range'][1]}"
            cw, ch = chunk['image'].size

            # 0. 上游纯空判定：排除网格线后暗像素占比 < 0.02% → 直接空表，不调 API
            # （0.02% 严格阈值经 B榜 4396 chunks 验证：单个数字如"961.61"≈0.06%即会保留）
            is_empty, dark_ratio = self._is_empty_chunk_debug(chunk['image'], threshold=0.0002)
            if is_empty:
                resp = self._generate_empty_table(n_rows, n_cols)
                logger.info(f"    [纯空跳过] {chunk_tag} {cw}x{ch} dark={dark_ratio*100:.3f}% → 生成 {n_rows}×{n_cols} 空表")
                return resp

            # 1. 密集 chunk 自动 2x 放大（每格 <25×15px 时字体太小，API 识别不准）
            work_img = chunk['image']
            cell_w = cw / n_cols
            cell_h = ch / n_rows
            zoomed = False
            if cell_w < 25 or cell_h < 15:
                new_size = (cw * 2, ch * 2)
                work_img = chunk['image'].resize(new_size, Image.LANCZOS)
                zoomed = True
                logger.info(f"    [密集放大2x] {chunk_tag} {cw}x{ch} → {new_size[0]}x{new_size[1]} (每格 {cell_h:.1f}x{cell_w:.1f}px)")

            # 2. Exx 空单元格检测+填充（在放大后的图上做，保证每格空间足够）
            api_img = work_img
            if self.enable_em_fill:
                filled_img, n_exx, filled_positions = self._fill_empty_cells_exx(
                    work_img, known_rows=n_rows, known_cols=n_cols)
                if n_exx == -1:
                    # 全空（自适应检测）→ 不调 API
                    resp = self._generate_empty_table(n_rows, n_cols)
                    logger.info(f"    [Exx全空] {chunk_tag} → 生成 {n_rows}×{n_cols} 空表")
                    return resp
                elif n_exx > 0:
                    api_img = filled_img
                    logger.info(f"    [Exx填充] {chunk_tag} 填了{n_exx}个空格{' (放大后)' if zoomed else ''}")
                    # 保存填充图供调试
                    if save_dir:
                        base = self._chunk_basename(chunk)
                        filled_img.save(os.path.join(save_dir, f"{base}_filled.jpg"), quality=92)
            else:
                n_exx = 0  # 禁用填充

            # 2. 调 API（失败时重试一次）
            resp = self.api.call_pil_image(api_img) or ''
            if not resp or len(resp) < 30:
                logger.warning(f"    [API重试] {chunk_tag} 返回空/过短(len={len(resp)})，重试中...")
                import time
                time.sleep(2)
                resp = self.api.call_pil_image(api_img) or ''

            # 保存原始 API 返回（后处理之前），便于区分哪些 chunk 做了后处理
            if save_dir:
                try:
                    raw_path = os.path.join(save_dir, f"{self._chunk_basename(chunk)}_raw.txt")
                    with open(raw_path, 'w', encoding='utf-8') as f:
                        f.write(resp)
                except Exception as e:
                    logger.warning(f"保存 raw 失败: {e}")

            # 3. 去除占位符（天干字符）
            if n_exx > 0:
                resp = self._strip_exx_labels(resp)

            # 【问题6-新增】上方文字词数==表列数 → 先并入表头行（确定性、免 API）；
            #   这样"表头行被甩到表上方"就不会被下面误判为漏行/漏列而触发切分重调。
            resp = self._pull_above_text_as_row(resp)

            # 【问题4】原生结果行列不齐/漏列(API 误读) 或 纯文本退化(密集数字中间组) → 沿竖线中点左右切分、各半重调、按行合并；
            #          仅当重调结果比原始更整齐才采纳（否则保留原始，走后处理兜底）。
            if self.enable_resplit and (self._is_ragged_response(resp, n_cols)
                                        or self._is_degenerate_text_response(resp, n_rows, n_cols)):
                resplit = self._resplit_and_recall(chunk['image'], n_rows, n_cols, chunk_tag)
                if resplit and self._ragged_score(resplit, n_cols) < self._ragged_score(resp, n_cols):
                    logger.info(f"    [左右切分重调] {chunk_tag} 原生行列不齐/纯文本退化 → 切分重调后更整齐，采纳")
                    resp = resplit

            # 【问题4-行】原生"均匀但少行"(API 漏读整行) → 沿横线中点上下切分、各半重调、纵向堆叠；
            #          仅当重调后行数更接近期望才采纳（否则保留原始，走后处理兜底）。
            if self.enable_resplit_row and self._is_underrow_response(resp, n_rows, n_cols):
                resplit_v = self._resplit_and_recall_vertical(chunk['image'], n_rows, n_cols, chunk_tag)
                if resplit_v and self._row_score(resplit_v, n_rows) < self._row_score(resp, n_rows):
                    logger.info(f"    [上下切分重调] {chunk_tag} 原生漏行 → 切分重调后行数更全，采纳")
                    resp = resplit_v

            # 4. 后处理：检查行列数 + 截断兜底
            resp = self._postprocess_chunk_response(resp, n_rows, n_cols, chunk_tag=chunk_tag)
            return resp

        if n == 1:
            results[0] = _process_chunk(0)
            if save_dir:
                try:
                    with open(_resp_path(chunks[0]), 'w', encoding='utf-8') as f:
                        f.write(results[0])
                except Exception as e:
                    logger.warning(f"保存 chunk#{chunks[0]['order']} 失败: {e}")
            return results

        workers = min(self.chunk_workers, n)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(_process_chunk, idx): idx
                for idx in range(n)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    resp = future.result()
                except Exception as e:
                    logger.error(f"chunk#{chunks[idx]['order']} 处理失败: {e}")
                    # 兜底：生成空表
                    n_rows = chunks[idx]['row_range'][1] - chunks[idx]['row_range'][0] + 1
                    n_cols = chunks[idx]['col_range'][1] - chunks[idx]['col_range'][0] + 1
                    resp = self._generate_empty_table(n_rows, n_cols)
                results[idx] = resp or ''

                if save_dir:
                    try:
                        with open(_resp_path(chunks[idx]), 'w', encoding='utf-8') as f:
                            f.write(results[idx])
                    except Exception as e:
                        logger.warning(f"保存 chunk#{chunks[idx]['order']} 失败: {e}")

        return results

    # ==================== 【问题4】行列不齐 → 左右切分重调 ====================

    @staticmethod
    def _native_col_counts(resp: str):
        """解析响应第一个 table 每行的原生 td/th 数（不补齐）。返回 (counts, has_span)。"""
        cleaned = Pipeline._clean_response(resp)
        if '<table' not in cleaned.lower():
            return [], False
        has_span = bool(re.search(r'\b(colspan|rowspan)\b', cleaned, re.IGNORECASE))
        trs = re.findall(r'<tr[^>]*>(.*?)</tr>', cleaned, re.DOTALL | re.IGNORECASE)
        counts = [len(re.findall(r'<t[dh][^>]*>', tr, re.IGNORECASE)) for tr in trs]
        counts = [c for c in counts if c > 0]
        return counts, has_span

    def _is_ragged_response(self, resp: str, n_cols: int) -> bool:
        """判定 chunk 结果需要"左右切分重调"：
          (a) 某行列数 > 期望（过读）；
          (b) 各行极差 ≥ 2（行列不齐）；
          (c) 【新增】均匀/近均匀（极差≤1）但整体少列（max < 期望列数）——经典漏列，
              如期望 11 列各行只返回 10 列。
        含 colspan/rowspan（结构化表头）或列数<4（无法左右切）时不触发。"""
        if n_cols < 4:
            return False
        counts, has_span = self._native_col_counts(resp)
        if has_span or len(counts) < 2:
            return False
        if max(counts) > n_cols or (max(counts) - min(counts)) >= 2:
            return True
        # 新增：均匀但少列（此时已知 max<=n_cols 且极差<=1）→ 经典漏列
        return max(counts) < n_cols

    def _is_degenerate_text_response(self, resp: str, n_rows: int, n_cols: int) -> bool:
        """判定 chunk 结果为"纯文本退化"：VLM 在密集纯数字中间组上无法维持二维列对齐，
        退化为纯文本流（无 <table>），典型两态：
          - 列坍塌：每行只吐 1~2 个数字（远少于期望列数），如 12 列却每行 2 个；
          - 溢出/循环：行数远超期望行数（如 12 行输出 51/1024 行），或大量重复循环。
        这类响应 `_native_col_counts` 恒为 0 → 既有 ragged/underrow 判据全部失效，需专门识别，
        命中后走列向 resplit（切窄后 VLM 可正确识别）。n_cols<4（无法左右切）时不触发。"""
        if n_cols < 4:
            return False
        cleaned = self._clean_response(resp)
        if '<table' in cleaned.lower():
            return False                      # 有表结构 → 交给 ragged/underrow 判据
        lines = [ln.strip() for ln in cleaned.split('\n') if ln.strip()]
        # 去 markdown 表格分隔行（|---|---|）
        lines = [ln for ln in lines if not re.fullmatch(r'[\|\s:\-]+', ln)]
        if len(lines) < 2:
            return False

        def _toks(ln):
            parts = ln.strip('|').split('|') if '|' in ln else ln.split()
            return [c for c in parts if c.strip()]
        tok_counts = sorted(len(_toks(ln)) for ln in lines)
        median_tok = tok_counts[len(tok_counts) // 2]
        overflow = len(lines) >= n_rows + 2                       # 行数溢出（流式漂移/循环）
        looped = len(lines) >= 20 and len(set(lines)) / len(lines) < 0.7  # 重复循环退化
        collapsed = median_tok * 2 <= n_cols                      # 每行 token 数 ≤ 期望列数一半 → 列坍塌
        return looped or (collapsed and (overflow or median_tok <= 2))

    def _ragged_score(self, resp: str, n_cols: int) -> int:
        """不齐程度打分（越小越整齐）：极差×10 + 与期望列数偏差。用于比较重调前后。"""
        counts, _ = self._native_col_counts(resp)
        if not counts:
            return 999
        spread = max(counts) - min(counts)
        dev = abs(max(counts) - n_cols) + abs(min(counts) - n_cols)
        return spread * 10 + dev

    def _call_single_image(self, img: Image.Image) -> str:
        """对单张(半)图调 API（尺寸偏小则 2x 放大提升识别），返回原始响应。"""
        w, h = img.size
        work = img
        if w < 500 or h < 300:
            work = img.resize((w * 2, h * 2), Image.LANCZOS)
        return self.api.call_pil_image(work) or ''

    @staticmethod
    def _extract_tr_cells(resp: str):
        """解析响应第一个 table 每行的单元格 HTML 列表：[[<td>..</td>,...], ...]。"""
        cleaned = Pipeline._clean_response(resp)
        m = re.search(r'<table.*?</table>', cleaned, re.DOTALL | re.IGNORECASE)
        body = m.group(0) if m else cleaned
        rows = []
        for tr in re.findall(r'<tr[^>]*>(.*?)</tr>', body, re.DOTALL | re.IGNORECASE):
            cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', tr, re.DOTALL | re.IGNORECASE)
            if cells:
                rows.append(cells)
        return rows

    def _resplit_and_recall(self, chunk_img: Image.Image, n_rows: int, n_cols: int,
                             chunk_tag: str = '') -> Optional[str]:
        """把 chunk 图沿"最接近宽度中点的竖线"切左右两半，各自调 API，按行索引横向合并。
        返回合并 <table> HTML；无法切分/结果为空时返回 None。"""
        import numpy as np
        w, h = chunk_img.size
        if w < 40 or n_cols < 4:
            return None
        gray = np.array(chunk_img.convert('L'), dtype=np.uint8)
        col_dark = (gray < 128).sum(axis=0) / max(1, h)
        # 检测竖线（列分隔线）：整列暗占比>0.5；连续暗列合成一条，取其中心。
        is_line = col_dark > 0.5
        seps, x = [], 6
        while x < w - 6:
            if is_line[x]:
                x0 = x
                while x < w - 6 and is_line[x]:
                    x += 1
                seps.append((x0 + x - 1) // 2)
            else:
                x += 1
        # 目标：左右两半列数尽量相等 → 取"按列数居中"的那条分隔线（seps 已按 x 升序）。
        if len(seps) >= 3:
            split_x = seps[len(seps) // 2]
        elif seps:
            split_x = min(seps, key=lambda s: abs(s - w // 2))
        else:
            split_x = w // 2
        # 护栏：切点须落在 [30%,70%] 宽度内（避免切得太偏成"1 列 vs 其余"）；
        # 越界则在该区间内挑最接近中点的分隔线，没有则退回几何中点。
        if not (0.30 * w <= split_x <= 0.70 * w):
            inrange = [s for s in seps if 0.30 * w <= s <= 0.70 * w]
            split_x = min(inrange, key=lambda s: abs(s - w // 2)) if inrange else w // 2
        if split_x <= 5 or split_x >= w - 5:
            return None
        left = chunk_img.crop((0, 0, split_x, h))
        right = chunk_img.crop((split_x, 0, w, h))
        lrows = self._extract_tr_cells(self._call_single_image(left))
        rrows = self._extract_tr_cells(self._call_single_image(right))
        if not lrows or not rrows:
            return None
        n = max(len(lrows), len(rrows))
        trs = []
        for i in range(n):
            lc = lrows[i] if i < len(lrows) else []
            rc = rrows[i] if i < len(rrows) else []
            trs.append('<tr>' + ''.join(lc + rc) + '</tr>')
        return '<table>' + ''.join(trs) + '</table>'

    # ==================== 【问题4-行】漏行 → 上下切分重调 ====================

    def _is_underrow_response(self, resp: str, n_rows: int, n_cols: int) -> bool:
        """判定 chunk 结果"漏行"（API 少读整行）：解析出的行数 < 期望行数。
        含 colspan/rowspan（结构化表头）或期望行数<6（不宜上下切）时不触发。"""
        if n_rows < 6:
            return False
        counts, has_span = self._native_col_counts(resp)
        if has_span or len(counts) < 2:
            return False
        return len(counts) < n_rows

    def _row_score(self, resp: str, n_rows: int) -> int:
        """上下切分重调的整齐度打分（越小越好）：|实际行数 - 期望行数|×10 + 列参差。
        既要行数接近期望，又不能把列切碎。用于比较重调前后。"""
        counts, _ = self._native_col_counts(resp)
        if not counts:
            return 999
        row_dev = abs(len(counts) - n_rows)
        col_spread = max(counts) - min(counts)
        return row_dev * 10 + col_spread

    def _resplit_and_recall_vertical(self, chunk_img: Image.Image, n_rows: int,
                                      n_cols: int, chunk_tag: str = '') -> Optional[str]:
        """把 chunk 图沿"按行数居中的横线"切上下两半，各自调 API，按顺序纵向堆叠合并。
        用于经典漏行（返回行数 < 期望）。返回合并 <table> HTML；无法切分/结果为空时返回 None。"""
        import numpy as np
        w, h = chunk_img.size
        if h < 40 or n_rows < 6:
            return None
        gray = np.array(chunk_img.convert('L'), dtype=np.uint8)
        row_dark = (gray < 128).sum(axis=1) / max(1, w)
        # 检测横线（行分隔线）：整行暗占比>0.5；连续暗行合成一条，取其中心。
        is_line = row_dark > 0.5
        seps, y = [], 6
        while y < h - 6:
            if is_line[y]:
                y0 = y
                while y < h - 6 and is_line[y]:
                    y += 1
                seps.append((y0 + y - 1) // 2)
            else:
                y += 1
        # 目标：上下两半行数尽量相等 → 取"按行数居中"的那条横线（seps 已按 y 升序）。
        if len(seps) >= 3:
            split_y = seps[len(seps) // 2]
        elif seps:
            split_y = min(seps, key=lambda s: abs(s - h // 2))
        else:
            split_y = h // 2
        # 护栏：切点须落在 [30%,70%] 高度内（避免切得太偏成"1 行 vs 其余"）；
        # 越界则在该区间内挑最接近中点的横线，没有则退回几何中点。
        if not (0.30 * h <= split_y <= 0.70 * h):
            inrange = [s for s in seps if 0.30 * h <= s <= 0.70 * h]
            split_y = min(inrange, key=lambda s: abs(s - h // 2)) if inrange else h // 2
        if split_y <= 5 or split_y >= h - 5:
            return None
        top = chunk_img.crop((0, 0, w, split_y))
        bottom = chunk_img.crop((0, split_y, w, h))
        trows = self._extract_tr_cells(self._call_single_image(top))
        brows = self._extract_tr_cells(self._call_single_image(bottom))
        if not trows or not brows:
            return None
        # 纵向堆叠：上半所有行 + 下半所有行（各行本身已是全宽，直接顺序拼接）
        trs = ['<tr>' + ''.join(row) + '</tr>' for row in (trows + brows)]
        return '<table>' + ''.join(trs) + '</table>'

    # ==================== chunk 智能处理 ====================

    @staticmethod
    def _is_empty_chunk(img: Image.Image, threshold: float = 0.005) -> bool:
        """
        检测 chunk 是否纯空（只有网格线，无文字内容）。

        策略：排除横线/竖线后，检查剩余暗像素占比。
        如果排除线条后暗像素 < threshold，认为纯空。
        """
        is_empty, _ = Pipeline._is_empty_chunk_debug(img, threshold)
        return is_empty

    @staticmethod
    def _is_empty_chunk_debug(img: Image.Image, threshold: float = 0.005) -> Tuple[bool, float]:
        """
        检测并返回具体的暗像素占比（供日志使用）。
        Returns: (is_empty, dark_ratio_after_line_removal)
        """
        import numpy as np
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        if h < 5 or w < 5:
            return True, 0.0

        dark = gray < 200
        row_dark_ratio = dark.sum(axis=1) / w
        line_rows = row_dark_ratio > 0.5
        col_dark_ratio = dark.sum(axis=0) / h
        line_cols = col_dark_ratio > 0.5

        mask = dark.copy()
        mask[line_rows, :] = False
        mask[:, line_cols] = False

        remaining_dark = mask.sum()
        total_pixels = h * w
        ratio = remaining_dark / total_pixels

        return ratio < threshold, float(ratio)

    # ==================== Exx 空单元格填充 ====================

    @staticmethod
    def _fill_empty_cells_exx(chunk_img, known_rows=None, known_cols=None) -> tuple:
        """检测chunk中的空单元格并填充随机Exx标签。
        返回 (filled_img, n_empty, positions)。n_empty=0 表示无需填充。

        网格策略：
        - 优先使用 known_rows × known_cols 构建均匀网格（权威真相）
        - 图像网格检测仅作最终 fallback（无 known 参数时）

        注：调用方需自行保证图像尺寸足够（推荐每格 >= 15×25px）；
            对密集小格 chunk，应在上游先放大再传入本函数。
        """
        import numpy as np
        from PIL import ImageDraw, ImageFont

        gray = np.array(chunk_img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        if known_rows and known_cols and known_rows > 0 and known_cols > 0:
            row_bounds = list(np.linspace(0, h, known_rows + 1, dtype=int))
            col_bounds = list(np.linspace(0, w, known_cols + 1, dtype=int))
            n_rows = known_rows
            n_cols = known_cols
        else:
            # 无 known 参数：fallback 到图像检测
            def _find_lines(g, axis):
                if axis == 'h':
                    darks = (g < 128).sum(axis=1)
                    th = g.shape[1] * 0.5
                else:
                    darks = (g < 128).sum(axis=0)
                    th = g.shape[0] * 0.5
                cands = np.where(darks > th)[0]
                if len(cands) == 0:
                    return []
                groups = []
                s = cands[0]
                for i in range(1, len(cands)):
                    if cands[i] - cands[i-1] > 3:
                        groups.append(int((s + cands[i-1]) / 2))
                        s = cands[i]
                groups.append(int((s + cands[-1]) / 2))
                return groups

            h_lines = _find_lines(gray, 'h')
            v_lines = _find_lines(gray, 'v')
            row_bounds = [0] + h_lines + [h]
            col_bounds = [0] + v_lines + [w]
            n_rows = len(row_bounds) - 1
            n_cols = len(col_bounds) - 1

        if n_rows <= 0 or n_cols <= 0:
            return chunk_img, 0, []

        # 计算每格 <200 暗像素占比
        ratios = [[0.0] * n_cols for _ in range(n_rows)]
        for r in range(n_rows):
            for c in range(n_cols):
                y0, y1 = row_bounds[r], row_bounds[r+1]
                x0, x1 = col_bounds[c], col_bounds[c+1]
                yy0, yy1 = y0 + 4, y1 - 4
                xx0, xx1 = x0 + 4, x1 - 4
                if yy1 <= yy0 or xx1 <= xx0:
                    continue
                cell = gray[yy0:yy1, xx0:xx1]
                ratios[r][c] = (cell < 250).mean() if cell.size > 0 else 0.0

        # 自适应阈值
        all_ratios = sorted(ratios[r][c] for r in range(n_rows) for c in range(n_cols))
        max_ratio = all_ratios[-1] if all_ratios else 0
        if max_ratio <= 0.03:
            # 纯空 chunk
            return chunk_img, -1, []  # -1 表示全空
        # 找跳变
        adaptive_th = 0.01
        max_gap = 0
        for i in range(len(all_ratios) - 1):
            lo, hi = all_ratios[i], all_ratios[i+1]
            if lo <= 0.02:
                gap = hi - lo
                if gap > max_gap:
                    max_gap = gap
                    adaptive_th = (lo + hi) / 2
        if max_gap < 0.003:
            adaptive_th = 0.01
        adaptive_th = min(adaptive_th, 0.01)

        empty = [[ratios[r][c] <= adaptive_th for c in range(n_cols)] for r in range(n_rows)]
        n_empty = sum(sum(row) for row in empty)
        if n_empty == 0:
            return chunk_img, 0, []

        # 填充中文占位符（纯天干字符，随机分布，防复读+防误读为数字）
        import random
        filled = chunk_img.copy()
        fd = ImageDraw.Draw(filled)
        empty_positions = [(r, c) for r in range(n_rows) for c in range(n_cols) if empty[r][c]]
        # 字母占位符（多样化防复读，不易被误读为数字）
        _PLACEHOLDERS = ["AB", "CD", "E", "H", "K", "MN"]
        labels = [random.choice(_PLACEHOLDERS) for _ in range(len(empty_positions))]

        # 字体
        import os
        font_path = None
        for fp in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                   "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]:
            if os.path.exists(fp):
                font_path = fp
                break

        for idx, (r, c) in enumerate(empty_positions):
            y0, y1 = row_bounds[r], row_bounds[r+1]
            x0, x1 = col_bounds[c], col_bounds[c+1]
            cell_h = y1 - y0
            if cell_h <= 4 or x1 - x0 <= 4:
                continue
            label = labels[idx]
            font_size = max(9, int(cell_h * 0.45))
            try:
                font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
            except:
                font = ImageFont.load_default()
            try:
                tb = fd.textbbox((0, 0), label, font=font)
                tw, th = tb[2]-tb[0], tb[3]-tb[1]
            except:
                tw, th = font_size, font_size
            tx = x0 + (x1 - x0 - tw) // 2
            ty = y0 + (y1 - y0 - th) // 2
            fd.text((tx, ty), label, fill=(0, 0, 0), font=font)

        return filled, n_empty, empty_positions

    @staticmethod
    def _strip_exx_labels(resp: str) -> str:
        """从API返回中去除占位符（AB/CD/E/H/K/MN，替换为空）。"""
        import re
        # 匹配独立的占位符（前后是标签边界）
        return re.sub(r'(?<=>)\s*(?:AB|CD|MN|[EHK])\s*(?=<)', '', resp, flags=re.IGNORECASE)


    @staticmethod
    def _generate_empty_table(n_rows: int, n_cols: int) -> str:
        """生成指定行列数的空表格 markdown。"""
        empty_row = '<tr>' + '<td></td>' * n_cols + '</tr>'
        rows = [empty_row] * n_rows
        return '<table>' + ''.join(rows) + '</table>'

    def _postprocess_chunk_response(self, resp: str, n_rows: int, n_cols: int,
                                      chunk_tag: str = '') -> str:
        """
        对 API 返回的单个 chunk 结果做后处理：
        1. 空响应兜底：直接用空表代替
        2. 截断兜底：table 不闭合时，直接用空表代替
        3. 行数不足时，用空行补全到 n_rows
        4. 行中列数不足时，每行末尾补空单元格到 n_cols
        """
        if not resp:
            logger.warning(f"    [空响应兜底] {chunk_tag} API 返回空 → 生成 {n_rows}×{n_cols} 空表")
            return self._generate_empty_table(n_rows, n_cols)

        # 清理 markdown 代码块包裹
        cleaned = self._clean_response(resp)

        # 【问题6-新增】上方文字词数==表格列数 → 先作为表格第一行并入（确定性，在补行前做，
        #   这样后续行数统计已含表头行、不会被误补空行）。
        cleaned = self._pull_above_text_as_row(cleaned)

        # 【问题6】捕获 <table> 上方的短文字标签（如"投保年龄"），稍后并入首行首列
        above_text = ''
        low = cleaned.lower()
        if '<table' in low:
            tbl_pos = low.find('<table')
            pre = cleaned[:tbl_pos]
            pre = re.sub(r'</?[a-zA-Z][^>]*>', '', pre)   # 去残留标签
            pre = re.sub(r'[#*`>|]+', ' ', pre)            # 去 markdown 噪声符
            pre_lines = [ln.strip() for ln in pre.split('\n') if ln.strip()]
            if pre_lines:
                cand = pre_lines[-1]  # 取最接近表格的一行
                # 仅当是"短标签"（无空白、长度合理）才并入，避免误吞正文/长句；
                # 且排除代码块围栏残留词（markdown/html/md 等），避免把 "```markdown" 并进首格
                _fence_words = {'markdown', 'html', 'md', 'text', 'json', 'xml', 'csv'}
                if (0 < len(cand) <= 15 and not re.search(r'\s', cand)
                        and cand.lower() not in _fence_words):
                    above_text = cand
                    cleaned = cleaned[tbl_pos:]  # 去掉表格上方文字

        # 【问题2/5】非 HTML 兜底：API 返回无 <table>（纯文本/markdown）→ 按 n_rows×n_cols 重建
        if '<table' not in cleaned.lower():
            rebuilt = self._rebuild_table_from_text(cleaned, n_rows, n_cols)
            if rebuilt:
                logger.info(f"    [非HTML重建] {chunk_tag} 无<table> → 按 {n_rows}×{n_cols} 重建")
                cleaned = rebuilt
            else:
                logger.warning(f"    [非HTML兜底] {chunk_tag} 无<table>且无法解析 → 生成 {n_rows}×{n_cols} 空表")
                return self._generate_empty_table(n_rows, n_cols)

        # 检查是否截断（table 不闭合）
        if '<table' in cleaned.lower() and '</table>' not in cleaned.lower():
            # 截取已有数据到 n_rows×n_cols，补上 </table>
            import re as _re
            trs = _re.findall(r'<tr[^>]*>.*?</tr>', cleaned, _re.DOTALL | _re.IGNORECASE)
            if trs:
                # 截取前 n_rows 行
                trs = trs[:n_rows]
                # 每行截取前 n_cols 个 td/th
                trimmed_trs = []
                for tr in trs:
                    cells = _re.findall(r'<t[dh][^>]*>.*?</t[dh]>', tr, _re.DOTALL | _re.IGNORECASE)
                    cells = cells[:n_cols]
                    # 补不足的列
                    while len(cells) < n_cols:
                        cells.append('<td></td>')
                    trimmed_trs.append('<tr>' + ''.join(cells) + '</tr>')
                result = '<table>' + ''.join(trimmed_trs) + '</table>'
                logger.warning(f"    [截断修复] {chunk_tag} </table>缺失，截取{len(trimmed_trs)}行×{n_cols}列")
                cleaned = result
            else:
                logger.warning(f"    [截断兜底] {chunk_tag} 返回长度={len(resp)} 但无有效<tr> → 生成空表")
                return self._generate_empty_table(n_rows, n_cols)

        # 统计实际行数（数 <tr 标签，排除 thead 中的）
        # 移除 thead 部分再数 tr
        body_part = cleaned
        thead_match = re.search(r'<thead>.*?</thead>', cleaned, re.DOTALL | re.IGNORECASE)
        if thead_match:
            body_part = cleaned[thead_match.end():]
        tr_matches = list(re.finditer(r'<tr[^>]*>(.*?)</tr>', body_part, re.DOTALL | re.IGNORECASE))
        # 总行数 = thead 中的 tr + body 中的 tr（thead 行也属于 chunk 的一行）
        thead_tr_count = 0
        if thead_match:
            thead_tr_count = len(re.findall(r'<tr[^>]*>', thead_match.group(0), re.IGNORECASE))
        actual_rows = len(tr_matches) + thead_tr_count

        # 检查列数（每行 td/th 个数）
        row_col_counts = []
        for tr_m in tr_matches:
            tr_content = tr_m.group(1)
            n_td = len(re.findall(r'<t[dh][^>]*>', tr_content, re.IGNORECASE))
            row_col_counts.append(n_td)

        # 行列不匹配日志
        actual_cols_summary = ''
        if row_col_counts:
            min_cols = min(row_col_counts)
            max_cols = max(row_col_counts)
            actual_cols_summary = f"{min_cols}~{max_cols}" if min_cols != max_cols else str(min_cols)
        else:
            actual_cols_summary = '0'

        # 行数判断：缺 1 行及以上就补尾部空行
        # （原 min(n_rows*0.7, n_rows-3) 取小值过严：如 7/11 不触发；
        #   现改为只要 API 返回行数 < grid 行数就补齐尾部空行）
        row_mismatch = actual_rows < n_rows
        col_mismatch = any(c < n_cols for c in row_col_counts)

        # 【开关】行列不足补全：默认常开（补行/补列后处理）。如需对比实验临时关闭，改此处为 False。
        ENABLE_RC_PADDING = True
        if not ENABLE_RC_PADDING:
            row_mismatch = False
            col_mismatch = False

        # 检测：列数多了且首列全空 → API 把前置白边误识为空列，需要裁掉
        # 判断标准：多数行（含 thead）都是 n_cols+1 列，且这些行首单元格全空
        excess_leading = False
        # 收集所有 tr（含 thead）
        all_tr_matches = list(re.finditer(r'<tr[^>]*>(.*?)</tr>', cleaned,
                                            re.DOTALL | re.IGNORECASE))
        if all_tr_matches:
            all_row_counts = [len(re.findall(r'<t[dh][^>]*>', m.group(1), re.IGNORECASE))
                              for m in all_tr_matches]
            n_plus1 = sum(1 for c in all_row_counts if c == n_cols + 1)
            # 多数行（至少 60%）是 n_cols+1 列
            if n_plus1 >= max(1, len(all_row_counts) * 0.6):
                # 检查这些行首单元格是否全空
                all_first_empty = True
                for m in all_tr_matches:
                    if len(re.findall(r'<t[dh][^>]*>', m.group(1), re.IGNORECASE)) == n_cols + 1:
                        first_cell = re.search(r'<t[dh][^>]*>(.*?)</t[dh]>', m.group(1),
                                                re.DOTALL | re.IGNORECASE)
                        if first_cell and first_cell.group(1).strip():
                            all_first_empty = False
                            break
                if all_first_empty:
                    excess_leading = True

        if excess_leading:
            logger.info(f"    [裁掉多余首列] {chunk_tag} API 返回多数行为 {n_cols+1} 列（预期 {n_cols}）且首列全空 → 裁掉")
            # 删除每行的首个 <td></td> 或 <th></th>（只对 n_cols+1 列的行）
            def _drop_first_cell(match):
                content = match.group(1)
                n_cells = len(re.findall(r'<t[dh][^>]*>', content, re.IGNORECASE))
                if n_cells == n_cols + 1:
                    new_content = re.sub(r'^\s*<t[dh][^>]*>\s*</t[dh]>', '', content,
                                          count=1, flags=re.IGNORECASE)
                    return f'<tr>{new_content}</tr>'
                return match.group(0)
            cleaned = re.sub(r'<tr[^>]*>(.*?)</tr>', _drop_first_cell, cleaned,
                              flags=re.DOTALL | re.IGNORECASE)
            # 重新提取 body_part 和 tr_matches
            thead_match = re.search(r'<thead>.*?</thead>', cleaned, re.DOTALL | re.IGNORECASE)
            body_part = cleaned[thead_match.end():] if thead_match else cleaned
            tr_matches = list(re.finditer(r'<tr[^>]*>(.*?)</tr>', body_part,
                                           re.DOTALL | re.IGNORECASE))
            actual_rows = len(tr_matches)
            row_col_counts = [len(re.findall(r'<t[dh][^>]*>', m.group(1), re.IGNORECASE))
                              for m in tr_matches]
            row_mismatch = actual_rows < n_rows
            col_mismatch = any(c < n_cols for c in row_col_counts)
            # 【开关】同上：禁用行列不足补全
            if not ENABLE_RC_PADDING:
                row_mismatch = False
                col_mismatch = False

        if row_mismatch or col_mismatch:
            actual_cols_summary = f"{min(row_col_counts)}~{max(row_col_counts)}" if row_col_counts and min(row_col_counts) != max(row_col_counts) else (str(row_col_counts[0]) if row_col_counts else '0')
            logger.info(f"    [行列不足补全] {chunk_tag} 返回 {actual_rows}×{actual_cols_summary}，预期 {n_rows}×{n_cols}")

        # 列数不足时：每行补空单元格
        if col_mismatch:
            def _pad_tr(match):
                tr_content = match.group(1)
                n_td = len(re.findall(r'<t[dh][^>]*>', tr_content, re.IGNORECASE))
                if n_td < n_cols:
                    pad = '<td></td>' * (n_cols - n_td)
                    return f'<tr>{tr_content}{pad}</tr>'
                return match.group(0)
            # 在 body_part 中替换每个 tr
            new_body = re.sub(r'<tr[^>]*>(.*?)</tr>', _pad_tr, body_part,
                              flags=re.DOTALL | re.IGNORECASE)
            # 重新拼接
            if thead_match:
                cleaned = cleaned[:thead_match.end()] + new_body
            else:
                # cleaned 不含 thead，直接代替 body 部分
                # body_part == cleaned 开头至末，直接用 new_body 重建
                cleaned = cleaned.replace(body_part, new_body)

        # 行数不足时补空行
        if row_mismatch:
            empty_row = '<tr>' + '<td></td>' * n_cols + '</tr>'
            pad_rows = empty_row * (n_rows - actual_rows)
            if '</table>' in cleaned:
                cleaned = cleaned.replace('</table>', pad_rows + '</table>')
            else:
                cleaned = cleaned + pad_rows + '</table>'

        # 行数过多时裁掉尾部空行（API 有时幻觉多输出空行，导致合并时错位）
        if actual_rows > n_rows:
            # 重新提取所有 tr（含 thead）
            all_tr = list(re.finditer(r'<tr[^>]*>(.*?)</tr>', cleaned,
                                       re.DOTALL | re.IGNORECASE))
            if len(all_tr) > n_rows:
                # 从尾部往前找连续的"全空行"
                trailing_empty = 0
                for m in reversed(all_tr):
                    cells = re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', m.group(1),
                                        re.DOTALL | re.IGNORECASE)
                    if cells and all(not c.strip() for c in cells):
                        trailing_empty += 1
                    else:
                        break
                # 需要裁掉的行数 = min(超出的行数, 尾部空行数)
                to_trim = min(len(all_tr) - n_rows, trailing_empty)
                if to_trim > 0:
                    logger.info(f"    [裁尾部空行] {chunk_tag} API 返回 {len(all_tr)} 行，预期 {n_rows}，裁掉尾部 {to_trim} 空行")
                    # 找出要保留的最后一个 tr 的结束位置
                    last_keep = all_tr[-(to_trim + 1)]
                    keep_end = last_keep.end()
                    # 拼接：保留位置 + 剩余（应该是 </tbody></table> 之类的闭合标签）
                    remainder = cleaned[all_tr[-1].end():]
                    cleaned = cleaned[:keep_end] + remainder

        # 【问题6】把捕获到的表格上方文字并入首行首列
        if above_text:
            cleaned = self._merge_above_text_into_first_cell(cleaned, above_text)

        return cleaned

    @staticmethod
    def _rebuild_table_from_text(text: str, n_rows: int, n_cols: int) -> Optional[str]:
        """【问题2/5】API 返回非 HTML（纯文本/markdown）时，按已知行列数重建 <table>。

        规则：每个非空行 → 一行；行内按分隔符（markdown `|` 或空白）切成单元格，
        按阅读顺序左→右填入，单元格不足补空、超出截断；行数不足补空行、超出截断。
        例 chunk_012（7列）：每行7个数字 → 11行×7列；
        例 chunk_065（无边框末块）："16128.75 0.00" / "0.00" → 行0前两格、行1第一格。
        """
        if not text or not text.strip():
            return None
        raw_lines = [ln for ln in text.split('\n') if ln.strip()]
        rows_cells = []
        for ln in raw_lines:
            ln = ln.strip()
            # 跳过 markdown 表格分隔行（|---|---|）
            if '-' in ln and re.fullmatch(r'[\|\s:\-]+', ln):
                continue
            if '|' in ln:
                cells = [c.strip() for c in ln.strip('|').split('|')]
            else:
                cells = ln.split()
            if cells:
                rows_cells.append(cells)
        if not rows_cells:
            return None
        trs = []
        for i in range(n_rows):
            cells = rows_cells[i] if i < len(rows_cells) else []
            cells = cells[:n_cols] + [''] * max(0, n_cols - len(cells))
            tds = ''.join(f'<td>{c}</td>' for c in cells)
            trs.append(f'<tr>{tds}</tr>')
        return '<table>' + ''.join(trs) + '</table>'

    @staticmethod
    def _pull_above_text_as_row(html: str) -> str:
        """【问题6-新增】若 <table> 上方文字按空格分割的词数 == 表格列数（≥2 词），
        则把该文字作为表格第一行插入（表头行被 VLM 甩到表上方的情形，如
        "第37保单年度 … 第44保单年度" 恰好 8 词 == 8 列）。否则原样返回。
        幂等：并入后表上方已无文字，再次调用为 no-op。"""
        low = html.lower()
        if '<table' not in low:
            return html
        tbl_pos = low.find('<table')
        pre = re.sub(r'[#*`>|]+', ' ', re.sub(r'</?[a-zA-Z][^>]*>', '', html[:tbl_pos]))
        pre_lines = [ln.strip() for ln in pre.split('\n') if ln.strip()]
        if not pre_lines:
            return html
        cand = pre_lines[-1]
        toks = cand.split()
        _fence_words = {'markdown', 'html', 'md', 'text', 'json', 'xml', 'csv'}
        if len(toks) < 2 or cand.lower() in _fence_words:
            return html
        tbl_body = html[tbl_pos:]
        # 表格实际列数：取各行 td/th 数的众数（稳健，避免个别畸形行干扰）
        _rc = [len(re.findall(r'<t[dh][^>]*>', _tr))
               for _tr in re.findall(r'<tr[^>]*>(.*?)</tr>', tbl_body, re.DOTALL | re.IGNORECASE)]
        _rc = [c for c in _rc if c > 0]
        tbl_cols = max(set(_rc), key=_rc.count) if _rc else 0
        if not tbl_cols or len(toks) != tbl_cols:
            return html
        new_tr = '<tr>' + ''.join(f'<td>{t}</td>' for t in toks) + '</tr>'
        m = re.search(r'<tr[^>]*>', tbl_body, re.IGNORECASE)
        if m:
            return tbl_body[:m.start()] + new_tr + tbl_body[m.start():]
        m2 = re.search(r'<table[^>]*>', tbl_body, re.IGNORECASE)
        return tbl_body[:m2.end()] + new_tr + tbl_body[m2.end():] if m2 else html

    @staticmethod
    def _merge_above_text_into_first_cell(html: str, above_text: str) -> str:
        """【问题6】把表格上方文字并入首行首列：首列为空则直接填入，非空则用 \\ 拼接
        （结果为 首列内容\\上方文字，如 "保单年度末\\投保年龄"）。"""
        m = re.search(r'(<tr[^>]*>\s*<t[dh][^>]*>)(.*?)(</t[dh]>)',
                      html, re.DOTALL | re.IGNORECASE)
        if not m:
            return html
        first = re.sub(r'<[^>]+>', '', m.group(2)).strip()
        merged = f"{first}\\{above_text}" if first else above_text
        return html[:m.start()] + m.group(1) + merged + m.group(3) + html[m.end():]

    @staticmethod
    def _chunk_basename(chunk: Dict) -> str:
        """chunk 文件名: chunk_XXX_rN-M_cN-M"""
        r0, r1 = chunk['row_range']
        c0, c1 = chunk['col_range']
        return f"chunk_{chunk['order']:03d}_r{r0}-{r1}_c{c0}-{c1}"

    # ==================== 表格后处理 ====================

    @staticmethod
    def _postprocess_table_markdown(markdown: str) -> str:
        """
        表格文档后处理：
        1. 如果表格上方最近的文字是"投保年龄"，合并到表头首单元格
           例: "投保年龄\n\n<table>...<td>保单年度末</td>..."
           → "<table>...<td>保单年度末/投保年龄</td>..."
        2. 把所有 <th> 替换为 <td>，</th> 替换为 </td>
        """
        if not markdown:
            return markdown

        # 1. 投保年龄合并到表头
        # 匹配模式: "投保年龄" 后跟空行，再跟 <table>
        # 把 "投保年龄" 合并到表格第一行第一个单元格
        pattern = r'投保年龄\s*\n\s*(<table[^>]*>\s*<tr[^>]*>\s*<t[dh][^>]*>)([^<]*)(</t[dh]>)'
        def merge_header(m):
            prefix = m.group(1)  # <table...><tr...><td>
            cell_content = m.group(2)  # 保单年度末
            close_tag = m.group(3)  # </td>
            # 合并: 保单年度末/投保年龄
            merged = f"{cell_content}/投保年龄" if cell_content else "投保年龄"
            return f"{prefix}{merged}{close_tag}"
        markdown = re.sub(pattern, merge_header, markdown, flags=re.IGNORECASE)

        # 2. <th> → <td>，</th> → </td>
        markdown = re.sub(r'<th\b', '<td', markdown, flags=re.IGNORECASE)
        markdown = re.sub(r'</th>', '</td>', markdown, flags=re.IGNORECASE)

        # 2.5【问题3】剔除"被内容夹在中间的连续全空列/行"（grid 过检产生的幻影列、空行）
        markdown = Pipeline._remove_empty_interior_rows_cols(markdown)

        # 2.6【每列去重"-"】每列最多保留一个 <td>-</td>（保留最上面的第一个），其余同列的"-"置空
        markdown = Pipeline._dedup_dash_columns(markdown)

        # 3. 表格内部"投保年龄"行合并：
        # 如果某行只有第一个单元格有值且为"投保年龄"（其余全空），
        # 则将其与上一行同位置合并为"xxx\投保年龄"，并删除该行。
        # 例: <tr><td>保单年度末</td><td>21</td>...</tr>
        #     <tr><td>投保年龄</td><td></td>...</tr>
        # →  <tr><td>保单年度末\投保年龄</td><td>21</td>...</tr>
        def _merge_tbbnl_row(html):
            # 匹配: 前一行 + 当前"投保年龄"行
            pattern = re.compile(
                r'(<tr[^>]*>)\s*(<td[^>]*>)([^<]*?)(</td>)'  # 前一行第一个td
                r'((?:\s*<td[^>]*>[^<]*</td>)*\s*</tr>)'  # 前一行剩余部分
                r'\s*'
                r'<tr[^>]*>\s*<td[^>]*>\s*投保年龄\s*</td>'  # 当前行第一个td="投保年龄"
                r'(?:\s*<td[^>]*>\s*</td>)*\s*</tr>',  # 当前行其余td全空
                re.IGNORECASE
            )
            def repl(m):
                open_tr = m.group(1)
                open_td = m.group(2)
                prev_content = m.group(3)
                close_td = m.group(4)
                rest = m.group(5)
                merged = f"{prev_content}\\投保年龄" if prev_content else "投保年龄"
                return f"{open_tr}{open_td}{merged}{close_td}{rest}"
            return pattern.sub(repl, html)
        markdown = _merge_tbbnl_row(markdown)

        # 4. 表头行 colspan 合并（仅对每个表格的第一行，且严格限制模式）：
        # 仅当首行符合 "N×空 + 内容 + M×空" 模式时才合并（内容格前后都有空格）。
        # 这排除了密集表头行（如 保单年度|21|22|23... 不应合并）。
        # 例: <th></th><th></th><th>保单年度</th><th></th><th></th>... → 合并
        #     <th>保单年度</th><th>21</th><th>22</th>... → 不合并（无前置空格）
        def _merge_colspan_first_row(html):
            def process_first_row(table_m):
                table_html = table_m.group(0)
                first_tr_m = re.search(r'<tr[^>]*>.*?</tr>', table_html, re.DOTALL)
                if not first_tr_m:
                    return table_html
                row_html = first_tr_m.group(0)
                cell_pattern = re.compile(r'<(td|th)([^>]*)>(.*?)</\1>', re.DOTALL)
                cells = list(cell_pattern.finditer(row_html))
                if not cells or len(cells) < 3:
                    return table_html

                # 判断每格是否为空
                def is_empty_cell(cell_m):
                    c = re.sub(r'<br\s*/?>', '', cell_m.group(3)).strip()
                    return c == ''

                empties = [is_empty_cell(c) for c in cells]
                n_total = len(cells)
                n_empty = sum(empties)

                # 严格条件：首行大部分是空格（>=50%），且至少有一个非空格被空格包围
                if n_empty < n_total * 0.5:
                    return table_html

                # 找"内容格后面紧跟空格"的模式进行合并
                new_cells = []
                i = 0
                while i < n_total:
                    if not empties[i]:
                        # 内容格：看后面有多少连续空格
                        tag = cells[i].group(1)
                        attrs = cells[i].group(2)
                        content = cells[i].group(3).strip()
                        span = 1
                        j = i + 1
                        while j < n_total and empties[j]:
                            span += 1
                            j += 1
                        # 只有后面确实跟了空格才合并
                        if span > 1:
                            new_cells.append(f'<{tag}{attrs} colspan="{span}">{content}</{tag}>')
                        else:
                            new_cells.append(f'<{tag}{attrs}>{content}</{tag}>')
                        i = j
                    else:
                        tag = cells[i].group(1)
                        attrs = cells[i].group(2)
                        new_cells.append(f'<{tag}{attrs}></{tag}>')
                        i += 1

                tr_open = re.match(r'<tr[^>]*>', row_html).group(0)
                new_row = tr_open + ''.join(new_cells) + '</tr>'
                return table_html[:first_tr_m.start()] + new_row + table_html[first_tr_m.end():]
            
            return re.sub(r'<table[^>]*>.*?</table>', process_first_row, html, flags=re.DOTALL)
        markdown = _merge_colspan_first_row(markdown)

        # 5. 删除每个表格末尾的纯空行（补行/合并残留的尾部空 <tr>，影响 TEDS）
        markdown = Pipeline._remove_trailing_empty_row(markdown)

        # 6.【rule4】纯空表格（所有单元格都空）整体删除
        markdown = Pipeline._remove_empty_tables(markdown)

        # 7.【rule3】小表格（行数×列数 < 50）去掉 HTML 格式，转为纯文本
        markdown = Pipeline._small_table_to_text(markdown)

        # 8.【rule2】限制非表格文本段长度 ≤200 字（放最后：覆盖 rule3 转出的文本、rule5 表外文本）
        markdown = Pipeline._cap_text_segments(markdown, limit=200)

        return markdown

    @staticmethod
    def _cap_text_segments(markdown: str, limit: int = 200) -> str:
        """【rule2】按 <table>...</table> 切分，对每个非表格文本段（标题/说明/小表转文本等）
        若 >limit 字则截断（防止小图幻觉产生的超长重复/泄漏文本，如 20472 字“上海分公司…”重复）。"""
        if not markdown:
            return markdown
        parts = re.split(r'(<table[^>]*>.*?</table>)', markdown, flags=re.S | re.I)
        out = []
        for part in parts:
            if re.match(r'\s*<table', part, re.I):
                out.append(part)                       # 表格原样保留
            else:
                stripped = part.strip()
                if len(stripped) > limit:
                    out.append('\n\n' + stripped[:limit].rstrip() + '\n\n')
                else:
                    out.append(part)
        return re.sub(r'\n{3,}', '\n\n', ''.join(out)).strip()

    @staticmethod
    def _remove_empty_tables(markdown: str) -> str:
        """【rule4】删除纯空 <table>（所有单元格去标签后均为空）。"""
        if not markdown or '<table' not in markdown.lower():
            return markdown

        def _repl(m):
            tb = m.group(0)
            for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tb, re.S | re.I):
                if re.sub(r'<[^>]+>', '', c).replace('&nbsp;', '').replace('\u3000', '').strip():
                    return tb            # 有非空单元格 → 保留
            return ''                    # 全空 → 删除
        md = re.sub(r'<table[^>]*>.*?</table>', _repl, markdown, flags=re.S | re.I)
        return re.sub(r'\n{3,}', '\n\n', md).strip()

    @staticmethod
    def _small_table_to_text(markdown: str) -> str:
        """【rule3】行数×列数 < 50 的小表格去掉 HTML，转纯文本
        （每行单元格空格连接、行间换行）。"""
        if not markdown or '<table' not in markdown.lower():
            return markdown

        def _repl(m):
            tb = m.group(0)
            trs = re.findall(r'<tr[^>]*>(.*?)</tr>', tb, re.S | re.I)
            if not trs:
                return tb
            n_cols = max((len(re.findall(r'<t[dh]\b', t)) for t in trs), default=0)
            if len(trs) * n_cols >= 50:
                return tb
            lines = []
            for tr in trs:
                cells = [re.sub(r'<[^>]+>', '', c).replace('&nbsp;', '').strip()
                         for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', tr, re.S | re.I)]
                cells = [c for c in cells if c]
                if cells:
                    lines.append(' '.join(cells))
            return '\n'.join(lines)
        return re.sub(r'<table[^>]*>.*?</table>', _repl, markdown, flags=re.S | re.I)

    @staticmethod
    def _remove_trailing_empty_row(markdown: str) -> str:
        """删除每个 <table> 末尾连续的"纯空 <tr>"（所有单元格为空），至少保留 1 行。
        补行/合并常在表尾留下空 <tr>，与 GT 不符会拉低 table_TEDS。"""
        if not markdown or '<table' not in markdown.lower():
            return markdown

        def _cell_empty(cell_html):
            inner = re.sub(r'<[^>]+>', '', cell_html)
            inner = inner.replace('&nbsp;', '').replace('\u3000', '').strip()
            return inner == ''

        def process_table(tm):
            table_html = tm.group(0)
            trs = list(re.finditer(r'<tr[^>]*>.*?</tr>', table_html, re.DOTALL | re.IGNORECASE))
            if len(trs) <= 1:
                return table_html
            last_keep = len(trs)
            for m in reversed(trs):
                cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', m.group(0), re.DOTALL | re.IGNORECASE)
                if cells and all(_cell_empty(c) for c in cells):
                    last_keep -= 1
                else:
                    break
            if last_keep >= len(trs) or last_keep < 1:
                return table_html
            return table_html[:trs[last_keep].start()] + table_html[trs[-1].end():]
        return re.sub(r'<table[^>]*>.*?</table>', process_table, markdown,
                      flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _remove_empty_interior_rows_cols(markdown: str) -> str:
        """【问题3】剔除表格中"被内容夹在中间的连续全空列/行"。

        - 全空列区域：某些列在所有行都为空，且其左、右都还存在非空列 → 删除（grid 过检的幻影列）。
        - 全空行区域：某行所有单元格都为空，且其上、下都还存在非空行 → 删除。
        仅处理无 colspan/rowspan 的规整表格，避免破坏合并单元格结构；
        阶梯表右下角的"尾部空格"因其所在列在上方行有内容，不会被误删。
        """
        if not markdown or '<table' not in markdown.lower():
            return markdown

        def _cell_empty(cell_html):
            inner = re.sub(r'<[^>]+>', '', cell_html)
            inner = inner.replace('&nbsp;', '').replace('\u3000', '').strip()
            return inner == ''

        def process_table(tm):
            table_html = tm.group(0)
            # 含 colspan/rowspan 的表跳过（结构复杂，避免误删）
            if re.search(r'\b(colspan|rowspan)\b', table_html, re.IGNORECASE):
                return table_html
            tr_list = re.findall(r'<tr[^>]*>.*?</tr>', table_html, re.DOTALL | re.IGNORECASE)
            if len(tr_list) < 3:
                return table_html
            rows = []
            for tr in tr_list:
                cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', tr, re.DOTALL | re.IGNORECASE)
                rows.append(cells)
            max_c = max(len(r) for r in rows)
            if max_c < 3:
                return table_html
            for r in rows:
                while len(r) < max_c:
                    r.append('<td></td>')
            n_r = len(rows)
            col_empty = [all(_cell_empty(rows[i][j]) for i in range(n_r)) for j in range(max_c)]
            row_empty = [all(_cell_empty(c) for c in rows[i]) for i in range(n_r)]

            def before(arr, idx):
                return any(not arr[k] for k in range(idx))

            def after(arr, idx):
                return any(not arr[k] for k in range(idx + 1, len(arr)))

            del_cols = {j for j in range(max_c)
                        if col_empty[j] and before(col_empty, j) and after(col_empty, j)}
            del_rows = {i for i in range(n_r)
                        if row_empty[i] and before(row_empty, i) and after(row_empty, i)}
            if not del_cols and not del_rows:
                return table_html
            new_trs = []
            for i in range(n_r):
                if i in del_rows:
                    continue
                kept = [rows[i][j] for j in range(max_c) if j not in del_cols]
                new_trs.append('<tr>' + ''.join(kept) + '</tr>')
            tbl_open_m = re.match(r'<table[^>]*>', table_html, re.IGNORECASE)
            tbl_open = tbl_open_m.group(0) if tbl_open_m else '<table>'
            return tbl_open + ''.join(new_trs) + '</table>'

        return re.sub(r'<table[^>]*>.*?</table>', process_table, markdown,
                      flags=re.DOTALL | re.IGNORECASE)

    @staticmethod
    def _dedup_dash_columns(markdown: str) -> str:
        """每列最多保留一个 `<td>-</td>`：按行从上到下扫描，某列第一次出现"-"予以保留，
        该列后续行再出现"-"则置为空 `<td></td>`。用于阶梯表中 API 沿列重复吐"-"（边界标记
        被向下传播、或整列杂"-"）的清理。仅处理无 colspan/rowspan 的规整表（保列索引可靠）。"""
        if not markdown or '<table' not in markdown.lower():
            return markdown

        def _is_dash(cell_html):
            return re.sub(r'<[^>]+>', '', cell_html).strip() == '-'

        def process_table(tm):
            table_html = tm.group(0)
            if re.search(r'\b(colspan|rowspan)\b', table_html, re.IGNORECASE):
                return table_html
            seen = set()  # 已保留过"-"的列索引

            def process_tr(trm):
                tr = trm.group(0)
                tr_open = re.match(r'<tr[^>]*>', tr, re.IGNORECASE).group(0)
                cells = re.findall(r'<t[dh][^>]*>.*?</t[dh]>', tr, re.DOTALL | re.IGNORECASE)
                out = []
                for j, c in enumerate(cells):
                    if _is_dash(c):
                        if j in seen:
                            out.append('<td></td>')
                        else:
                            seen.add(j)
                            out.append(c)
                    else:
                        out.append(c)
                return tr_open + ''.join(out) + '</tr>'

            return re.sub(r'<tr[^>]*>.*?</tr>', process_tr, table_html,
                          flags=re.DOTALL | re.IGNORECASE)

        return re.sub(r'<table[^>]*>.*?</table>', process_table, markdown,
                      flags=re.DOTALL | re.IGNORECASE)

    # ==================== 工具 ====================

    @staticmethod
    def _clean_response(resp: str) -> str:
        """去掉 markdown 代码块围栏（```/~~~ + 可选语言标识，如 ```markdown）。
        不强制要求围栏后有换行——避免 "```markdown<table>" 这种粘连时语言词泄漏进内容。"""
        if not resp:
            return ''
        s = resp.strip()
        # 开头围栏：``` 或 ~~~ + 可选空格 + 可选语言标识(markdown/html/md/...) + 可选空格/换行
        s = re.sub(r'^\s*(?:```|~~~)[ \t]*[a-zA-Z0-9]*[ \t]*\n?', '', s)
        # 结尾围栏
        s = re.sub(r'\n?\s*(?:```|~~~)\s*$', '', s.strip())
        # 【rule1】剔除 VLM 提示词泄漏（偶发把提示词当内容输出）
        s = re.sub(r'#*\s*Markdown Parsing Task\s*', '', s, flags=re.I)
        s = re.sub(r'Extract the text from the image\.?\s*', '', s, flags=re.I)
        return s.strip()

    @staticmethod
    def _strip_html_tags(text: str) -> str:
        """把 text region 识别结果中的 HTML 格式剔除，转为纯文本/markdown。

        text region 应是纯文本（标题/说明/注释），不应含 HTML。若 API 返回了
        <p>/<br>/<table>/<b> 等标签，去掉标签保留内容：块级标签→换行、单元格→空格、
        其余标签删除，再反转义 HTML 实体、规整空白。
        """
        if not text:
            return text
        # 仅当存在疑似 HTML 标签（以字母或 / 开头）时才处理，
        # 避免误伤文本中的数学符号（如 "年龄 < 18 且 x > 0"）
        if not re.search(r'</?[a-zA-Z][^>]*>', text):
            return text
        import html as _html2
        s = text
        # <br> → 换行
        s = re.sub(r'<br\s*/?>', '\n', s, flags=re.IGNORECASE)
        # 块级结束标签 → 换行
        s = re.sub(r'</(p|div|tr|h[1-6]|li|ul|ol|table|thead|tbody|caption)>',
                   '\n', s, flags=re.IGNORECASE)
        # 单元格结束 → 空格（避免数字/文字粘连）
        s = re.sub(r'</(td|th)>', ' ', s, flags=re.IGNORECASE)
        # 删除所有剩余标签（以字母或 / 开头，不误伤数学符号）
        s = re.sub(r'</?[a-zA-Z][^>]*>', '', s)
        # 反转义 HTML 实体（&nbsp; &amp; 等）
        s = _html2.unescape(s)
        # 规整空白
        s = re.sub(r'[ \t]+', ' ', s)
        s = re.sub(r'\n[ \t]+', '\n', s)
        s = re.sub(r'\n{3,}', '\n\n', s)
        return s.strip()

    @staticmethod
    def _save_text(path: str, content: str):
        """安全写入文本文件"""
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content or '')
        except Exception as e:
            logger.warning(f"保存文件失败 {path}: {e}")

    # ==================== HTML 可视化报告 ====================

    def _save_html_report(self, save_dir: str, img_path: str,
                           region_details: List[Dict], final_md: str):
        """
        生成 report.html：Level 1 分割 + 每个 region 详情（含 chunks + API 原始返回）+ 最终 md
        """
        img_name = os.path.basename(img_path)

        # ---- 头部 + 样式 ----
        css = """body{font-family:Arial,sans-serif;max-width:1600px;margin:20px auto;padding:0 20px;color:#333}
h2{border-bottom:2px solid #333;padding-bottom:8px}
h3{margin-top:32px;border-left:4px solid #3498db;padding-left:10px}
h4{margin-top:24px;color:#e67e22}
h5{margin-top:20px;color:#555}
table{border-collapse:collapse;margin:12px 0}
th{background:#f5f5f5;padding:6px;border:1px solid #ccc;text-align:left}
td{border:1px solid #ccc;vertical-align:top;padding:6px}
img{max-width:100%;border:1px solid #ddd}
pre{white-space:pre-wrap;max-height:400px;overflow:auto;background:#f8f8f8;padding:10px;border:1px solid #ddd;font-size:12px;line-height:1.4}
.meta{color:#666;font-size:13px;margin:4px 0}
.text-region{background:#eef}
.table-region{background:#fee}
.stat{color:#3498db;font-weight:bold}
.raw{background:#fffbe6;border-color:#ffe58f}
.processed{background:#f6ffed;border-color:#b7eb8f}
.final{background:#e6f7ff;border:2px solid #91d5ff;padding:15px;border-radius:4px}
.chunk-card{border:1px solid #e0e0e0;border-radius:8px;margin:12px 0;overflow:hidden}
.chunk-header{background:#f7f7f7;padding:8px 12px;display:flex;gap:12px;align-items:center;font-size:13px;border-bottom:1px solid #e0e0e0}
.chunk-id{font-weight:bold;color:#e67e22;font-size:15px}
.chunk-stat{margin-left:auto;color:#888;font-size:12px}
.chunk-body{display:flex;gap:0}
.chunk-img-col{flex:0 0 auto;padding:10px;border-right:1px solid #eee;max-width:45%}
.chunk-img{max-width:100%;max-height:300px;object-fit:contain}
.chunk-result-col{flex:1;padding:10px;overflow:auto;max-height:400px}
.render-table{border-collapse:collapse;font-size:11px;width:auto}
.render-table td,.render-table th{border:1px solid #d1d5db;padding:2px 5px;text-align:center;white-space:nowrap;min-width:25px}
.render-table tr:first-child{background:#eff6ff;font-weight:500}
.render-table tr:nth-child(even){background:#f9fafb}
.merged-render{overflow:auto;max-height:600px;border:1px solid #e0e0e0;padding:10px;border-radius:4px}"""

        # ---- Level 1 概览表 ----
        overview_rows = []
        for d in region_details:
            row_cls = 'text-region' if d['type'] == 'text' else 'table-region'
            overview_rows.append(
                f"<tr class='{row_cls}'><td>{d['order']}</td>"
                f"<td>{d['type']}</td>"
                f"<td>{d['size'][0]}×{d['size'][1]}</td>"
                f"<td>bbox={d['bbox']}</td>"
                f"<td>{len(d['chunks'])} chunks</td>"
                f"<td>{len(d['processed'])} 字符</td>"
                f"<td><a href='#region_{d['order']:02d}'>详情</a></td></tr>"
            )
        overview_html = (
            "<h3>Level 1 区域分割概览</h3>"
            "<table><thead><tr><th>#</th><th>类型</th><th>尺寸</th>"
            "<th>bbox</th><th>chunks</th><th>md 长度</th><th></th></tr></thead>"
            f"<tbody>{''.join(overview_rows)}</tbody></table>"
        )

        # ---- 每个 region 的详细 ----
        region_html_parts = []
        for d in region_details:
            region_html_parts.append(self._render_region_html(d))

        # ---- 最终 markdown ----
        final_html = (
            "<h3 id='final'>最终完整 Markdown（合并 + 后处理）</h3>"
            f"<div class='meta'>长度: <span class='stat'>{len(final_md)} 字符</span></div>"
            f"<pre class='final'>{_html.escape(final_md[:200000])}"
            f"{'...(截断)' if len(final_md) > 200000 else ''}</pre>"
        )

        html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{_html.escape(img_name)}</title>
<style>{css}</style></head><body>
<h2>{_html.escape(img_name)}</h2>
<p class='meta'>原图: {_html.escape(img_path)}</p>
<p class='meta'>Regions: <span class='stat'>{len(region_details)}</span>
(text: {sum(1 for d in region_details if d['type']=='text')},
table: {sum(1 for d in region_details if d['type']=='table')})</p>
{overview_html}
{''.join(region_html_parts)}
{final_html}
</body></html>"""

        with open(os.path.join(save_dir, 'report.html'), 'w', encoding='utf-8') as f:
            f.write(html_doc)

    def _render_region_html(self, d: Dict) -> str:
        """渲染单个 region 的 HTML 片段"""
        order = d['order']
        rtype = d['type']
        image_file = d['image_file']

        # 头部 + 该 region 的原图
        img_tag = (f"<img src='{_html.escape(image_file)}' alt='region {order}'>"
                   if os.path.exists(image_file) or True else '(图片缺失)')

        if rtype == 'text':
            # text region：显示原图 + 原始 API 返回 + 处理后 md
            return (
                f"<h4 id='region_{order:02d}'>Region #{order} (text) "
                f"{d['size'][0]}×{d['size'][1]}</h4>"
                f"<div class='meta'>bbox={d['bbox']}</div>"
                f"<table><tr>"
                f"<td style='width:35%'>{img_tag}</td>"
                f"<td style='width:32%'>"
                f"<div class='meta'>API 原始返回 (len={len(d['raw'])}):</div>"
                f"<pre class='raw'>{_html.escape(d['raw'][:3000])}"
                f"{'...(截断)' if len(d['raw']) > 3000 else ''}</pre></td>"
                f"<td style='width:33%'>"
                f"<div class='meta'>后处理 (去 markdown 代码块) (len={len(d['processed'])}):</div>"
                f"<pre class='processed'>{_html.escape(d['processed'][:3000])}"
                f"{'...(截断)' if len(d['processed']) > 3000 else ''}</pre></td>"
                f"</tr></table>"
            )
        else:
            # table region：chunk 图片 vs 渲染结果 并排对比
            chunk_cards = []
            for c in d['chunks']:
                # 渲染 chunk 的 HTML 表格（不转义，直接渲染）
                resp_html = c['response']
                if '<table' not in resp_html.lower():
                    resp_html = f"<table border='1'>{resp_html}</table>"
                # 给渲染表格加样式
                resp_html = resp_html.replace('<table', "<table class='render-table'")

                filled_tag = ""
                # 检查是否有填充图（去掉尺寸后缀再加_filled）
                import re as _re
                base_no_size = _re.sub(r'_\d+x\d+\.jpg$', '', c['image_file'])
                filled_path = f"{base_no_size}_filled.jpg"
                filled_tag = f"<div class='meta'><a href='{_html.escape(filled_path)}'>查看填充图</a></div>"

                chunk_cards.append(f"""
                <div class="chunk-card">
                  <div class="chunk-header">
                    <span class="chunk-id">#{c['order']}</span>
                    <span>r{c['row_range'][0]}-{c['row_range'][1]} c{c['col_range'][0]}-{c['col_range'][1]}</span>
                    <span>{c['size'][0]}×{c['size'][1]}</span>
                    <span class="chunk-stat">tr={c['response'].lower().count('<tr')} td={c['response'].lower().count('<td')+c['response'].lower().count('<th')}</span>
                  </div>
                  <div class="chunk-body">
                    <div class="chunk-img-col">
                      <img src='{_html.escape(c['image_file'])}' class="chunk-img">
                      {filled_tag}
                    </div>
                    <div class="chunk-result-col">{resp_html}</div>
                  </div>
                </div>""")

            # 合并后的表格也渲染出来
            merged_html = d['processed']
            if '<table' not in merged_html.lower() and '<tr' in merged_html.lower():
                merged_html = f"<table border='1'>{merged_html}</table>"
            merged_html = merged_html.replace('<table', "<table class='render-table'")

            return f"""
            <h4 id='region_{order:02d}'>Region #{order} (table) {d['size'][0]}×{d['size'][1]} — {len(d['chunks'])} chunks</h4>
            <div class='meta'>bbox={d['bbox']}</div>
            <details><summary>原表格图（点击展开）</summary>{img_tag}</details>

            <h5>合并后表格（渲染）</h5>
            <div class="merged-render">{merged_html}</div>

            <h5>Chunk 逐个对比（左:图片 | 右:识别渲染）</h5>
            {''.join(chunk_cards)}
            """

    # ==================== 长文档 HTML 报告 ====================

    def _save_long_html_report(self, save_dir: str, img_path: str,
                                 chunk_details: List[Dict],
                                 merged_raw: str, merged_post: str,
                                 merged_final: str):
        """
        长文档 report.html：Chunks 列表 + API 原始返回 + 三层合并结果
        """
        img_name = os.path.basename(img_path)

        css = """body{font-family:Arial,sans-serif;max-width:1400px;margin:20px auto;padding:0 20px;color:#333}
h2{border-bottom:2px solid #333;padding-bottom:8px}
h3{margin-top:32px;border-left:4px solid #3498db;padding-left:10px}
table{border-collapse:collapse;width:100%;margin:12px 0}
th{background:#f5f5f5;padding:8px;border:1px solid #ccc;text-align:left}
td{border:1px solid #ccc;vertical-align:top;padding:8px}
img{max-width:400px;max-height:600px;border:1px solid #ddd}
pre{white-space:pre-wrap;max-height:400px;overflow:auto;background:#f8f8f8;padding:10px;border:1px solid #ddd;font-size:12px;line-height:1.4}
.meta{color:#666;font-size:13px;margin:4px 0}
.stat{color:#3498db;font-weight:bold}
.raw{background:#fffbe6;border-color:#ffe58f}
.processed{background:#f6ffed;border-color:#b7eb8f}
.final{background:#e6f7ff;border:2px solid #91d5ff;padding:15px;border-radius:4px}"""

        # Chunks 详情
        chunk_rows = []
        for c in chunk_details:
            chunk_rows.append(
                f"<tr><td>{c['order']}</td>"
                f"<td>{_html.escape(c['position'])}<br>{c['size'][0]}×{c['size'][1]}</td>"
                f"<td><a href='{_html.escape(c['image_file'])}'>"
                f"<img src='{_html.escape(c['image_file'])}'></a></td>"
                f"<td><div class='meta'>len={len(c['response'])}, "
                f"tr={c['response'].lower().count('<tr')}, "
                f"td/th={c['response'].lower().count('<td')+c['response'].lower().count('<th')}</div>"
                f"<pre class='raw'>{_html.escape(c['response'][:3000])}"
                f"{'...(截断)' if len(c['response']) > 3000 else ''}</pre></td></tr>"
            )
        chunks_table = (
            "<table><thead><tr><th>#</th><th>坐标/尺寸</th>"
            "<th>chunk 图</th><th>API 原始返回</th></tr></thead>"
            f"<tbody>{''.join(chunk_rows)}</tbody></table>"
        )

        # 三层 md
        def render_md_block(title: str, content: str, cls: str, max_len: int = 20000):
            return (
                f"<h3>{title}</h3>"
                f"<div class='meta'>长度: <span class='stat'>{len(content)} 字符</span></div>"
                f"<pre class='{cls}'>{_html.escape(content[:max_len])}"
                f"{'...(截断)' if len(content) > max_len else ''}</pre>"
            )

        html_doc = f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{_html.escape(img_name)}</title>
<style>{css}</style></head><body>
<h2>{_html.escape(img_name)} (long)</h2>
<p class='meta'>原图: {_html.escape(img_path)}</p>
<p class='meta'>Chunks: <span class='stat'>{len(chunk_details)}</span></p>
<h3>Chunks 详情（每 chunk 图 + API 原始返回）</h3>
{chunks_table}
{render_md_block("① merge_vertical 后 (merged_raw)", merged_raw, "raw")}
{render_md_block("② post_process 后 (merged_post)", merged_post, "processed")}
{render_md_block("③ post_process_headings 后 (最终 final.md)", merged_final, "final")}
</body></html>"""

        self._save_text(os.path.join(save_dir, 'report.html'), html_doc)
