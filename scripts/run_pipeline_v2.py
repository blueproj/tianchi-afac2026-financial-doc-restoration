"""
【复现入口】B 榜全量跑批脚本：遍历测试集图片 → 逐张调用 Pipeline 解析 → 写提交 CSV。

推荐直接使用包根目录的一键脚本（内部即调用本脚本）：
    bash run.sh

也可手动指定参数运行（与 B 榜提交时完全一致的命令）：
    python scripts/run_pipeline_v2.py \\
        --table-dir "data/finix_huge_table_rest_B/images" \\
        --long-dir  "data/finix_huge_long_rest_B/images" \\
        --no-em-fill --no-resume \\
        --save-dir output/pipeline_B \\
        --output submission_B.csv

参数说明：
    --table-dir / --long-dir  表格类 / 长文档类图片目录（各 50 张，按目录决定处理链路）
    --no-em-fill              禁用空单元格占位填充（最终提交配置；实测占位符弊大于利）
    --no-resume               从头处理全部图片（不读取已有 CSV 续跑）
    --save-dir                中间产物目录（每样本一个子目录，含 final.md / report.html）
    --output                  提交 CSV 路径（两列：file_name, ground_truth）

运行行为：
- 每完成一张图立即增量写盘 CSV（中断可恢复，去掉 --no-resume 即续跑）；
- 单样本异常不影响全局（兜底空结果 + 日志记录）；
- 所有中间产物实时保存，便于审核时逐样本逐 chunk 核对。
"""
import os
import sys
import csv
import time
import argparse
import logging
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, CODE_DIR)

# 提高 csv 字段大小限制（long 文档 markdown 可能超过默认 131072）
csv.field_size_limit(100 * 1024 * 1024)  # 100 MB

from src.pipeline_v2 import Pipeline
from src.api_client import FinixDocClient
from configs.config import OUTPUT_DIR, TEST_A_LONG_DIR, TEST_A_TABLE_DIR


# ==================== 日志 ====================
logger = logging.getLogger("run_pipeline")
logger.setLevel(logging.INFO)
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(h)


# ==================== 断点续传 ====================
_save_lock = threading.Lock()


def load_completed(output_csv):
    completed = {}
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed[row['file_name']] = row['ground_truth']
        logger.info(f"断点续传：已完成 {len(completed)} 张")
    return completed


def save_incremental(output_csv, completed, file_name, markdown):
    with _save_lock:
        completed[file_name] = markdown
        with open(output_csv, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['file_name', 'ground_truth'],
                                     quoting=csv.QUOTE_ALL)
            writer.writeheader()
            for fname in sorted(completed.keys()):
                writer.writerow({
                    'file_name': fname,
                    'ground_truth': completed[fname]
                })


# ==================== 数据源准备 ====================

def build_task_list(args):
    """
    构造待处理任务列表：[(img_path, doc_type_hint), ...]
    """
    tasks = []

    # 优先级 1: 显式指定的 --table-dir / --long-dir（交错混合）
    if args.table_dir or args.long_dir:
        table_tasks = _scan_dir(args.table_dir, 'table') if args.table_dir else []
        long_tasks = _scan_dir(args.long_dir, 'long') if args.long_dir else []
        interleaved = []
        for i in range(max(len(table_tasks), len(long_tasks))):
            if i < len(table_tasks):
                interleaved.append(table_tasks[i])
            if i < len(long_tasks):
                interleaved.append(long_tasks[i])
        return interleaved

    # 优先级 2: --mode test_a 用预设
    if args.mode == 'test_a':
        table_dir = os.path.join(TEST_A_TABLE_DIR, 'images')
        long_dir = os.path.join(TEST_A_LONG_DIR, 'images')
        table_tasks = _scan_dir(table_dir, 'table')
        long_tasks = _scan_dir(long_dir, 'long')
        # 交错混合：table long table long ... 便于 --limit 时两种都被采样到
        interleaved = []
        for i in range(max(len(table_tasks), len(long_tasks))):
            if i < len(table_tasks):
                interleaved.append(table_tasks[i])
            if i < len(long_tasks):
                interleaved.append(long_tasks[i])
        return interleaved

    # 优先级 3: --input-dir + --doc-type
    if args.input_dir:
        return _scan_dir(args.input_dir, args.doc_type)

    return []


def _scan_dir(directory, doc_type_hint):
    """扫描一个图片目录，返回 [(path, doc_type_hint), ...]"""
    if not os.path.isdir(directory):
        logger.warning(f"目录不存在: {directory}")
        return []
    files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    return [(os.path.join(directory, f), doc_type_hint) for f in files]


# ==================== 主流程 ====================

def make_sid(fname):
    """生成唯一短ID，避免同一保单多页（如 21473_..._page1/page6）因取前8字符
    (fname[:8]='21473_1.') 相同而导致中间结果目录互相覆盖。与 run_pre_api_B 一致：
    - UUID 名（含 '-'）：取第一段
    - 中文长名（含 _pageN）：数字前缀 + page 号，如 21473_p1
    - 其他：去扩展名后前 12 字符
    """
    import re
    if '-' in fname:
        return fname.split('-')[0]
    m = re.match(r'^(\d+)_.*_page(\d+)', fname)
    if m:
        return f"{m.group(1)}_p{m.group(2)}"
    return fname.rsplit('.', 1)[0][:12]


def process_one(pipeline: Pipeline, img_path: str, doc_type: str,
                save_root: str = None) -> str:
    fname = os.path.basename(img_path)
    save_dir = None
    if save_root:
        short_id = make_sid(fname)
        save_dir = os.path.join(save_root, short_id)
    try:
        return pipeline.process_image(img_path, save_dir=save_dir, doc_type=doc_type)
    except Exception as e:
        logger.error(f"处理失败: {fname}, 错误: {e}\n{traceback.format_exc()}")
        return ''


def main():
    parser = argparse.ArgumentParser(description="新版 pipeline 运行器（默认保存所有中间文件）")
    # 输入模式（三选一）
    parser.add_argument('--mode', choices=['test_a', 'custom'], default='custom',
                        help='test_a=用 config 预设 A 榜双目录；custom=用 --table-dir/--long-dir/--input-dir')
    parser.add_argument('--table-dir', default=None,
                        help='表格图片目录（doc_type=table）')
    parser.add_argument('--long-dir', default=None,
                        help='长文档图片目录（doc_type=long）')
    parser.add_argument('--input-dir', default=None,
                        help='通用目录，配合 --doc-type 使用')
    parser.add_argument('--doc-type', choices=['auto', 'table', 'long'], default='auto',
                        help='--input-dir 场景下的文档类型（默认 auto 用宽高比判断）')
    # 输出
    parser.add_argument('--output', default='submission.csv',
                        help='输出 CSV 路径')
    # 并发
    parser.add_argument('--workers', type=int, default=3,
                        help='图片级并发数')
    parser.add_argument('--chunk-workers', type=int, default=8,
                        help='chunk 级并发数')
    # 控制
    parser.add_argument('--limit', type=int, default=None,
                        help='只处理前 N 张（快速测试）')
    parser.add_argument('--no-resume', action='store_true',
                        help='不用断点续传（从头重跑）')
    parser.add_argument('--no-save', action='store_true',
                        help='关闭中间文件保存（仅保留 CSV 输出 + API 缓存）')
    parser.add_argument('--save-dir', default=None,
                        help='中间文件保存根目录（默认 output/pipeline_run/）')
    parser.add_argument('--no-em-fill', action='store_true',
                        help='禁用 EM 空单元格占位符填充（保留纯空跳过+密集放大）')
    args = parser.parse_args()

    # 构造任务列表
    tasks = build_task_list(args)
    if not tasks:
        logger.error("没有任务：请指定 --table-dir/--long-dir/--input-dir 或 --mode test_a")
        sys.exit(1)

    if args.limit:
        tasks = tasks[:args.limit]

    # 统计
    n_table = sum(1 for _, t in tasks if t == 'table')
    n_long = sum(1 for _, t in tasks if t == 'long')
    n_auto = sum(1 for _, t in tasks if t == 'auto')
    logger.info(f"待处理: 共 {len(tasks)} 张 "
                f"(table: {n_table}, long: {n_long}, auto: {n_auto})")

    # 断点续传
    output_csv = os.path.abspath(args.output)
    completed = {} if args.no_resume else load_completed(output_csv)

    # 过滤已完成
    todo = [(p, t) for p, t in tasks if os.path.basename(p) not in completed]
    if not todo:
        logger.info("全部已完成，无需处理")
        return
    logger.info(f"实际待处理: {len(todo)} 张（已完成 {len(completed)}）")

    # 中间保存目录
    save_root = None
    if not args.no_save:
        save_root = args.save_dir or os.path.join(OUTPUT_DIR, 'pipeline_run')
        os.makedirs(save_root, exist_ok=True)
        logger.info(f"中间文件保存目录: {save_root}")
    else:
        logger.info("中间文件保存: 已关闭（--no-save）")

    # 共享的 API client
    api_client = FinixDocClient()

    t_start = time.time()
    n_done = 0
    n_fail = 0

    def worker(task):
        img_path, doc_type = task
        pipeline = Pipeline(api_client=api_client, chunk_workers=args.chunk_workers,
                            enable_em_fill=not args.no_em_fill)
        md = process_one(pipeline, img_path, doc_type, save_root=save_root)
        return os.path.basename(img_path), md

    if args.workers <= 1:
        for i, task in enumerate(todo):
            fname_, md = worker(task)
            if md:
                save_incremental(output_csv, completed, fname_, md)
                n_done += 1
            else:
                n_fail += 1
            elapsed = time.time() - t_start
            logger.info(f"[{i+1}/{len(todo)}] {task[0]} [{task[1]}] "
                        f"→ {len(md)} 字符 (done={n_done}, fail={n_fail}, "
                        f"elapsed={elapsed:.0f}s)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(worker, task): task for task in todo}
            for future in as_completed(futures):
                task = futures[future]
                try:
                    fname_, md = future.result()
                    if md:
                        save_incremental(output_csv, completed, fname_, md)
                        n_done += 1
                    else:
                        n_fail += 1
                except Exception as e:
                    logger.error(f"任务异常 {task[0]}: {e}")
                    n_fail += 1
                elapsed = time.time() - t_start
                logger.info(f"[进度 {n_done+n_fail}/{len(todo)}] "
                            f"{os.path.basename(task[0])} [{task[1]}] "
                            f"→ done={n_done}, fail={n_fail}, elapsed={elapsed:.0f}s")

    # 汇总
    total_elapsed = time.time() - t_start
    logger.info(f"\n===== 完成 =====")
    logger.info(f"输入: {len(tasks)} 张 (table={n_table}, long={n_long}), "
                f"已完成: {len(completed)}, 本次成功: {n_done}, 失败: {n_fail}, "
                f"耗时: {total_elapsed:.0f}s")
    logger.info(f"输出 CSV: {output_csv}")
    if save_root:
        logger.info(f"中间文件: {save_root}")

    if hasattr(api_client, 'stats'):
        s = api_client.stats
        logger.info(f"API 统计: 总调用={s.get('total_calls', 0)}, "
                    f"成功={s.get('success_calls', 0)}, "
                    f"失败={s.get('fail_calls', 0)}, "
                    f"总耗时={s.get('total_time', 0):.0f}s")


if __name__ == '__main__':
    main()
