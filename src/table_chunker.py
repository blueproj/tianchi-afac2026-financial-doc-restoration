"""
Level 2 表格切块模块

对 Level 1 切出的单个 table 区域进行细粒度切分，让每个 chunk 能被 API 完整识别。

核心思路：
1. 结构分析：检测所有行/列边界
2. 找表格真正的第一行（黑色像素跨列数占比 >= 60% 的第一行）
3. 找表格真正的第一列（黑色像素跨行数占比 >= 60% 的第一列）
4. 拆出第一行上方 / 第一列左侧的额外内容作为独立 chunk
5. 上下切分（每 chunk 拼首行）或 2D 切分（列多时拼首行 + 首列）
"""
import os
import io
import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.subtable_splitter import SubTableSplitter

logger = logging.getLogger("table_chunker")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)

# ==================== 配置 ====================
WHITE_THRESHOLD = 252

# 表格结构判据
HEADER_ROW_COVERAGE = 0.6       # 表格真正第一行的列覆盖率阈值
FIRST_COL_COVERAGE = 0.6        # 表格真正第一列的行覆盖率阈值

# 行/列边界检测
MIN_ROW_GAP = 3                 # 行间空白最小高度（像素）
MIN_COL_GAP = 3                 # 列间空白最小宽度（像素）

# 合并间距太近的段：字符内的窄空隔（如"713.46"中"3"和"."之间）不应作为切分点
MERGE_COL_GAP = 20              # 列间空白 < 此值 → 合并（保护千分位逗号如"2,714.26"）
MERGE_ROW_GAP = 5               # 行间空白 < 此值 → 合并

# chunk 粒度（基于 API 实测反馈）
# 关键结论：API 输出上限约 11000 字符，粒度约束在 cells 数
# 保守设 400 cells，应对高密度数字表（每格约 25 字符 × 400 = 10000，留 1000 buffer）
CELLS_PER_CHUNK = 400
COLS_PER_SEGMENT = 20           # 左右切分时每段列数
MAX_COLS_FULL_WIDTH = 25        # 列数超过此值必须做左右切分
MAX_ROWS_PER_CHUNK = 30         # 单 chunk 行数硬上限

DEFAULT_ROWS_PER_CHUNK = 20     # 默认每 chunk 行数
DEFAULT_COLS_PER_CHUNK = COLS_PER_SEGMENT

# 小图片填补配置（API 对超小图识别质量崩溃）
MIN_CHUNK_WIDTH = 200           # 小于此宽度的图会 pad 白色
MIN_CHUNK_HEIGHT = 100          # 小于此高度的图会 pad 白色
PAD_LR = 100                    # 左右每边填补像素
PAD_TB = 50                     # 上下每边填补像素

# 拼接首行/首列时留的边距
GLUE_PADDING = 3


def uniform_split(total: int, target_size: int) -> List[int]:
    """
    把 total 均匀切成大小接近 target_size 的段，避免小尾巴。

    规则：
    - total <= target_size: 不切（一段）
    - total > target_size: 必须切（至少 2 段，段数按四舍五入）

    例：
      uniform_split(41, 20)  → [21, 20]        (2 段)
      uniform_split(100, 20) → [20, 20, 20, 20, 20]
      uniform_split(65, 20)  → [22, 22, 21]
      uniform_split(21, 15)  → [11, 10]        (round=1 但 total>target 强制切)
      uniform_split(19, 20)  → [19]            (不切)
      uniform_split(1, 20)   → [1]
    """
    if total <= 0:
        return []
    if total <= target_size:
        return [total]
    # total > target_size：至少切 2 段
    n = max(2, round(total / target_size))
    base = total // n
    extra = total % n
    # 多出来的 extra 段大 1，均摊到前面
    return [base + 1] * extra + [base] * (n - extra)


def compute_chunk_plan(n_rows: int, n_cols: int,
                       table_h: int = 0, table_w: int = 0,
                       prefix_h: int = 0, prefix_w: int = 0) -> Tuple[List[int], List[int]]:
    """
    根据表格行/列数 + 像素尺寸计算最优 chunk 切分计划。

    核心约束：
    - 每 chunk 目标 cells <= 400（API 输出上限 ~10000 字符 / 25字符/cell）
    - chunk 每边像素不超过 2500px（API 对超大图识别质量下降）
    - chunk 每边像素不低于 300px（太小导致 API 幻觉）
    - 切分后数据区域必须按行/列边界切

    动态计算（当提供 table_h, table_w 时）：
    - 用 avg_row_h 和 avg_col_w 反推 chunk 内可容纳的行/列数
    - 确保 chunk 像素尺寸在 [300, 2500] 之间
    - 若提供 prefix_h/prefix_w（表头/首列像素）: 从预算中扣除

    兼容旧逻辑（未提供 table_h, table_w 时）：
    - 回退到基于行列数的决策矩阵

    Args:
        n_rows: 数据行数（不含表头）
        n_cols: 数据列数（不含首列）
        table_h: 数据区域高度（像素），0 表示未提供
        table_w: 数据区域宽度（像素），0 表示未提供

    Returns:
        (row_sizes, col_sizes): 每段大小列表
    """
    if n_rows <= 0 or n_cols <= 0:
        return [max(1, n_rows)], [max(1, n_cols)]

    # 极小表：整体送
    if n_rows * n_cols < 100:
        return [n_rows], [n_cols]

    # 单向长表
    if n_rows == 1:
        return [1], uniform_split(n_cols, 20)
    if n_cols == 1:
        return uniform_split(n_rows, 20), [1]

    # ==================== 像素感知模式 ====================
    if table_h > 0 and table_w > 0:
        MAX_CELLS = 400
        MAX_CHUNK_PX = 2500
        MIN_CHUNK_PX = 300
        MAX_ROWS_PER_CHUNK = 12   # 硬约束：单 chunk 最多 12 行
        MAX_COLS_PER_CHUNK = 12   # 硬约束：单 chunk 最多 12 列

        avg_row_h = table_h / n_rows
        avg_col_w = table_w / n_cols

        # 像素约束下的最大行/列数
        # 扣除表头/首列的像素（这些会拼接到 body chunk 上）
        # 保留至少 300px 给 body（防止 prefix 过大导致 body 无法切分）
        avail_h = max(300, MAX_CHUNK_PX - prefix_h)
        avail_w = max(300, MAX_CHUNK_PX - prefix_w)
        max_rows_by_px = max(3, int(avail_h / max(1, avg_row_h)))
        max_cols_by_px = max(3, int(avail_w / max(1, avg_col_w)))

        # cells 约束下的目标行/列数
        # 优先让 chunk 接近正方形（视觉平衡 + API 更好识别）
        target_cols = min(n_cols, max_cols_by_px, MAX_COLS_PER_CHUNK,
                          max(5, int((MAX_CELLS * avg_col_w / max(1, avg_row_h)) ** 0.5)))
        target_rows = min(n_rows, max_rows_by_px, MAX_ROWS_PER_CHUNK,
                          max(5, MAX_CELLS // max(1, target_cols)))

        # 最小尺寸约束（chunk 不能太小）
        min_rows = max(3, int(MIN_CHUNK_PX / max(1, avg_row_h)))
        min_cols = max(3, int(MIN_CHUNK_PX / max(1, avg_col_w)))
        target_rows = max(target_rows, min_rows)
        target_cols = max(target_cols, min_cols)

        # 硬约束再次夹紧（防止 min 反弹超过 12）
        target_rows = min(target_rows, MAX_ROWS_PER_CHUNK, n_rows)
        target_cols = min(target_cols, MAX_COLS_PER_CHUNK, n_cols)

        # 如果整表就满足 cells 约束，不切
        if n_rows <= target_rows and n_cols <= target_cols:
            return [n_rows], [n_cols]

        # 用 ceil 保证每段严格 <= target（uniform_split 使用 round 可能产生比 target 大 1 的段）
        import math
        n_row_chunks = max(1, math.ceil(n_rows / target_rows))
        n_col_chunks = max(1, math.ceil(n_cols / target_cols))
        row_target_final = math.ceil(n_rows / n_row_chunks)
        col_target_final = math.ceil(n_cols / n_col_chunks)

        row_sizes = uniform_split(n_rows, row_target_final)
        col_sizes = uniform_split(n_cols, col_target_final)
        return row_sizes, col_sizes

    # ==================== 旧模式（无像素信息）====================
    if n_rows <= 20:
        row_target = n_rows
    elif n_rows <= 30:
        row_target = 15
    else:
        row_target = 20

    if n_cols <= 20:
        col_target = n_cols
    elif n_cols <= 30:
        col_target = 15
    else:
        col_target = 20

    row_sizes = uniform_split(n_rows, row_target)
    col_sizes = uniform_split(n_cols, col_target)
    return row_sizes, col_sizes


class TableChunker:
    """单表格切块器"""

    def __init__(self,
                 rows_per_chunk=None,
                 cols_per_chunk=None,
                 max_cols_before_hsplit=MAX_COLS_FULL_WIDTH,
                 header_coverage=HEADER_ROW_COVERAGE,
                 first_col_coverage=FIRST_COL_COVERAGE,
                 white_threshold=WHITE_THRESHOLD,
                 auto_granularity=True):
        """
        Args:
            rows_per_chunk: 每 chunk 行数（None 时用 auto_granularity）
            cols_per_chunk: 每 chunk 列数（None 时用 auto_granularity）
            auto_granularity: True 时按表格列数自动计算粒度（推荐）
        """
        self.rows_per_chunk = rows_per_chunk
        self.cols_per_chunk = cols_per_chunk
        self.max_cols_before_hsplit = max_cols_before_hsplit
        self.header_coverage = header_coverage
        self.first_col_coverage = first_col_coverage
        self.white_threshold = white_threshold
        self.auto_granularity = auto_granularity
        # Level 1.5 子表切分器（画线后、切 chunk 前，检测同 region 内多阶梯子表）
        self.subtable_splitter = SubTableSplitter()

    # ==================== 主流程 ====================

    @staticmethod
    def _uniform_group_indices(n_items: int, n_groups: int) -> List[List[int]]:
        """
        把 n_items 个索引均匀分成 n_groups 组。

        保证每组大小差异不超过 1，避免出现“大块 + 小尾巴”。

        例:
            _uniform_group_indices(10, 3) → [[0,1,2,3], [4,5,6], [7,8,9]]
            _uniform_group_indices(40, 2)  → [[0..19], [20..39]]
            _uniform_group_indices(5, 1)   → [[0,1,2,3,4]]
        """
        if n_groups <= 0:
            n_groups = 1
        if n_groups >= n_items:
            # 每个元素一组
            return [[i] for i in range(n_items)]
        base = n_items // n_groups
        extra = n_items % n_groups
        groups = []
        idx = 0
        for g in range(n_groups):
            size = base + (1 if g < extra else 0)
            groups.append(list(range(idx, idx + size)))
            idx += size
        return groups

    def chunk_table(self, table_img: Image.Image) -> List[Dict]:
        """
        对单个表格图片进行切块。

        简化流程：画线 → 沿 row_bounds/col_bounds 边界裁切 → 返回 chunks。
        不拼接表头/列头，直接切。

        Returns:
            list of dict: {'image', 'order', 'kind', 'row_range', 'col_range', 'bbox'}
        """
        w, h = table_img.size
        logger.info(f"Level2 切块: 表格尺寸={w}x{h}")

        # 假表格过滤（阈值 110，与 pipeline_v2 一致；曾试 200 导致真矮表误判已回退）
        if h < 110:
            return [{'image': table_img, 'order': 0, 'kind': 'body',
                     'row_range': (1, 1), 'col_range': (1, 1),
                     'bbox': (0, 0, w, h)}]

        # Step 1: 结构分析
        structure = self.analyze_structure(table_img)
        row_bounds = structure['row_bounds']
        col_bounds = structure['col_bounds']
        table_type = structure.get('table_type', 'borderless')

        # Step 2: 画线
        _orig_gray_pre_draw = None
        if table_type == 'borderless':
            # 画线前先存原图灰度（供错位标签行合并用，避免被绘制的网格线干扰内容判定）
            _orig_gray_pre_draw = np.array(table_img.convert('L'), dtype=np.uint8)
            table_img, h_lines, v_lines = self._draw_borderless_grid_lines(
                table_img, col_bounds_hint=col_bounds)
            # 用实际画线位置做切分
            all_h = sorted([0] + h_lines + [h])
            row_bounds = [(all_h[i], all_h[i+1]) for i in range(len(all_h)-1)
                          if all_h[i+1] - all_h[i] > 3]
            all_v = sorted([0] + v_lines + [w])
            col_bounds = [(all_v[i], all_v[i+1]) for i in range(len(all_v)-1)
                          if all_v[i+1] - all_v[i] > 3]
        elif table_type == 'semi_bordered':
            table_img = self._draw_bounds_grid_lines(
                table_img, row_bounds, col_bounds,
                first_row_idx=structure['first_row_idx'],
                first_col_idx=structure['first_col_idx'])
        # bordered: 不画线，直接用 analyze_structure 的 bounds
        # 过滤掉太窄的行/列（外框线边界，宽/高 < 10px）
        row_bounds = [(s, e) for s, e in row_bounds if e - s >= 10]
        col_bounds = [(s, e) for s, e in col_bounds if e - s >= 10]

        # 修复"数据带/标签带错位"导致的行翻倍（borderless 特有）：
        # 某些表行标签(1,2,3)相对数据值垂直偏移，形成"数据带 + 标签带"两条，被误检为 2 行。
        # 若"仅数据"行紧跟"仅标签"行的配对占比高，合并为一行。带占比护栏，正常表零影响。
        if table_type == 'borderless' and _orig_gray_pre_draw is not None:
            row_bounds = self._merge_offset_label_rows(_orig_gray_pre_draw, row_bounds, col_bounds)

        n_rows = len(row_bounds)
        n_cols = len(col_bounds)
        logger.info(f"  结构: rows={n_rows}, cols={n_cols}, type={table_type}")

        # 结构检测失败：整表当单个 chunk
        if n_rows == 0 or n_cols == 0:
            return [{'image': table_img, 'order': 0, 'kind': 'body',
                     'row_range': (1, 1), 'col_range': (1, 1),
                     'bbox': (0, 0, w, h)}]

        # Step 3: 均匀切分计划
        # 主约束：行列数 12×12（保证 API 输出不截断）
        # 兜底约束：单边像素上限 3000（防止超大 chunk）
        # 宽高比：≤4:1（允许略扁/略高的 chunk，避免为凑比例把 6 列切成 3 组）
        MAX_CELLS_PER_DIM = 12   # 每个 chunk 最多行/列数
        MAX_CHUNK_PX = 3000      # 单边像素兜底上限
        MAX_ASPECT_RATIO = 4.0   # 最大宽高比

        # 计算总高度和总宽度
        total_h = row_bounds[-1][1] - row_bounds[0][0]
        total_w = col_bounds[-1][1] - col_bounds[0][0]

        import math
        # 列约束（整表共享）
        n_col_groups_cells = max(1, math.ceil(n_cols / MAX_CELLS_PER_DIM))
        n_col_groups_px = max(1, math.ceil(total_w / MAX_CHUNK_PX))
        n_col_groups = max(n_col_groups_cells, n_col_groups_px)

        # ---- Level 1.5: 子表切分（画线之后、切 chunk 之前）----
        # 在画线后的行边界上检测多阶梯子表边界，让行分组不跨子表边界。
        # 传入 col_bounds → 用单元格法测内容右边缘，排除边框/网格线干扰（有边框表必需）。
        drawn_gray = np.array(table_img.convert('L'), dtype=np.uint8)
        cut_idxs = self.subtable_splitter.detect_row_cut_indices(
            drawn_gray, row_bounds, w, h, col_bounds, table_type)
        subtable_groups = []
        start = 0
        for c in cut_idxs:
            subtable_groups.append(list(range(start, c + 1)))
            start = c + 1
        subtable_groups.append(list(range(start, n_rows)))

        if len(subtable_groups) <= 1:
            # 单子表：原逻辑（完全不变，零回归）
            n_row_groups_cells = max(1, math.ceil(n_rows / MAX_CELLS_PER_DIM))
            n_row_groups_px = max(1, math.ceil(total_h / MAX_CHUNK_PX))
            n_row_groups = max(n_row_groups_cells, n_row_groups_px)
            row_groups = self._uniform_group_indices(n_rows, n_row_groups)
            col_groups = self._uniform_group_indices(n_cols, n_col_groups)
            row_group_st = [0] * len(row_groups)

            # 宽高比保护
            avg_chunk_h = total_h / n_row_groups
            avg_chunk_w = total_w / n_col_groups
            if avg_chunk_h > 0 and avg_chunk_w > 0:
                ratio = max(avg_chunk_w / avg_chunk_h, avg_chunk_h / avg_chunk_w)
                if ratio > MAX_ASPECT_RATIO:
                    if avg_chunk_w > avg_chunk_h:
                        n_col_groups = max(n_col_groups, math.ceil(total_w / (avg_chunk_h * MAX_ASPECT_RATIO)))
                        col_groups = self._uniform_group_indices(n_cols, n_col_groups)
                    else:
                        n_row_groups = max(n_row_groups, math.ceil(total_h / (avg_chunk_w * MAX_ASPECT_RATIO)))
                        row_groups = self._uniform_group_indices(n_rows, n_row_groups)
                        row_group_st = [0] * len(row_groups)
        else:
            # 多子表：每个子表内部独立均匀分组（行组不跨子表边界）
            logger.info(f"  [子表切分] 检测到 {len(subtable_groups)} 个子表: "
                        f"{[(g[0]+1, g[-1]+1) for g in subtable_groups]}")
            row_groups = []
            row_group_st = []
            for st_id, st_rows in enumerate(subtable_groups):
                n_st = len(st_rows)
                st_h = row_bounds[st_rows[-1]][1] - row_bounds[st_rows[0]][0]
                n_grp = max(math.ceil(n_st / MAX_CELLS_PER_DIM),
                            math.ceil(st_h / MAX_CHUNK_PX), 1)
                for grp_local in self._uniform_group_indices(n_st, n_grp):
                    row_groups.append([st_rows[i] for i in grp_local])
                    row_group_st.append(st_id)
            col_groups = self._uniform_group_indices(n_cols, n_col_groups)

            # 宽高比保护（多子表时只调列组，避免破坏子表行边界）
            avg_chunk_h = total_h / max(1, len(row_groups))
            avg_chunk_w = total_w / max(1, n_col_groups)
            if avg_chunk_h > 0 and avg_chunk_w > 0:
                ratio = max(avg_chunk_w / avg_chunk_h, avg_chunk_h / avg_chunk_w)
                if ratio > MAX_ASPECT_RATIO and avg_chunk_w > avg_chunk_h:
                    n_col_groups = max(n_col_groups, math.ceil(total_w / (avg_chunk_h * MAX_ASPECT_RATIO)))
                    col_groups = self._uniform_group_indices(n_cols, n_col_groups)

        logger.info(f"  切分计划: {len(row_groups)}行组 × {len(col_groups)}列组 = {len(row_groups)*len(col_groups)} chunks")

        # Step 4: 裁切
        chunks = []
        order = 0
        pad = 2  # 边缘padding确保边框线完整
        for rg_idx, rg in enumerate(row_groups):
            y0 = max(0, row_bounds[rg[0]][0] - pad)
            y1 = min(h, row_bounds[rg[-1]][1] + pad)
            for cg_idx, cg in enumerate(col_groups):
                x0 = max(0, col_bounds[cg[0]][0] - pad)
                x1 = min(w, col_bounds[cg[-1]][1] + pad)
                chunk_img = table_img.crop((x0, y0, x1, y1))

                # 裁掉底部空白：如果最后一条横线下方无有效内容，截断到横线位置
                chunk_arr = np.array(chunk_img.convert('L'), dtype=np.uint8)
                ch_h, ch_w = chunk_arr.shape
                if ch_h > 10:
                    # 从底部向上找最后一条横线（该行暗像素 > 10%）
                    last_hline_y = None
                    for yy in range(ch_h - 3, max(0, ch_h - 50), -1):
                        dark_ratio = (chunk_arr[yy, :] < 128).sum() / ch_w
                        if dark_ratio > 0.1:
                            last_hline_y = yy
                            break
                    if last_hline_y is not None and last_hline_y < ch_h - 3:
                        # 横线下方是否有有效内容（排除竖线延伸）
                        below = chunk_arr[last_hline_y + 2:, :]
                        if below.size > 0:
                            # 每列暗像素数，排除竖线位置（竖线列每行都暗）
                            col_dark = (below < 128).sum(axis=0)
                            below_h = below.shape[0]
                            # 非竖线列 = 不是每行都暗的列
                            non_vline_dark = col_dark[col_dark < below_h * 0.8].sum()
                            if non_vline_dark < below.size * 0.005:
                                # 下方无内容（排除竖线后），裁掉
                                chunk_img = chunk_img.crop((0, 0, ch_w, last_hline_y + 2))

                # 紧致裁切：去掉左右多余白边，避免 API 把白边误判为额外列
                chunk_img = self._tight_crop_horizontal(chunk_img, margin=3)

                chunks.append({
                    'image': chunk_img,
                    'order': order,
                    'kind': 'body',
                    'row_range': (rg[0] + 1, rg[-1] + 1),  # 1-based
                    'col_range': (cg[0] + 1, cg[-1] + 1),  # 1-based
                    'bbox': (x0, y0, x1, y1),
                    'subtable': row_group_st[rg_idx],
                })
                order += 1

        # 记录基本信息
        for c in chunks:
            c['table_type'] = table_type
            c['structure_rows'] = n_rows
            c['structure_cols'] = n_cols
            c['n_row_groups'] = len(row_groups)
            c['n_col_groups'] = len(col_groups)

        logger.info(f"  → {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _tight_crop_horizontal(img: Image.Image, margin: int = 5) -> Image.Image:
        """
        紧致裁切 chunk 四周多余白边，避免 API 把白边误判为额外行/列。

        策略：从每侧边缘向内扫描：
        - 先遇到“线”（暗像素占比 > 50%）→ 裁到线上（保留线作边框）
        - 先遇到“内容”（非线非空）→ 在内容前保留 margin
        - 都没遇到 → 保持原边缘
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        if h < 10 or w < 10:
            return img

        dark = gray < 200
        row_dark_ratio = dark.sum(axis=1) / w
        col_dark_ratio = dark.sum(axis=0) / h

        # === 水平方向 ===
        non_hline_mask = row_dark_ratio <= 0.5
        n_non_hline = non_hline_mask.sum()
        left, right = 0, w
        if n_non_hline >= 5:
            # 排除横线行后的每列暗像素（用于判断内容）
            col_dark = dark[non_hline_mask, :].sum(axis=0)
            vline_threshold_local = n_non_hline * 0.6
            min_col_content = max(3, int(n_non_hline * 0.08))
            # 表头区稀疏文字保护：横向表头文字（如"保单年度\投保年龄"）每列暗像素
            # 很少（2-8px），达不到 min_col_content（按整列多行设计），会被误裁。
            # 额外检查顶部表头区每列暗像素，有稀疏文字即停。
            header_h = min(30, h // 4)
            header_col_dark = dark[:header_h, :].sum(axis=0)

            # 从左边扫到中：先遇到竖线就停在竖线，先遇到内容就在内容前留 margin
            for x in range(w):
                if col_dark_ratio[x] > 0.5:  # 到达竖线
                    left = x
                    break
                if (col_dark[x] >= min_col_content and col_dark[x] < vline_threshold_local) \
                        or header_col_dark[x] >= 3:
                    # 到达内容（含表头稀疏文字）
                    left = max(0, x - margin)
                    break

            # 从右边扫到中
            for x in range(w - 1, -1, -1):
                if col_dark_ratio[x] > 0.5:  # 竖线
                    right = x + 1
                    break
                if (col_dark[x] >= min_col_content and col_dark[x] < vline_threshold_local) \
                        or header_col_dark[x] >= 3:
                    right = min(w, x + 1 + margin)
                    break

        # === 垂直方向 ===
        non_vline_mask = col_dark_ratio <= 0.5
        n_non_vline = non_vline_mask.sum()
        top, bottom = 0, h
        if n_non_vline >= 5:
            row_dark = dark[:, non_vline_mask].sum(axis=1)
            hline_threshold_local = n_non_vline * 0.6
            min_row_content = max(3, int(n_non_vline * 0.08))
            # 表头区稀疏文字保护（与水平方向同理）：表头文字行全行暗像素可能
            # 很少（如"投保年龄（周岁）"仅36px宽，全行占比<5%），达不到
            # min_row_content，导致垂直裁切跳过表头从下方开始，切掉文字顶部。
            # 额外检查顶部30px每行暗像素，有稀疏文字即停。
            header_v_h = min(30, h // 4)

            # 从上边扫到中
            for y in range(h):
                if row_dark_ratio[y] > 0.5:  # 横线
                    top = y
                    break
                if row_dark[y] >= min_row_content and row_dark[y] < hline_threshold_local:
                    top = max(0, y - margin)
                    break
                # 表头区稀疏文字：顶部30px内任何有内容的行都停
                if y < header_v_h and row_dark[y] >= 3:
                    top = max(0, y - margin)
                    break

            # 从下边扫到中
            for y in range(h - 1, -1, -1):
                if row_dark_ratio[y] > 0.5:  # 横线
                    bottom = y + 1
                    break
                if row_dark[y] >= min_row_content and row_dark[y] < hline_threshold_local:
                    bottom = min(h, y + 1 + margin)
                    break

        # 安全检查
        if right - left < 50 or bottom - top < 20:
            return img

        # 只有实际需要裁时才裁
        if left + (w - right) + top + (h - bottom) < 5:
            return img

        return img.crop((left, top, right, bottom))


    # ==================== Step 1: 结构分析 ====================

    def analyze_structure(self, img: Image.Image) -> Dict:
        """
        全面分析表格结构。

        策略：**优先尝试边框检测**（对有边框表格更准确），失败再用密度分析。

        Returns:
            dict:
                'row_bounds': list of (y_start, y_end)
                'col_bounds': list of (x_start, x_end)
                'first_row_idx': int | None
                'first_col_idx': int | None
                'top_extra_y': int | None
                'left_extra_x': int | None
                'has_borders': bool  # 是否检测到边框
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 优先尝试边框检测
        row_bounds, col_bounds, has_borders, n_h_lines, n_v_lines = \
            self._detect_bordered_bounds(gray, h, w)
        total_lines = n_h_lines + n_v_lines

        # 表格类型分类
        if total_lines >= 10:
            table_type = 'bordered'     # 有边框
        elif total_lines > 0:
            table_type = 'semi_bordered'  # 半边框
        else:
            table_type = 'borderless'    # 无边框

        # 合理性检查：如果边框检测得到的行高过大（可能漏检了行）
        # 或行数远少于密度分析结果，用空白间隙检测的行边界替代（列边界保留边框结果）
        # 用完全空白检测（而非密度分析），避免误分
        if has_borders and len(row_bounds) > 0:
            avg_row_h = h / len(row_bounds)
            # 表格行高一般 20-50px，若平均 > 80px 说明可能把多行合并了
            if avg_row_h > 80:
                blank_row_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=False)
                if len(blank_row_bounds) > len(row_bounds) * 2:
                    logger.info(
                        f"    边框行高偏大 ({avg_row_h:.0f}px/行, {len(row_bounds)}行)，"
                        f"空白间隙检测到 {len(blank_row_bounds)} 行 → 改用空白间隙行边界"
                    )
                    row_bounds = blank_row_bounds

        # 半边框补行：有边框但 row_bounds 为空（如只有顶/底外框 2 条横线）
        # → 用空白间隙检测行边界
        if has_borders and len(row_bounds) == 0:
            row_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=False)
            logger.info(f"    半边框补行(空白间隙): 检测到 {len(row_bounds)} 行")

        # 半边框处理：有横线但列边界为空 → 用完全空白列检测补列边界
        if has_borders and len(col_bounds) == 0 and len(row_bounds) > 0:
            col_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=True)
            logger.info(f"    半边框补列(空白间隙): 检测到 {len(col_bounds)} 列")

        # 列宽合理性检查：如果平均列宽 > 500px 且表格宽 > 1000px，
        # 说明 CV 检测的竖线太少（如只有外框），用空白间隙检测替代列边界
        # （214px/列对有边框表格正常，不应触发；1952px/列才是真正异常）
        # 用完全空白检测（而非密度分析），避免把数字内小间隙误判为列分隔
        if has_borders and len(col_bounds) > 0 and w > 1000:
            avg_col_w = w / len(col_bounds)
            if avg_col_w > 500:
                blank_col_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=True)
                if len(blank_col_bounds) > len(col_bounds) * 2:
                    logger.info(
                        f"    列宽偏大 ({avg_col_w:.0f}px/列, {len(col_bounds)}列)，"
                        f"空白间隙检测到 {len(blank_col_bounds)} 列 → 改用空白间隙列边界"
                    )
                    col_bounds = blank_col_bounds

        if not has_borders:
            # 无边框：用完全空白检测（与画线逻辑一致）
            row_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=False)
            col_bounds = self._detect_bounds_by_blank_gaps(gray, h, w, is_col=True)

        # 半边框行精化：空白间隙检测可能把多个密集数据行合并（行间无完全空白），
        # 用 Otsu 自适应阈值尝试拆分。仅当 Otsu 检测出明显更多且间距规律的行时才替代。
        if table_type == 'semi_bordered' and len(row_bounds) > 0:
            row_bounds = self._refine_rows_by_otsu(gray, h, w, row_bounds)

        # 找表格真正的第一行
        first_row_idx = self._find_first_row(gray, row_bounds, col_bounds, w)

        # 半边框列碎片合并：空白间隙检测可能把"数据窄但列头文字宽"的列拆成多个
        # 碎片 bound（如"保单年度"列数据只有个位数，但列头文字横跨多个碎片）。
        # 合并"列头行有内容横跨且两侧 bound 都窄"的相邻碎片。
        if table_type == 'semi_bordered' and len(col_bounds) > 2 and first_row_idx is not None:
            col_bounds = self._merge_header_spanning_cols(
                gray, row_bounds, col_bounds, first_row_idx)

        # 半边框列漏检补偿：阶梯表右侧稀疏列内容远低于全局数据列，
        # 全局空白阈值（P25*1.5）把整个右侧区域当成空白，导致列漏检。
        # 用局部阈值在右侧区域补检。
        if table_type == 'semi_bordered' and len(col_bounds) > 0:
            col_bounds = self._extend_tail_cols(gray, col_bounds, h, w)

        # 半边框间距校验：基于线间距分布，拆分过宽 bound（漏画线）。
        # 某个 bound 宽度明显大于其他（>中位数*1.8），说明中间漏画了线，拆分。
        if table_type == 'semi_bordered':
            row_bounds = self._validate_bounds_by_spacing(gray, row_bounds, h, is_col=False)
            col_bounds = self._validate_bounds_by_spacing(gray, col_bounds, w, is_col=True)
            # 校验后 bounds 变了，重新算 first_row_idx
            first_row_idx = self._find_first_row(gray, row_bounds, col_bounds, w)

        # 找表格真正的第一列
        first_col_idx = self._find_first_col(gray, row_bounds, col_bounds, h)

        # 判断第一行之上是否有额外内容
        top_extra_y = None
        if first_row_idx is not None and first_row_idx > 0:
            top_extra_y = row_bounds[first_row_idx][0]

        # 判断第一列之左是否有额外内容
        left_extra_x = None
        if first_col_idx is not None and first_col_idx > 0:
            left_extra_x = col_bounds[first_col_idx][0]

        return {
            'row_bounds': row_bounds,
            'col_bounds': col_bounds,
            'first_row_idx': first_row_idx,
            'first_col_idx': first_col_idx,
            'top_extra_y': top_extra_y,
            'left_extra_x': left_extra_x,
            'has_borders': has_borders,
            'table_type': table_type,
            'n_h_lines': n_h_lines,
            'n_v_lines': n_v_lines,
        }

    def _detect_bordered_bounds(self, gray: np.ndarray, h: int, w: int
                                 ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], bool, int, int]:
        """
        对表格用形态学检测水平/垂直分隔线，直接得到行列边界。

        分类（基于 h_lines + v_lines 总数）：
        - ≥ 10: 有边框表格 → 返回精确 row_bounds + col_bounds
        - 1-9:  半边框表格 → 返回部分边界
        - = 0:  无边框表格 → 返回 ([], [], False, 0, 0)

        Returns:
            (row_bounds, col_bounds, has_borders, n_h_lines, n_v_lines)
        """
        if not HAS_CV2:
            return [], [], False, 0, 0

        try:
            _, binary = cv2.threshold(gray, 0, 255,
                                       cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # 检测水平线：宽核（宽度至少图宽 30%；矮表保底 min(100, 70%宽度)）
            # 矮/窄表格的边框线长度可能 < 100px，固定 100 保底会漏检
            h_kernel_w = max(int(w * 0.3), min(100, int(w * 0.7)))
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_w, 1))
            h_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
            row_sums = h_lines_img.sum(axis=1)
            h_line_y = np.where(row_sums > 0)[0]
            h_lines = self._merge_close_positions(h_line_y, gap=5)

            # 检测竖线：高核（高度至少图高 30%；矮表保底 min(100, 70%高度)）
            v_kernel_h = max(int(h * 0.3), min(100, int(h * 0.7)))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_h))
            v_lines_img = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)
            col_sums = v_lines_img.sum(axis=0)
            v_line_x = np.where(col_sums > 0)[0]
            v_lines = self._merge_close_positions(v_line_x, gap=5)

            n_h = len(h_lines)
            n_v = len(v_lines)
            total = n_h + n_v

            # 有边框表格：横线 + 竖线 >= 10
            if total >= 10 and n_h >= 3 and n_v >= 3:
                row_bounds = []
                for i in range(n_h - 1):
                    y0, y1 = h_lines[i], h_lines[i + 1]
                    if y1 - y0 >= 8:
                        row_bounds.append((y0, y1))

                col_bounds = []
                for i in range(n_v - 1):
                    x0, x1 = v_lines[i], v_lines[i + 1]
                    if x1 - x0 >= 8:
                        col_bounds.append((x0, x1))

                if row_bounds and col_bounds:
                    logger.info(f"    边框检测: {n_h} 条横线, {n_v} 条竖线 "
                                f"(total={total}) → 有边框 "
                                f"{len(row_bounds)} 行 × {len(col_bounds)} 列")
                    return row_bounds, col_bounds, True, n_h, n_v
                return [], [], False, n_h, n_v

            # 半边框：有一些线但不够多
            if total > 0:
                row_bounds = []
                if n_h >= 3:
                    for i in range(n_h - 1):
                        y0, y1 = h_lines[i], h_lines[i + 1]
                        if y1 - y0 >= 8:
                            row_bounds.append((y0, y1))
                col_bounds = []
                if n_v >= 3:
                    for i in range(n_v - 1):
                        x0, x1 = v_lines[i], v_lines[i + 1]
                        if x1 - x0 >= 8:
                            col_bounds.append((x0, x1))
                logger.info(f"    边框检测: {n_h} 条横线, {n_v} 条竖线 "
                            f"(total={total}) → 半边框")
                if row_bounds or col_bounds:
                    return row_bounds, col_bounds, True, n_h, n_v
                return [], [], False, n_h, n_v

            # 无边框
            return [], [], False, 0, 0

        except Exception as e:
            logger.warning(f"    边框检测异常: {e}")
            return [], [], False, 0, 0

    @staticmethod
    def _merge_close_positions(positions: np.ndarray, gap: int = 5) -> List[int]:
        """
        合并接近的位置（同一条边框线的多像素）。取中点。
        """
        if len(positions) == 0:
            return []
        positions = sorted(positions.tolist())
        merged = []
        start = positions[0]
        prev = positions[0]
        for p in positions[1:]:
            if p - prev > gap:
                merged.append((start + prev) // 2)
                start = p
            prev = p
        merged.append((start + prev) // 2)
        return merged

    def _merge_offset_label_rows(self, gray: np.ndarray,
                                 row_bounds: List[Tuple[int, int]],
                                 col_bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """修复"数据带/标签带错位"导致的行翻倍。

        某些 borderless 表的行标签列(如"保单年度"1,2,3)相对数据值垂直偏移，
        一个逻辑行被拆成"仅数据"带 + "仅标签"带两条。这里把"仅数据行紧跟仅标签行"
        （或反序）的配对合并为一行。带占比护栏（配对≥25%行数才生效），正常表零影响。
        """
        n = len(row_bounds)
        if n < 6 or len(col_bounds) < 2:
            return row_bounds
        x_lab0, x_lab1 = col_bounds[0]
        x_dat0, x_dat1 = col_bounds[1][0], col_bounds[-1][1]

        def has_ink(y0, y1, x0, x1):
            sub = gray[y0:y1, x0:x1]
            return sub.size > 0 and float((sub < 160).mean()) > 0.012

        # 每行分类：(是否标签列有内容, 是否数据列有内容)
        cls = []
        for (s, e) in row_bounds:
            cls.append((has_ink(s, e, x_lab0, x_lab1), has_ink(s, e, x_dat0, x_dat1)))

        LABEL_ONLY = (True, False)
        DATA_ONLY = (False, True)
        merged, pairs, i = [], 0, 0
        while i < n:
            if i + 1 < n and (
                (cls[i] == DATA_ONLY and cls[i + 1] == LABEL_ONLY) or
                (cls[i] == LABEL_ONLY and cls[i + 1] == DATA_ONLY)):
                merged.append((row_bounds[i][0], row_bounds[i + 1][1]))
                pairs += 1
                i += 2
            else:
                merged.append(row_bounds[i])
                i += 1
        if pairs >= max(3, int(n * 0.25)):
            logger.info(f"  [错位标签行合并] {n} 行 → {len(merged)} 行（合并 {pairs} 对数据/标签带）")
            return merged
        return row_bounds


    def _split_dense_segment(self, row_content: np.ndarray,
                              y_start: int, y_end: int) -> List[Tuple[int, int]]:
        """
        对密集内容段按行密度局部低点做行边界检测。

        原理：即使密集表行间没有完全空白，每行文字之间还是会有**密度相对低**的位置
        （比如数字上下留白）。用滑动窗口找局部最小值作为行分界。
        """
        seg = row_content[y_start:y_end].astype(np.float32)
        seg_h = y_end - y_start

        # 用较宽窗口平滑（去噪）
        smooth_window = 3
        smoothed = np.zeros_like(seg)
        for i in range(len(seg)):
            lo = max(0, i - smooth_window)
            hi = min(len(seg), i + smooth_window + 1)
            smoothed[i] = seg[lo:hi].mean()

        # 找局部最小值：在 min_distance 窗口内的最小值
        # 估算行高：先粗略估计为 30px，然后 min_distance = 行高的一半
        est_row_h = 25
        half_window = max(5, est_row_h // 3)

        low_points = []
        for i in range(half_window, len(smoothed) - half_window):
            window_vals = smoothed[i - half_window:i + half_window + 1]
            window_min = window_vals.min()
            window_max = window_vals.max()
            # 相对差异 > 20%，且当前是窗口最小
            if smoothed[i] == window_min and (window_max - window_min) > window_max * 0.15:
                low_points.append(i)

        # 合并太近的低点
        if not low_points:
            return []
        merged = [low_points[0]]
        min_row_h = 12
        for p in low_points[1:]:
            if p - merged[-1] >= min_row_h:
                merged.append(p)

        # 从低点位置构造行边界
        if len(merged) < 2:
            return []

        # 每两个相邻低点之间是一行
        result = []
        prev = 0
        for p in merged:
            if p - prev >= min_row_h:
                result.append((y_start + prev, y_start + p))
                prev = p
        # 最后一行到 seg 结尾
        if seg_h - prev >= min_row_h:
            result.append((y_start + prev, y_end))

        return result

    def _detect_bounds_by_blank_gaps(self, gray: np.ndarray, h: int, w: int,
                                       is_col: bool = True) -> List[Tuple[int, int]]:
        """
        用**几乎完全空白**行/列检测表格边界（适用于半边框表格）。

        算法：
        1. 计算每列/行的像素密度分布
        2. 动态阈值：用 P25 分位数作为"背景噪声"上界（边框贡献）
        3. 密度 <= 阈值的位置视为"blank"（无数据内容）
        4. 分组为连续 bands，自适应筛选真正的列间隙（宽度双峰跳变）

        Args:
            is_col: True 检测列（垂直），False 检测行（水平）

        Returns:
            边界列表 [(start, end), ...]
        """
        content_mask = gray < self.white_threshold
        if is_col:
            content_axis = content_mask.sum(axis=0)
            total_len = w
            # 对于列检测：只用"内容高"的行来分析列间隙
            # 稀疏表格（阶梯型/上三角）大部分行是空的或只有边框，
            # 用中位数以上的行做列分析，避免被空行稀释
            row_content = content_mask.sum(axis=1)
            if len(row_content) > 10:
                row_median = float(np.median(row_content))
                if row_median > 0:
                    # 用中位数作为门槛：只保留内容 > median 的行
                    data_row_mask = row_content > row_median
                    n_data_rows = int(data_row_mask.sum())
                    # 至少 5 行才启用，且不能把所有行都过滤掉
                    if n_data_rows >= 5 and n_data_rows < len(row_content):
                        content_axis = content_mask[data_row_mask, :].sum(axis=0)
        else:
            content_axis = content_mask.sum(axis=1)
            total_len = h
            # 对于行检测：只用"内容高"的列来分析行间隙
            col_content = content_mask.sum(axis=0)
            if len(col_content) > 10:
                col_median = float(np.median(col_content))
                if col_median > 0:
                    data_col_mask = col_content > col_median
                    n_data_cols = int(data_col_mask.sum())
                    if n_data_cols >= 5 and n_data_cols < len(col_content):
                        content_axis = content_mask[:, data_col_mask].sum(axis=1)

        # 动态阈值：用 P25 * 1.5 作为"blank"上界
        # 逻辑：数据列内容量高，间隙列只有边框像素
        # P25 通常处于"间隙"数量级；乘 1.5 留些余量
        sorted_vals = np.sort(content_axis)
        p25_val = float(sorted_vals[len(sorted_vals) // 4])
        # 保底：绝对上界不超过 max*0.4（防止数据列被误判）
        # cap 从 0.15 放宽到 0.4：对内容压缩的表格（如短表 h<200）
        # max 值本身很小，0.15 cap 会把 P25*1.5 的合理阈值卡下来
        max_val = float(content_axis.max())
        blank_threshold = int(min(p25_val * 1.5, max_val * 0.4))
        blank_threshold = max(5, blank_threshold)  # 最少 5 像素容忍度

        # "几乎空白"位置
        blank_positions = np.where(content_axis <= blank_threshold)[0]
        if len(blank_positions) == 0:
            return [(0, total_len)]

        # 分组为连续 bands
        bands = self._group_consecutive(blank_positions)

        # 分析用 bands：排除边缘和大离群 bands
        # - 边缘 band：图片外边距，不是列间隙
        # - 大离群 band（宽度 > total_len*10%）：大片空白区域，不是列间隙
        # 但这些 band 仍作为切分点保留（在 real_gaps 中）
        max_analysis_width = max(50, int(total_len * 0.1))
        analysis_bands = [b for b in bands
                          if b[0] > 0 and b[1] < total_len
                          and (b[1] - b[0]) <= max_analysis_width]
        if not analysis_bands:
            analysis_bands = bands  # fallback

        # 自适应阈值（基于内部 bands）
        widths = sorted([e - s for s, e in analysis_bands])
        threshold = 4
        if len(widths) >= 2:
            from collections import Counter as _Counter
            width_counts = _Counter(widths)
            distinct = sorted(width_counts.keys())
            total_bands = len(widths)

            # 特殊情况：如果 >=80% 的 bands 集中在单一宽度，说明这是规律行/列间距
            # 全部作为真间隙，不过滤（阈值保持默认 4，或该主宽度）
            most_common_width, most_common_count = width_counts.most_common(1)[0]
            if most_common_count > total_bands * 0.8:
                threshold = max(1, most_common_width - 1)  # 主宽度都通过
            else:
                # 常规双峰过滤：找到"噪声→真间隙"的分界即 break
                # 关键：n_below 必须在 [20%, 90%] 范围内才算有意义的分界
                # - < 20%: 分界太靠前（噪声太少，可能主体是真间隙的单峰分布）
                # - > 90%: 分界太靠后（真间隙集群内部的划分，会过滤真间隙）
                for i in range(len(distinct) - 1):
                    w_lo = distinct[i]
                    w_hi = distinct[i + 1]
                    gap = w_hi - w_lo
                    rule_a = gap > 2 and w_lo <= 6
                    # 规则 B：大跳变（w_hi >= w_lo*1.5 且 gap >= 10）
                    rule_b = w_hi >= w_lo * 1.5 and gap >= 10
                    if rule_a or rule_b:
                        n_below = sum(width_counts[dw] for dw in distinct[:i + 1])
                        pct = n_below / total_bands
                        if 0.2 <= pct <= 0.9:  # 有意义的双峰分界
                            threshold = w_lo + 1
                            break

        real_gaps = [(s, e) for s, e in bands if (e - s) >= threshold]

        # 过滤 real_gaps 里的离群大 band（大空白区域，不是真正的列间隙）
        # 如：表格末端的空白区、上三角表格的空白角
        # 用真间隙宽度的中位数作为参考，超过 5x median 视为异常
        outlier_bands = []
        if len(real_gaps) >= 3:
            gap_widths = sorted([e - s for s, e in real_gaps])
            median_gw = gap_widths[len(gap_widths) // 2]
            outlier_cap = max(30, median_gw * 5)
            new_real_gaps = []
            for s, e in real_gaps:
                if (e - s) <= outlier_cap:
                    new_real_gaps.append((s, e))
                else:
                    outlier_bands.append((s, e))
            real_gaps = new_real_gaps

        # 确定有效数据范围：如果末端有大离群 band 触到边界，截断 total_len
        # 类似地，如果起始有大离群 band 从 0 开始，跳过它
        # 用 total_len 的 2% 作为"贴边"容忍度
        edge_tol = max(20, int(total_len * 0.02))
        effective_start = 0
        effective_end = total_len
        for s, e in outlier_bands:
            if e >= total_len - edge_tol:  # 触到右/下边缘
                effective_end = min(effective_end, s)
            if s <= edge_tol:  # 触到左/上边缘
                effective_start = max(effective_start, e)

        cut_points = [effective_start]
        for s, e in real_gaps:
            mid = (s + e) // 2
            if effective_start < mid < effective_end:
                cut_points.append(mid)
        cut_points.append(effective_end)

        result = []
        for i in range(len(cut_points) - 1):
            start = cut_points[i]
            end = cut_points[i + 1]
            if end - start >= 5:
                result.append((start, end))
        return result

    def _refine_rows_by_otsu(self, gray: np.ndarray, h: int, w: int,
                              row_bounds: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        半边框行精化：用 Otsu 自适应阈值拆分被合并的密集数据行。

        背景：_detect_bounds_by_blank_gaps 用 P25*1.5 作为空白阈值，对密集数据行
        （行间无完全空白，全表投影叠加后行间 content 仍超阈值）会把多行合并成
        一个大 bound，导致画线时缺少横线。Otsu 阈值能自适应找到"间隙/数据行"
        的双峰分界，拆出被合并的行。

        启用判据（保守，避免误检）：
        - Otsu 空白带数 >= 15（排除行少表格的内容波动误检）
        - Otsu 空白带数 > 当前行数 * 1.5（确认当前方法合并了行）
        - 空白带间距中位数 < 50px（确认是密集小行，排除大行表格）

        Args:
            gray: 灰度图
            h, w: 图高/宽
            row_bounds: 当前行边界（空白间隙检测结果）

        Returns:
            精化后的 row_bounds（满足判据）或原 row_bounds（不满足）
        """
        if not HAS_CV2:
            return row_bounds

        content_mask = gray < self.white_threshold
        row_content = content_mask.sum(axis=1).astype(np.float32)
        max_val = float(row_content.max())
        if max_val <= 0:
            return row_bounds

        # Otsu 阈值（归一化到 0-255 再映射回原始量纲）
        norm = (row_content / max_val * 255).astype(np.uint8)
        otsu_th_norm, _ = cv2.threshold(norm.reshape(-1, 1), 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_th = otsu_th_norm / 255.0 * max_val

        # 找空白带（content <= Otsu 阈值）
        blank_rows = np.where(row_content <= otsu_th)[0]
        if len(blank_rows) == 0:
            return row_bounds
        bands = self._group_consecutive(blank_rows)
        # 过滤：太窄（<2px，噪声）和太宽（>h*20%，大空白区域）的带
        bands = [(s, e) for s, e in bands if 2 <= (e - s) <= h * 0.2]

        # 判据 1：空白带数 >= 15
        if len(bands) < 15:
            return row_bounds
        # 判据 2：空白带数 > 当前行数 * 1.5（确认合并）
        if len(bands) <= len(row_bounds) * 1.5:
            return row_bounds
        # 判据 3：空白带间距中位数 < 50px（密集小行）
        centers = [(s + e) // 2 for s, e in bands]
        gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
        if not gaps or float(np.median(gaps)) >= 50:
            return row_bounds

        # 用空白带中点作为 cut_point，生成新 row_bounds
        cut_points = [0]
        for s, e in bands:
            cut_points.append((s + e) // 2)
        cut_points.append(h)
        # 去重 + 排序
        cut_points = sorted(set(cut_points))

        new_bounds = []
        for i in range(len(cut_points) - 1):
            start, end = cut_points[i], cut_points[i + 1]
            if end - start >= 5:
                new_bounds.append((start, end))

        if len(new_bounds) <= len(row_bounds):
            return row_bounds

        logger.info(f"    半边框行精化(Otsu): {len(row_bounds)} 行 → {len(new_bounds)} 行 "
                     f"(otsu_th={otsu_th:.0f}, bands={len(bands)}, "
                     f"gap_median={float(np.median(gaps)):.0f}px)")
        return new_bounds

    def _merge_header_spanning_cols(self, gray: np.ndarray,
                                     row_bounds: List[Tuple[int, int]],
                                     col_bounds: List[Tuple[int, int]],
                                     first_row_idx: int) -> List[Tuple[int, int]]:
        """
        半边框列碎片合并：合并"列头行有内容横跨且两侧 bound 都窄"的相邻碎片。

        背景：空白间隙检测按数据区的空白分列。对"数据窄但列头文字宽"的列
        （如"保单年度"列数据只有个位数 x≈30-50，但列头文字横跨 x≈20-80），
        数据区的空白间隙（x=26,70）被当成列边界，把这一列拆成多个碎片。
        这些碎片边界在列头行有文字横跨，说明不是真正的列间隙。

        合并判据（保守，避免合并真实列边界）：
        - cut_point 在列头行中间区域（排除上下 25% 边界）有内容（暗像素）
        - 且合并后 bound 宽度 < 所有 bound 宽度中位数 * 1.3
          （"保单年度"单列合并后约 90px < 中位数 76*1.3=98.8；
           投保年龄列+第1列合并后 73px > 中位数 48*1.3=62.4，不合并）

        Args:
            gray: 灰度图
            row_bounds: 行边界
            col_bounds: 列边界
            first_row_idx: 表头行索引

        Returns:
            合并后的 col_bounds
        """
        if first_row_idx is None or first_row_idx >= len(row_bounds):
            return col_bounds
        if len(col_bounds) < 2:
            return col_bounds

        content_mask = gray < self.white_threshold
        # 列头区域：first_row_idx 的行 + 1 行（覆盖"保单年度"文字可能在下一行的情况，
        # 如 6dfcd28f first_row_idx=0 指向标题行 (0,10)，但列头文字在 row_bounds[1]）
        header_end_idx = min(first_row_idx + 1, len(row_bounds) - 1)
        hy0 = row_bounds[first_row_idx][0]
        hy1 = row_bounds[header_end_idx][1]
        hpad = max(2, int((hy1 - hy0) * 0.15))
        header_strip = content_mask[hy0 + hpad: hy1 - hpad, :]
        if header_strip.size == 0:
            return col_bounds

        widths = [e - s for s, e in col_bounds]
        median_w = float(np.median(widths))
        merge_cap = median_w * 1.3

        # 迭代合并直到稳定
        merged = list(col_bounds)
        changed = True
        while changed:
            changed = False
            for i in range(len(merged) - 1):
                x_cut = merged[i][1]  # = merged[i+1][0]
                # 合并后宽度
                w_merged = merged[i + 1][1] - merged[i][0]
                if w_merged >= merge_cap:
                    continue
                # cut_point 在列头行有内容（±2px 范围内有暗像素）
                x0 = max(0, x_cut - 2)
                x1 = min(gray.shape[1], x_cut + 3)
                if header_strip[:, x0:x1].sum() > 0:
                    # 合并 merged[i] 和 merged[i+1]
                    merged[i] = (merged[i][0], merged[i + 1][1])
                    del merged[i + 1]
                    changed = True
                    break

        if len(merged) < len(col_bounds):
            logger.info(f"    半边框列碎片合并: {len(col_bounds)} 列 → {len(merged)} 列 "
                         f"(median_w={median_w:.0f}, cap={merge_cap:.0f})")
        return merged

    def _extend_tail_cols(self, gray: np.ndarray,
                           col_bounds: List[Tuple[int, int]],
                           h: int, w: int) -> List[Tuple[int, int]]:
        """
        半边框列漏检补偿：用局部阈值在右侧稀疏区域补检列。

        背景：阶梯表右侧稀疏列（如 4e32dba9 的列 101-106）内容（3-55）远低于
        全局数据列（200-400），全局空白阈值（P25*1.5=54）把整个右侧区域当成
        空白，导致这些列漏检、画线时缺竖线。此方法在右侧区域用局部阈值
        （局部 P25*1.5）重新检测列间隙。

        启用条件：col_bounds 右端 < w - 50（右侧有超过 50px 的未检测区域），
        且该区域有内容（不是纯空白边距）。

        Args:
            gray: 灰度图
            col_bounds: 当前列边界
            h, w: 图高/宽

        Returns:
            补检后的 col_bounds（无漏检则返回原值）
        """
        if not col_bounds:
            return col_bounds
        tail_start = col_bounds[-1][1]
        if tail_start >= w - 50:
            return col_bounds  # 右侧无足够未检测区域

        content_mask = gray < self.white_threshold
        # 用与 _detect_bounds_by_blank_gaps 相同的行过滤（内容高的行）
        row_content = content_mask.sum(axis=1)
        row_median = float(np.median(row_content))
        if row_median > 0:
            data_row_mask = row_content > row_median
            if data_row_mask.sum() >= 5:
                col_content = content_mask[data_row_mask, :].sum(axis=0)
            else:
                col_content = content_mask.sum(axis=0)
        else:
            col_content = content_mask.sum(axis=0)

        # 右侧区域 [tail_start, w)
        tail_content = col_content[tail_start:w].astype(np.float32)
        if tail_content.size == 0 or tail_content.max() <= 3:
            return col_bounds  # 右侧纯空白边距

        # 局部阈值：局部 P25 * 1.5
        sorted_vals = np.sort(tail_content)
        local_p25 = float(sorted_vals[len(sorted_vals) // 4])
        local_max = float(tail_content.max())
        local_th = max(3, int(min(local_p25 * 1.5, local_max * 0.4)))

        # 找空白列（content <= 局部阈值）
        blank_cols = np.where(tail_content <= local_th)[0]
        if len(blank_cols) == 0:
            return col_bounds
        bands = self._group_consecutive(blank_cols)
        # 过滤太窄的 band（<3px，噪声）和太宽的 band（>剩余宽度 50%，大空白）
        remaining = w - tail_start
        bands = [(s, e) for s, e in bands
                 if 3 <= (e - s) <= remaining * 0.5]
        if not bands:
            return col_bounds

        # 在 band 中点切 cut_point，生成子区域 col_bounds
        cut_points = [0]  # 相对 tail_start
        for s, e in bands:
            cut_points.append((s + e) // 2)
        cut_points.append(remaining)
        cut_points = sorted(set(cut_points))

        new_tail = []
        for i in range(len(cut_points) - 1):
            s = tail_start + cut_points[i]
            e = tail_start + cut_points[i + 1]
            if e - s >= 5:
                new_tail.append((s, e))

        if not new_tail:
            return col_bounds

        # 合并补检碎片：宽度 < new_tail 中位数 * 0.5 的碎片合并到相邻更宽的 bound
        if len(new_tail) >= 3:
            tail_widths = [e - s for s, e in new_tail]
            tail_median = float(np.median(tail_widths))
            frag_cap = tail_median * 0.5
            merged_tail = [new_tail[0]]
            for s, e in new_tail[1:]:
                prev_s, prev_e = merged_tail[-1]
                # 当前是碎片 → 合并到上一个
                if e - s < frag_cap:
                    merged_tail[-1] = (prev_s, e)
                # 上一个是碎片且当前更宽 → 合并到当前
                elif prev_e - prev_s < frag_cap and e - s >= frag_cap:
                    merged_tail[-1] = (prev_s, e)
                else:
                    merged_tail.append((s, e))
            new_tail = merged_tail

        result = list(col_bounds) + new_tail
        logger.info(f"    半边框列漏检补偿: 右侧 {remaining}px 补检 {len(new_tail)} 列 "
                     f"(local_th={local_th}, bands={len(bands)})")
        return result

    def _validate_lines_by_spacing(self, existing: List[int],
                                     candidate: List[int],
                                     gray: np.ndarray,
                                     range_start: int, range_end: int,
                                     is_h: bool,
                                     grid_a: int, grid_b: int) -> List[int]:
        """
        统一线间距校验：结合原表已有线 + 候选线，按间距分布决定最终画哪些候选线。

        校验思路（用户澄清）：
        1. 合并 existing + candidate 为完整线序列（标记来源）
        2. 相邻线间距 < 中位数*0.4 → 打架，删除候选线（保留 existing 或位置更居中的）
        3. 相邻线间距 > 中位数*1.8 → 漏画，在中间补候选线（用 content 最小值定位）

        Args:
            existing: 原表已有线位置
            candidate: 候选线位置（bounds 边界）
            gray: 灰度图
            range_start, range_end: 校验范围（行=top/bot，列=left/right）
            is_h: True=横线，False=竖线
            grid_a, grid_b: 垂直方向范围（横线用[left,right]，竖线用[top,bot]）

        Returns:
            候选线中最终要画的位置列表
        """
        # 标记来源: 'E' = existing, 'C' = candidate
        marked = [(p, 'E') for p in existing] + [(p, 'C') for p in candidate]
        marked.sort(key=lambda x: x[0])

        def _line_content_score(p: int, radius: int = 3) -> int:
            """画线位置附近的内容量；越小越像真实空白间隙。"""
            if is_h:
                y0 = max(0, p - radius)
                y1 = min(gray.shape[0], p + radius + 1)
                strip = gray[y0:y1, max(0, grid_a):min(gray.shape[1], grid_b)]
                if strip.size == 0:
                    return 10 ** 9
                return int((strip < self.white_threshold).sum(axis=1).max())
            x0 = max(0, p - radius)
            x1 = min(gray.shape[1], p + radius + 1)
            strip = gray[max(0, grid_a):min(gray.shape[0], grid_b), x0:x1]
            if strip.size == 0:
                return 10 ** 9
            return int((strip < self.white_threshold).sum(axis=0).max())

        def _choose_safer_line(positions: List[int]) -> int:
            """近距离候选合并时，不盲目取均值，优先选内容最少的位置。"""
            center = sum(positions) / len(positions)
            return min(positions, key=lambda p: (_line_content_score(p), abs(p - center)))

        if len(marked) < 2:
            return [p for p, t in marked if t == 'C']

        # Step 1: 合并近距离线（gap<3），优先保留 existing
        # 若两条线 gap<3，且一条是 E 一条是 C，删 C；若都是 C，选内容量最低的位置；都是 E 保留两个
        merged = [marked[0]]
        for p, t in marked[1:]:
            last_p, last_t = merged[-1]
            if p - last_p < 3:
                if last_t == 'E' and t == 'C':
                    pass  # 保留 last E，丢弃 C
                elif last_t == 'C' and t == 'E':
                    merged[-1] = (p, 'E')  # 用 E 替代 C
                elif last_t == 'C' and t == 'C':
                    merged[-1] = (_choose_safer_line([last_p, p, (last_p + p) // 2]), 'C')
                # 都是 E 时保留两个（不常见，位置真的很近）
                else:
                    merged.append((p, t))
            else:
                merged.append((p, t))

        # Step 2: 间距校验
        # 只用 candidate 间距估算中位数（existing 可能只有外框，间距很大不可信）
        c_only = [p for p, t in merged if t == 'C']
        if len(c_only) >= 3:
            gaps_c = [c_only[i+1] - c_only[i] for i in range(len(c_only)-1)]
            median_gap = float(np.median(gaps_c))
        else:
            all_pos = [p for p, _ in merged]
            gaps_all = [all_pos[i+1] - all_pos[i] for i in range(len(all_pos)-1)]
            median_gap = float(np.median(gaps_all)) if gaps_all else 0

        if median_gap <= 0:
            return [p for p, t in merged if t == 'C']

        # Step 2a: 删除过近打架的候选线
        # 判据 1：绝对过近（gap < median*0.4）→ 删除候选（保留 existing）
        # 判据 2：成对识别（gap < median*0.65 且两侧都是 C）→ 阶梯表底部行拆分过度
        #        的典型场景：836,843,850,864,... 间距7,7,14,7,7...，删中间那条
        small_gap = median_gap * 0.4
        pair_gap = median_gap * 0.65
        i = 0
        while i < len(merged) - 1:
            p1, t1 = merged[i]
            p2, t2 = merged[i+1]
            gap = p2 - p1
            # 判据 1：绝对过近
            if gap < small_gap:
                if t1 == 'E' and t2 == 'C':
                    del merged[i+1]
                    continue
                elif t1 == 'C' and t2 == 'E':
                    del merged[i]
                    continue
                elif t1 == 'C' and t2 == 'C':
                    merged[i] = (_choose_safer_line([p1, p2, (p1 + p2) // 2]), 'C')
                    del merged[i+1]
                    continue
            # 判据 2：成对识别
            # 若当前 C-C 对 gap < pair_gap，且下一段 gap 也 < pair_gap，
            # 说明本区间被过度拆分，删除中间那条 C
            if t1 == 'C' and t2 == 'C' and gap < pair_gap:
                if i + 2 < len(merged):
                    p3, t3 = merged[i+2]
                    gap2 = p3 - p2
                    if gap2 < pair_gap and t3 == 'C':
                        # 三条连续 C 且两段间距都小 → 删中间的 p2
                        del merged[i+1]
                        continue
                    # p2-p3 已经是正常间距，但 p1-p2 太小 → 合并 p1,p2
                    if gap2 >= pair_gap:
                        merged[i] = (_choose_safer_line([p1, p2, (p1 + p2) // 2]), 'C')
                        del merged[i+1]
                        continue
            i += 1

        # Step 2a-2: 删除“一宽一窄”误分割线
        # 如果相邻两段间距一大一小（大>median*1.1 且 小<median*0.85），
        # 且两段之和≈ 2*median（±40%），删“内容更多”的那条线。
        # 若内容相当，默认删中间(p2)，让迭代补线在正确位置补回。
        i = 0
        while i < len(merged) - 2:
            p1, t1 = merged[i]
            p2, t2 = merged[i+1]
            p3, t3 = merged[i+2]
            gap_a = p2 - p1
            gap_b = p3 - p2
            total = gap_a + gap_b
            if (abs(total - 2 * median_gap) < median_gap * 0.4
                    and ((gap_a > median_gap * 1.1 and gap_b < median_gap * 0.85)
                         or (gap_b > median_gap * 1.1 and gap_a < median_gap * 0.85))):
                # 用原图内容量决定删哪个：保留更空白的（正确边界）
                if not is_h:
                    s2 = int((gray[grid_a:grid_b, max(0,p2-1):min(gray.shape[1],p2+2)] < 200).sum())
                    s3 = int((gray[grid_a:grid_b, max(0,p3-1):min(gray.shape[1],p3+2)] < 200).sum())
                else:
                    s2 = int((gray[max(0,p2-1):min(gray.shape[0],p2+2), grid_a:grid_b] < 200).sum())
                    s3 = int((gray[max(0,p3-1):min(gray.shape[0],p3+2), grid_a:grid_b] < 200).sum())
                if s2 * 2 < s3 and t3 == 'C':
                    # p2明显更空白，保留p2，删p3
                    del merged[i+2]
                    continue
                elif t2 == 'C':
                    # 其他情况删p2（让迭代补线在正确位置补）
                    del merged[i+1]
                    continue
            i += 1

        # Step 2b: 补充过大间距（间距 > median*1.8）
        # 用 white_threshold 投影的 content 最小值定位补线位置，并要求补线点是低内容空白带
        large_gap = median_gap * 1.8
        added = []
        for i in range(len(merged) - 1):
            p1, _ = merged[i]
            p2, _ = merged[i+1]
            gap = p2 - p1
            if gap > large_gap:
                n_sub = int(round(gap / median_gap))
                if n_sub < 2:
                    continue
                # 在 (p1, p2) 内按期望位置找 content 最小值
                if is_h:
                    proj = (gray[p1:p2, grid_a:grid_b] < self.white_threshold).sum(axis=1).astype(np.float32)
                else:
                    proj = (gray[grid_a:grid_b, p1:p2] < self.white_threshold).sum(axis=0).astype(np.float32)
                expected = gap / n_sub
                search_r = max(3, int(expected / 3))
                last_pos = 0
                for k in range(1, n_sub):
                    ideal = int(round(k * expected))
                    lo = max(last_pos + 5, ideal - search_r)
                    hi = min(gap - 5, ideal + search_r)
                    if lo >= hi:
                        continue
                    seg = proj[lo:hi]
                    min_val = seg.min()
                    # 找连续最小值区域的中心（而非 argmin 起始位置）
                    min_indices = np.where(seg <= min_val + 1)[0]
                    if len(min_indices) > 0:
                        min_pos = lo + int((min_indices[0] + min_indices[-1]) // 2)
                    else:
                        min_pos = lo + int(np.argmin(seg))
                    new_pos = p1 + min_pos
                    max_content = max(5, int(abs(grid_b - grid_a) * 0.03))
                    if _line_content_score(new_pos) > max_content:
                        continue
                    added.append(new_pos)
                    last_pos = min_pos

        # 合并新增的 candidate
        for p in added:
            merged.append((p, 'C'))
        merged.sort(key=lambda x: x[0])

        # 返回：只画 candidate 类型的线（existing 不重画）
        return [p for p, t in merged if t == 'C']

    def _validate_bounds_by_spacing(self, gray: np.ndarray,
                                     bounds: List[Tuple[int, int]],
                                     total_len: int,
                                     is_col: bool) -> List[Tuple[int, int]]:
        """
        半边框间距校验：基于线间距分布，拆分过宽 bound（漏画线）、合并过窄 bound（多画线）。

        原理：正常 bound 宽度应接近中位数。
        - 过宽（> 中位数*1.8）→ 漏画线，拆分
        - 过窄（< 中位数*0.35）→ 多画线（碎片），合并到相邻 bound

        Args:
            gray: 灰度图
            bounds: 行/列边界
            total_len: 总长度（行=h，列=w）
            is_col: True=列，False=行

        Returns:
            校验后的 bounds
        """
        if len(bounds) < 3:
            return bounds
        widths = [e - s for s, e in bounds]
        median_w = float(np.median(widths))
        if median_w <= 0:
            return bounds

        # === Step 1: 拆分过宽 bound（漏画线） ===
        step1 = []
        n_split = 0
        for s, e in bounds:
            if (e - s) > median_w * 1.8:
                sub = self._split_wide_bound(gray, s, e, median_w, is_col)
                if len(sub) > 1:
                    n_split += 1
                step1.extend(sub)
            else:
                step1.append((s, e))

        # === Step 2: 合并过窄 bound（多画线/碎片） ===
        # 判据：宽度 < 中位数 * 0.35（既排除首列窄格如"保单年度末"5px，
        # 也合并拆分碎片）。首/尾 bound 保留（可能是真实的窄边框）。
        narrow_cap = median_w * 0.35
        merged = []
        n_merged = 0
        i = 0
        while i < len(step1):
            s, e = step1[i]
            w = e - s
            # 保留首尾 bound
            is_edge = (i == 0 or i == len(step1) - 1)
            if w < narrow_cap and not is_edge and merged:
                # 合并到上一个 bound
                prev_s, prev_e = merged[-1]
                merged[-1] = (prev_s, e)
                n_merged += 1
            else:
                merged.append((s, e))
            i += 1

        if n_split > 0 or n_merged > 0:
            kind = '列' if is_col else '行'
            logger.info(f"    半边框间距校验({kind}): 拆分 {n_split} 个过宽 + 合并 {n_merged} 个过窄 "
                         f"({len(bounds)} → {len(merged)})")
        return merged

    def _split_wide_bound(self, gray: np.ndarray, s: int, e: int,
                           median_w: float, is_col: bool) -> List[Tuple[int, int]]:
        """
        在过宽 bound 内部按"期望宽度 + 局部最小值"拆分。

        Args:
            gray: 灰度图
            s, e: 过宽 bound 的起止
            median_w: 正常 bound 宽度中位数
            is_col: True=列，False=行

        Returns:
            拆分后的子 bound 列表（无法拆分则返回 [(s,e)]）
        """
        width = e - s
        n_sub = int(round(width / median_w))
        if n_sub <= 1:
            return [(s, e)]

        content_mask = gray < self.white_threshold
        # 计算该 bound 区域内的 content 投影
        if is_col:
            # 列：水平投影（用内容高的行过滤，与主检测一致）
            row_content = content_mask.sum(axis=1)
            row_median = float(np.median(row_content))
            if row_median > 0:
                data_row_mask = row_content > row_median
                if data_row_mask.sum() >= 5:
                    proj = content_mask[data_row_mask, s:e].sum(axis=0).astype(np.float32)
                else:
                    proj = content_mask[:, s:e].sum(axis=0).astype(np.float32)
            else:
                proj = content_mask[:, s:e].sum(axis=0).astype(np.float32)
        else:
            # 行：垂直投影
            proj = content_mask[s:e, :].sum(axis=1).astype(np.float32)

        if proj.size == 0:
            return [(s, e)]

        # 期望宽度
        expected_w = width / n_sub
        search_r = max(3, int(expected_w / 3))

        # 在期望位置附近找 content 最小值作为边界
        cut_points = [0]
        for k in range(1, n_sub):
            ideal = int(round(k * expected_w))
            lo = max(cut_points[-1] + 5, ideal - search_r)
            hi = min(width - 5, ideal + search_r)
            if lo >= hi:
                continue
            # 找 [lo,hi] 内 content 最小值位置
            seg = proj[lo:hi]
            min_pos = lo + int(np.argmin(seg))
            cut_points.append(min_pos)
        cut_points.append(width)
        cut_points = sorted(set(cut_points))

        result = []
        for i in range(len(cut_points) - 1):
            cs = s + cut_points[i]
            ce = s + cut_points[i + 1]
            if ce - cs >= 5:
                result.append((cs, ce))
        return result if len(result) > 1 else [(s, e)]


    def _find_first_row(self, gray: np.ndarray,
                         row_bounds: List[Tuple[int, int]],
                         col_bounds: List[Tuple[int, int]], w: int) -> Optional[int]:
        """
        找表格真正的第一行：黑色像素跨列数占比 >= 60% 的第一行。

        对每一行，检查它在多少个"列"里有内容（用列边界）。
        如果没有列结构（无边框），改用把宽度分成 10 段的方式判断。
        """
        content_mask = gray < self.white_threshold

        if len(col_bounds) >= 3:
            # 有列结构：用列边界判断
            n_cols = len(col_bounds)
            for i, (y0, y1) in enumerate(row_bounds):
                cells_with_content = 0
                for (cx0, cx1) in col_bounds:
                    cell = content_mask[y0:y1, cx0:cx1]
                    if cell.sum() > 3:
                        cells_with_content += 1
                if cells_with_content / n_cols >= self.header_coverage:
                    return i
        else:
            # 无列结构：把宽度分成 10 段判断
            n_segments = 10
            seg_width = w // n_segments
            for i, (y0, y1) in enumerate(row_bounds):
                segs_with_content = 0
                for s in range(n_segments):
                    x0 = s * seg_width
                    x1 = (s + 1) * seg_width if s < n_segments - 1 else w
                    seg = content_mask[y0:y1, x0:x1]
                    if seg.sum() > 3:
                        segs_with_content += 1
                if segs_with_content / n_segments >= self.header_coverage:
                    return i

        return None


    def _is_first_col(self, gray: np.ndarray,
                        row_bounds: List[Tuple[int, int]],
                        col_bounds: List[Tuple[int, int]],
                        col_idx: int) -> bool:
        """
        验证 col_idx 指向的列是否真的像 first_col（区别于数据列）。

        判据（任一满足即视为 first_col）：
        1. 该列 cell 内容 median < 数据列 median 的 0.7 倍
        2. 该列 max/median >= 3（有跨行合并）
        """
        if col_idx >= len(col_bounds) or len(row_bounds) < 3:
            return False
        if len(col_bounds) < col_idx + 4:
            return False

        content_mask = gray < self.white_threshold

        def col_cells(idx):
            x0, x1 = col_bounds[idx]
            cs = []
            for ry0, ry1 in row_bounds:
                cs.append(int(content_mask[ry0:ry1, x0:x1].sum()))
            return cs

        target_cells = col_cells(col_idx)
        if not target_cells:
            return False
        target_median = sorted(target_cells)[len(target_cells) // 2]
        target_max = max(target_cells)

        data_start = col_idx + 2
        data_end = min(col_idx + 6, len(col_bounds))
        if data_end <= data_start:
            return False
        data_medians = []
        for i in range(data_start, data_end):
            cs = col_cells(i)
            if cs:
                data_medians.append(sorted(cs)[len(cs) // 2])
        if not data_medians:
            return False
        data_ref = sorted(data_medians)[len(data_medians) // 2]
        if data_ref <= 0:
            return False

        if target_median < data_ref * 0.7:
            return True
        if target_median > 0 and target_max / target_median >= 3:
            return True
        return False

    def _detect_header_row_count(self, gray: np.ndarray,
                                   row_bounds: List[Tuple[int, int]],
                                   col_bounds: List[Tuple[int, int]],
                                   first_row_idx: int) -> int:
        """
        检测从 first_row_idx 开始有多少行是表头行（非数据行）。

        判据：比较前几行与后续行的**每 cell 平均内容像素数**。
        - 表头行：cell 内容少（标签 1-2 字符，如"保单年度末"或"1 2 3..."）
        - 数据行：cell 内容多（4-7 位小数，如"2714.26"）

        如果 row[i] 的 cell 内容中位数 < 数据行中位数的 0.7 倍 → 是表头行。

        Returns:
            表头行数（至少 1）
        """
        if len(col_bounds) < 3 or len(row_bounds) < first_row_idx + 4:
            return 1  # 列太少或行太少，无法判断

        content_mask = gray < self.white_threshold

        def row_cell_content_median(row_idx: int) -> float:
            """计算某行每个 cell 的内容像素数的中位数"""
            y0, y1 = row_bounds[row_idx]
            cell_contents = []
            for cx0, cx1 in col_bounds:
                cell = content_mask[y0:y1, cx0:cx1]
                cell_contents.append(int(cell.sum()))
            if not cell_contents:
                return 0
            return float(sorted(cell_contents)[len(cell_contents) // 2])

        def row_is_header_like(row_idx: int, data_ref: float) -> bool:
            """判断某行是否像表头（非数据行）"""
            y0, y1 = row_bounds[row_idx]
            cell_contents = []
            for cx0, cx1 in col_bounds:
                cell = content_mask[y0:y1, cx0:cx1]
                cell_contents.append(int(cell.sum()))
            if not cell_contents:
                return False
            median_c = float(sorted(cell_contents)[len(cell_contents) // 2])
            max_c = float(max(cell_contents))

            # 判据 1: cell 内容中位数 < 数据行的 0.7 倍（少内容）
            if data_ref > 0 and median_c < data_ref * 0.7:
                return True
            # 判据 2: max/median >= 3（有合并/大标签 cell，如"保单年度末\年龄"）
            # 数据行的 max/median 通常 < 2（各 cell 内容量均匀）
            if median_c > 0 and max_c / median_c >= 3:
                return True
            return False

        # 计算"数据行"参考值：取 row[first_row_idx+2 ... +5] 的中位数
        data_start = first_row_idx + 2
        data_end = min(first_row_idx + 6, len(row_bounds))
        if data_end <= data_start:
            return 1

        data_medians = [row_cell_content_median(i) for i in range(data_start, data_end)]
        data_ref = sorted(data_medians)[len(data_medians) // 2]

        if data_ref <= 0:
            return 1  # 数据行也没内容，无法判断

        # 从 first_row_idx 开始检查连续表头行
        n_header = 1  # 至少 1 行表头
        for i in range(first_row_idx + 1, min(first_row_idx + 4, len(row_bounds))):
            if row_is_header_like(i, data_ref):
                n_header += 1
            else:
                break

        if n_header > 1:
            logger.info(f"    多行表头检测: {n_header} 行表头 "
                        f"(data_ref={data_ref:.0f}px)")
        return n_header


    def _find_first_col(self, gray: np.ndarray,
                         row_bounds: List[Tuple[int, int]],
                         col_bounds: List[Tuple[int, int]], h: int) -> Optional[int]:
        """
        找表格真正的第一列：黑色像素跨行数占比 >= 60% 的第一列。
        """
        content_mask = gray < self.white_threshold

        if len(row_bounds) >= 3:
            # 有行结构：用行边界判断
            n_rows = len(row_bounds)
            for i, (x0, x1) in enumerate(col_bounds):
                rows_with_content = 0
                for (ry0, ry1) in row_bounds:
                    cell = content_mask[ry0:ry1, x0:x1]
                    if cell.sum() > 3:
                        rows_with_content += 1
                if rows_with_content / n_rows >= self.first_col_coverage:
                    return i
        else:
            n_segments = 10
            seg_h = h // n_segments
            for i, (x0, x1) in enumerate(col_bounds):
                segs_with_content = 0
                for s in range(n_segments):
                    y0 = s * seg_h
                    y1 = (s + 1) * seg_h if s < n_segments - 1 else h
                    seg = content_mask[y0:y1, x0:x1]
                    if seg.sum() > 3:
                        segs_with_content += 1
                if segs_with_content / n_segments >= self.first_col_coverage:
                    return i

        return None

    # ==================== Step 4: 切块实现 ====================


    # ==================== 辅助方法 ====================

    def _draw_bounds_grid_lines(self, img: Image.Image,
                                 row_bounds: List[Tuple[int, int]],
                                 col_bounds: List[Tuple[int, int]],
                                 first_row_idx: Optional[int] = None,
                                 first_col_idx: Optional[int] = None) -> Image.Image:
        """
        用已计算的 row_bounds/col_bounds 在表格图上画网格线（半边框专用）。

        半边框表格有外框边框，导致"完全空白"检测失效。
        此方法在相邻 bounds 的边界处画线，但做两项保护避免产生假空单元格：

        1. 网格区域限定：检测真实外边框（横线暗像素 >80% 宽 / 竖线 >80% 高），
           只在边框围成的区域内画线。表格上方的标题行（top_extra）、
           左侧边距（left_extra）不会被格成空单元格。
           找不到真实边框时回退到 first_row_idx/first_col_idx 对应的边界。
        2. 单元格内部内容检查：构建单元格内部掩码（排除每个 bound 边界 ±2px，
           即已有线/边框位置），只在"画线位置的单元格内部无内容"时才画。
           这样斜线表头（如 保单年度末/投保年龄 共用一个单元格）不会被
           横线切开，有内部竖线的表格也不会因竖线导致横线全被跳过。

        Args:
            img: 原始表格图片
            row_bounds: 行边界列表 [(y_start, y_end), ...]
            col_bounds: 列边界列表 [(x_start, x_end), ...]
            first_row_idx: 首个数据行索引（无真实边框时的区域回退）
            first_col_idx: 首个数据列索引（无真实边框时的区域回退）

        Returns:
            画线后的新图片
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        img_arr = np.array(img.copy())

        # === 1. 检测真实外边框，限定网格区域 ===
        row_dark = (gray < 200).sum(axis=1)
        col_dark = (gray < 200).sum(axis=0)

        real_top = next((y for y in range(h) if row_dark[y] > w * 0.8), None)
        real_bot = next((y for y in range(h - 1, -1, -1) if row_dark[y] > w * 0.8), None)
        real_left = next((x for x in range(w) if col_dark[x] > h * 0.8), None)
        real_right = next((x for x in range(w - 1, -1, -1) if col_dark[x] > h * 0.8), None)

        # top：first_row_idx 的行起点和真实上边框的较小值（更靠上）。
        # 避免真实上边框检测把数据行内部的横线误当上边框（如 185a2337_r00 top=178
        # 太靠下导致画线范围只有 12px）。
        if first_row_idx is not None and 0 < first_row_idx < len(row_bounds):
            first_row_top = row_bounds[first_row_idx][0]
            top = min(first_row_top, real_top) if real_top is not None else first_row_top
        else:
            # 用 row_bounds[0][0] 作为 top（避免 real_top 在表格中间导致头几行被排除）
            top = row_bounds[0][0] if row_bounds else (real_top if real_top is not None else 0)
        # bot：row_bounds[-1] 的下边界和真实下边框的较大值（更靠下）。
        if row_bounds:
            last_row_bot = row_bounds[-1][1]
            bot = max(last_row_bot, real_bot) if real_bot is not None else last_row_bot
        else:
            bot = real_bot if real_bot is not None else h - 1
        # 问题4：如果计算出的数据区(bot-top)太小（<表高的30%），说明 top 检测异常，回退到表格起始
        if (bot - top) < h * 0.3:
            top = row_bounds[0][0] if row_bounds else 0
            bot = max(bot, row_bounds[-1][1] if row_bounds else h - 1)
        # left：first_col_idx 的列起点和真实左边框的较小值
        if first_col_idx is not None and 0 < first_col_idx < len(col_bounds):
            first_col_left = col_bounds[first_col_idx][0]
            left = min(first_col_left, real_left) if real_left is not None else first_col_left
        else:
            left = real_left if real_left is not None else 0
        # right：col_bounds[-1] 的右边界和真实右边框的较大值
        if col_bounds:
            last_col_right = col_bounds[-1][1]
            right = max(last_col_right, real_right) if real_right is not None else last_col_right
        else:
            right = real_right if real_right is not None else w - 1

        # === 2. 斜线表头保护 ===
        # 斜线表头（如 保单年度末/投保年龄）横跨 first_col 和第 1 数据列。
        # first_col 右边界竖线若在列头区域（first_row_idx 的行 + 2 行）有内容，
        # 则该竖线从列头区域下边界开始画（列头区域不画，保持斜线表头完整）。
        diag_protect_x = None
        diag_y_start = None
        if (first_row_idx is not None and first_col_idx is not None
                and first_row_idx < len(row_bounds) and first_col_idx < len(col_bounds)):
            cand_x = col_bounds[first_col_idx][1]
            header_end_idx = min(first_row_idx + 2, len(row_bounds) - 1)
            hy0 = row_bounds[first_row_idx][0]
            hy1 = row_bounds[header_end_idx][1]
            strip = gray[hy0:hy1, max(0, cand_x - 2):min(w, cand_x + 3)]
            # 排除竖线本身的暗像素：只在该位置不是贯通竖线时才触发斜线保护
            # 检查 cand_x 是否是贯通竖线（如果是，那些暗像素是竖线而非斜线表头）
            is_vline = False
            v_check_len = bot - top
            if v_check_len > 0:
                vline_strip = gray[top:bot, max(0, cand_x-1):min(w, cand_x+2)]
                if vline_strip.size > 0 and (vline_strip < 200).sum(axis=0).max() > v_check_len * 0.5:
                    is_vline = True
            if not is_vline and strip.size > 0 and (strip < 200).sum() > strip.size * 0.1:
                diag_protect_x = cand_x
                diag_y_start = hy1

        # === 3. 收集线并做间距校验 ===
        # 核心思路（用户澄清）：把"原表已有的线"和"候选线（bounds 边界）"合并成
        # 一个完整的线序列，用序列的间距分布做校验：
        # - 间距 > 中位数*1.8 → 大间距，中间应补线
        # - 间距 < 中位数*0.4 → 小间距（打架），删除其中一条（保留原表已有线）
        # 最终画出：候选线中被保留的（原表已有线不重画）

        # 3.1 收集原表已有线（真实边框线：暗像素比例 > 80%）
        # 判据用 80% 而非 40%，因为数据列中间每行有数字也能达到 40%，
        # 只有真实的通栏/通列边框线才能达到 80%。
        def find_existing_lines(is_h: bool) -> List[int]:
            if is_h:
                lines = []
                seg_len = right - left
                if seg_len <= 0:
                    return []
                for y in range(max(0, top), min(h, bot + 1)):
                    strip = gray[y, left:right]
                    if (strip < 200).sum() > seg_len * 0.8:
                        lines.append(y)
            else:
                lines = []
                seg_len = bot - top
                if seg_len <= 0:
                    return []
                for x in range(max(0, left), min(w, right + 1)):
                    strip = gray[top:bot, x]
                    if (strip < 200).sum() > seg_len * 0.8:
                        lines.append(x)
            # 合并近距离（gap<3）的连续位置为一条
            if not lines:
                return []
            merged = [lines[0]]
            for p in lines[1:]:
                if p - merged[-1] > 3:
                    merged.append(p)
            return merged

        existing_h = find_existing_lines(is_h=True)
        existing_v = find_existing_lines(is_h=False)

        # 3.2 候选线（bounds 边界，在网格区内）
        # 横线：跳过表头区域（包含列头行）
        # 竖线：从表头下边界开始画（列头内不画）
        # 表头下边界 = first_row_idx 行的下边界（只排除表头行本身）
        # 斜线保护(diag_y_start)时才扩展到更多行
        header_y_end = 0
        if first_row_idx is not None and first_row_idx < len(row_bounds):
            header_y_end = row_bounds[first_row_idx][1]
        # 斜线保护区域覆盖更多行（如 4e32dba9 列头有 斜线行+数字行）
        if diag_y_start is not None and diag_y_start > header_y_end:
            header_y_end = diag_y_start
        # 问题4：如果 header_y_end 占了表格高度的80%+，说明识别不合理，重置
        if (bot - top) > 0 and header_y_end > top + (bot - top) * 0.8:
            header_y_end = 0

        # 只排除斜线表头区域内的横线
        # 有斜线且 diag_protect_x 存在：跳过斜线覆盖的行
        # 无斜线：不排除任何横线
        if diag_protect_x is not None and first_row_idx is not None:
            skip_idx = min(first_row_idx + 1, len(row_bounds) - 1)
            h_threshold = row_bounds[skip_idx][1]
        else:
            h_threshold = 0
        candidate_h = [row_bounds[i][1] for i in range(len(row_bounds) - 1)
                       if top < row_bounds[i][1] < bot
                       and row_bounds[i][1] >= h_threshold]
        # 列头右边界：半边框表格列头到第一条原表竖线为止
        # 用 existing_v 中第一条非外框的竖线位置作为列头右边界
        first_col_x_end = 0
        edge_tol = max(20, int(w * 0.02))  # 外框容差
        inner_v = [x for x in existing_v if x > left + edge_tol and x < right - edge_tol]
        # 合并双线（如548,552间距4px的双像素线）为单条
        if inner_v:
            merged_inner = [inner_v[0]]
            for x in inner_v[1:]:
                if x - merged_inner[-1] < 10:
                    merged_inner[-1] = (merged_inner[-1] + x) // 2  # 取中点
                else:
                    merged_inner.append(x)
            inner_v = merged_inner
        if inner_v:
            # 第一条内部竖线 = 列头右边界
            first_col_x_end = inner_v[0]
        elif first_col_idx is not None and first_col_idx > 0:
            # 没有内部竖线且 first_col_idx>0，用 col_bounds 的列头边界
            first_col_x_end = col_bounds[first_col_idx][1]
        # else: first_col_x_end=0，不排除任何列

        # 问题3：如果原表已有≥3条内部竖线且间距规律，不画额外竖线
        skip_extra_v = False
        if len(inner_v) >= 3:
            inner_gaps = [inner_v[i+1] - inner_v[i] for i in range(len(inner_v)-1)]
            if max(inner_gaps) < min(inner_gaps) * 2:  # 间距比较均匀
                skip_extra_v = True

        candidate_v = [col_bounds[j][1] for j in range(len(col_bounds) - 1)
                       if left < col_bounds[j][1] < right
                       and col_bounds[j][1] > first_col_x_end]  # 排除列头区域
        if skip_extra_v:
            candidate_v = []  # 原表竖线已足够，不画额外竖线
        
        # 3.3 间距校验：只对竖线做，横线直接用 candidate_h（row_bounds 边界已确认正确）
        # 横线去重：如果候选横线与已有横线在 dedup_tol 内，跳过（避免重复画线）
        # 修复：半边框表格的 row_bounds 边界（内容检测）与源图现有横线（边框）会偏移，
        # 实测偏移可达 7-9px（如 8a150db7/table_04：候选 y=15 vs 源线 y=8 偏 7px，
        # 候选 y=175 vs 源线 y=166 偏 9px）。且 find_existing_lines 把粗线合并到首位置，
        # 粗线厚 3-4px 会再增 3-4px 偏移（如 58b3cb9e/table_04：候选 104 vs 合并线 90 偏 14px）。
        # 故 dedup_tol 取 15（基础偏移 9 + 粗线厚度 4 + 余量 2）。
        dedup_tol = 15
        # 空白带吸附容差：子表头下方常有较宽空白带，使空白间隙行边界远离源图真实横线
        # （如 7cd180ab/table_04：候选 y=141 vs 源线 y=161 偏 20px > dedup_tol），
        # 此时候选线与源线之间是纯空白（同一逻辑边界），需跳过候选线避免画双线。
        # 判据：候选线距源线 ≤ snap_blank_tol 且中间带暗像素 < 2% → 视为同一边界。
        # （环境变量 SNAP_BLANK_TOL 可覆盖，设为 0 可关闭该修复用于回归对比）
        snap_blank_tol = int(os.environ.get('SNAP_BLANK_TOL', '35'))

        # 表头块保护：如果源图前两条横线之间跨度显著大于普通行高，
        # 视为多行表头块，跳过表头块内的所有候选横线
        # 修复 b1044f0e/table_02: 顶边y=7, 表头下y=52, 跨度45px, 而普通行高约20px
        # 上限约束：表头跨度不能超过 5 倍参考行高 或表格高度的 15%（避免把整个数据区当表头）
        header_block_bounds = None
        table_h = bot - top
        # 厚边框/双线会被 find_existing_lines（gap>3 合并）拆成多条邻近线
        # （如顶边 y=0 与 y=4、表头下双线 162/166/173/177），使 existing_h[1]-existing_h[0]
        # 退化为"边框厚度"(4px)而非表头高度 → 表头块判据失效、表头被中缝候选线切开
        # （e62e178c/table_01：y=88 切穿 2 行 banner 表头）。故先按边框厚度容差聚类，
        # 再用聚类中心的前两条判定表头块（薄单线相距远，聚类不会误并，无回归）。
        def _cluster_border_lines(lines, tol=8):
            if not lines:
                return []
            clusters = [[lines[0]]]
            for p in lines[1:]:
                if p - clusters[-1][-1] <= tol:
                    clusters[-1].append(p)
                else:
                    clusters.append([p])
            return [int(sum(c) / len(c)) for c in clusters]
        clustered_h = _cluster_border_lines(existing_h, tol=8)
        if len(clustered_h) >= 2:
            top_gap = clustered_h[1] - clustered_h[0]
            # 参考行高：用 row_bounds 高度中位数（直接反映真实行高）。
            # 修复 4e32dba9/table_07：不能用 existing_h 相邻间距——当现有横线很少时
            # （如仅 [顶边, 表头下, 底边] 三条），间距是横跨整个数据区的巨大值（980px），
            # 导致 ref_gap 过大、表头块判据 top_gap > ref_gap*1.8 永远不成立。
            rb_heights = [e - s for s, e in row_bounds]
            ref_gap = float(np.median(rb_heights)) if rb_heights else 0
            # 表头块判据：跨度显著 > 行高（约2行以上），但不能超出合理范围。
            # 去掉 top_gap>30 绝对阈值：双行表头（如"投保年龄\保单年度末" 28px，
            # 行高14px）会被它卡住；相对判据 top_gap>ref_gap*1.8 已能区分单行/多行。
            max_header_span = min(ref_gap * 5, table_h * 0.15) if ref_gap > 0 else 0
            if (ref_gap > 0 and top_gap > ref_gap * 1.8
                    and top_gap <= max_header_span):
                header_block_bounds = (clustered_h[0], clustered_h[1])

        final_h = []
        for y in candidate_h:
            # 与源图已有横线距离过近 → 跳过（避免双线）
            if any(abs(y - ey) <= dedup_tol for ey in existing_h):
                continue
            # 落在表头块内 → 跳过（避免把多行表头切开）
            if header_block_bounds is not None:
                hb0, hb1 = header_block_bounds
                if hb0 + dedup_tol < y < hb1 - dedup_tol:
                    continue
            # 空白带吸附：候选线距源线较近且中间为纯空白 → 同一逻辑边界，跳过
            near_ey = next((ey for ey in existing_h
                            if dedup_tol < abs(y - ey) <= snap_blank_tol), None)
            if near_ey is not None:
                # 取候选线与源线之间的带（排除源线本身及其厚度）
                if y < near_ey:
                    band = gray[y:near_ey, left:right]
                else:
                    band = gray[near_ey + 4:y, left:right]
                if band.size > 0 and (band < 128).sum() < band.size * 0.02:
                    continue
            final_h.append(y)

        final_v = self._validate_lines_by_spacing(
            existing_v, candidate_v, gray, left, right, is_h=False,
            grid_a=top, grid_b=bot)
        if skip_extra_v:
            final_v = []  # 原表竖线已足够，强制不画额外竖线
        
        # 最终过滤：确保不在列头区域内画竖线（横线不过滤，全部画）
        final_v = [x for x in final_v if x >= first_col_x_end]

        # 3.4 画横线（排除表格末尾无数据区域的横线）
        n_h_drawn = 0
        drawn_h_positions = []
        for y_line in final_h:
            # 表格末尾检查：如果横线在表格底部5%且下方无有效内容，不画
            if y_line > bot - max(5, (bot - top) * 0.05):
                below_strip = gray[y_line:bot, first_col_x_end:right]
                if below_strip.size > 0 and (below_strip < 200).sum() < below_strip.size * 0.01:
                    continue
            img_arr[y_line, left:right] = 0
            n_h_drawn += 1
            drawn_h_positions.append(y_line)

        # 3.4b 横线迭代补线：检查相邻横线间距，过大时补充
        all_h = sorted(set([top] + existing_h + drawn_h_positions + [bot]))
        if len(all_h) >= 3:
            h_gaps = [all_h[i+1] - all_h[i] for i in range(len(all_h)-1)]
            # 排除微小间距（< 5px，通常是外框线与bot/top的误差）再计算 median
            real_h_gaps = [g for g in h_gaps if g >= 5]
            if real_h_gaps:
                sorted_h_gaps = sorted(real_h_gaps)
                # 排除最大间距（可能是待补线的大空白），用剩余的中位数作为正常行高
                if len(sorted_h_gaps) >= 2:
                    median_h_gap = float(np.median(sorted_h_gaps[:-1]))
                else:
                    median_h_gap = float(sorted_h_gaps[0])
            else:
                median_h_gap = 0
        else:
            median_h_gap = 0
        for _iter in range(5):
            if median_h_gap <= 0:
                break
            large_h_gap = median_h_gap * 1.8
            added_any = False
            new_all_h = sorted(all_h)
            for i in range(len(new_all_h) - 1):
                y1, y2 = new_all_h[i], new_all_h[i+1]
                gap = y2 - y1
                if gap > large_h_gap:
                    # 表头块保护：跳过完全落在表头块内的补线。用容差范围匹配
                    # （header_block_bounds 为聚类中心，未必等于 all_h 中的原始边框线，
                    #  故不能用 == 精确匹配，否则厚边框拆线时保护失效）。
                    if (header_block_bounds is not None
                            and y1 >= header_block_bounds[0] - dedup_tol
                            and y2 <= header_block_bounds[1] + dedup_tol):
                        continue
                    n_sub = int(round(gap / median_h_gap))
                    if n_sub < 2:
                        continue
                    expected = gap / n_sub
                    for k in range(1, n_sub):
                        ideal_y = int(y1 + k * expected)
                        # 在 ideal_y 附近找投影最低值位置
                        search_lo = max(y1 + 3, ideal_y - int(expected / 3))
                        search_hi = min(y2 - 3, ideal_y + int(expected / 3))
                        if search_lo >= search_hi:
                            continue
                        seg = (gray[search_lo:search_hi, first_col_x_end:right] < self.white_threshold).sum(axis=1)
                        min_val = seg.min()
                        min_indices = np.where(seg <= min_val + 1)[0]
                        if len(min_indices) > 0:
                            draw_y = search_lo + int((min_indices[0] + min_indices[-1]) // 2)
                        else:
                            draw_y = search_lo + int(np.argmin(seg))
                        if all(abs(draw_y - ey) > 5 for ey in all_h):
                            img_arr[draw_y, left:right] = 0
                            n_h_drawn += 1
                            all_h.append(draw_y)
                            added_any = True
            if not added_any:
                break
            all_h = sorted(set(all_h))
            h_gaps = [all_h[i+1] - all_h[i] for i in range(len(all_h)-1)]
            median_h_gap = float(np.median(h_gaps))

        # === 3.4 画竖线（用户思路：找“至少3px连续空白列”，忽略横线黑像素，迭代补线） ===
        # 构建横线行掩码：开结果横线(existing+drawn)位置±2px在检查竖线内容时不计入
        h_line_mask = np.ones(h, dtype=bool)  # True = 非横线行，参与检查
        all_h_lines = sorted(set(existing_h + final_h))
        for hy in all_h_lines:
            for dy in range(-2, 3):
                yy = hy + dy
                if 0 <= yy < h:
                    h_line_mask[yy] = False
        # 只取数据区 [header_y_end, bot) 且非横线的行
        data_rows = np.where(h_line_mask[header_y_end:bot])[0]  # 相对于 header_y_end
        n_data_rows = len(data_rows)

        # 自适应空白列阈值：统一用 2%+上限50
        col_widths = [e - s for s, e in col_bounds if (e - s) > 10]
        median_col_w = float(np.median(col_widths)) if col_widths else 50
        _blank_threshold = max(2, min(50, int(n_data_rows * 0.02)))
        # 自适应最低空白带宽度
        v_min_blank = max(3, int(median_col_w * 0.03))  # 48px→3, 280px→8

        def _is_blank_col(x: int) -> bool:
            """判断 x 列在数据区（排除横线行）是否为空白列。"""
            if x < 0 or x >= w or n_data_rows == 0:
                return False
            col = gray[header_y_end:bot, x]
            dark = int((col[data_rows] < self.white_threshold).sum())
            # 根据列宽自适应阈值：
            # 窄列(48px)列间数字延伸不可避，宽松(10%)
            # 宽列(280px)防千分位逗号，严格(2%且上靐50)
            return dark <= _blank_threshold

        def _find_blank_band(x_center: int, search_r: int = 8,
                            min_width: int = 3) -> Optional[int]:
            """在 x_center ± search_r 内找“至少 min_width px连续空白列”的中心位置。"""
            best_start, best_len = -1, 0
            cur_start, cur_len = -1, 0
            for x in range(max(left, x_center - search_r), min(right, x_center + search_r + 1)):
                if _is_blank_col(x):
                    if cur_len == 0:
                        cur_start = x
                    cur_len += 1
                    if cur_len > best_len:
                        best_start, best_len = cur_start, cur_len
                else:
                    cur_len = 0
            if best_len >= min_width:
                return best_start + best_len // 2
            # 放宽到 min_width-1（最低2px）
            fallback = max(2, min_width - 1)
            if best_len >= fallback:
                return best_start + best_len // 2
            return None

        n_v_drawn = 0
        drawn_v_positions = []  # 记录实际画线的 x 位置
        skip_v_safety = (bot - header_y_end) < 80  # 矮表跳过安全检查

        for x_line in final_v:
            y_start = top
            if skip_v_safety:
                img_arr[y_start:bot, x_line] = 0
                n_v_drawn += 1
                drawn_v_positions.append(x_line)
                continue
            # 画在候选位置，但先做轻量微调：如果候选位置有内容且±5px内有更好的空白位置，偏移过去
            draw_x = x_line
            col_content = int((gray[header_y_end:bot, x_line] < self.white_threshold).sum())
            if col_content > max(3, n_data_rows * 0.02):  # 候选位置有内容
                best_x, best_c = x_line, col_content
                for dx in range(-5, 6):
                    cx = x_line + dx
                    if cx < left or cx >= right:
                        continue
                    c = int((gray[header_y_end:bot, cx] < self.white_threshold).sum())
                    if c < best_c:
                        best_c = c
                        best_x = cx
                draw_x = best_x
            img_arr[y_start:bot, draw_x] = 0
            n_v_drawn += 1
            drawn_v_positions.append(draw_x)

        # === 3.5 迭代补线：检查相邻竖线间距，过大时补充 ===
        # 只用已画竖线 + 右边框构成序列，不含列头边界（避免列头到第一列的大间距被误补）
        all_v = sorted(set(drawn_v_positions + [right]))
        # 只保留列头右边界以右的线
        all_v = [x for x in all_v if x > first_col_x_end]
        if len(all_v) >= 3:
            gaps = [all_v[i+1] - all_v[i] for i in range(len(all_v)-1)]
            median_v_gap = float(np.median(gaps)) if gaps else 0
        else:
            median_v_gap = 0

        max_iter = 5
        for _iter in range(max_iter):
            if median_v_gap <= 0 or skip_v_safety:
                break
            large_v_gap = median_v_gap * 1.8
            added_any = False
            new_all_v = sorted(all_v)
            for i in range(len(new_all_v) - 1):
                x1, x2 = new_all_v[i], new_all_v[i+1]
                gap = x2 - x1
                if gap > large_v_gap:
                    # 在 (x1, x2) 中间补线
                    n_sub = int(round(gap / median_v_gap))
                    if n_sub < 2:
                        continue
                    expected = gap / n_sub
                    for k in range(1, n_sub):
                        ideal_x = int(x1 + k * expected)
                        # 先尝试用空白带搜索
                        draw_x = _find_blank_band(ideal_x, search_r=max(5, int(expected / 3)),
                                                    min_width=v_min_blank)
                        if draw_x is None:
                            # 找不到空白带时，用投影最低值的中心位置直接画
                            # （间距校验已确认该位置需要线）
                            search_lo = max(x1 + 5, ideal_x - int(expected / 3))
                            search_hi = min(x2 - 5, ideal_x + int(expected / 3))
                            if search_lo < search_hi:
                                seg = gray[header_y_end:bot, search_lo:search_hi]
                                col_content = (seg < self.white_threshold).sum(axis=0)
                                # 找连续最低值区域的中心
                                min_val = col_content.min()
                                min_idx = np.where(col_content <= min_val + 1)[0]
                                if len(min_idx) > 0:
                                    draw_x = search_lo + int((min_idx[0] + min_idx[-1]) // 2)
                        if draw_x is not None and x1 < draw_x < x2:
                            # 确保不在列头区域内，且不和已有线太近
                            if draw_x >= first_col_x_end and all(abs(draw_x - ex) > 5 for ex in all_v):
                                img_arr[top:bot, draw_x] = 0
                                n_v_drawn += 1
                                all_v.append(draw_x)
                                drawn_v_positions.append(draw_x)
                                added_any = True
            if not added_any:
                break
            all_v = sorted(set(all_v))
            # 重新计算 median
            gaps = [all_v[i+1] - all_v[i] for i in range(len(all_v)-1)]
            median_v_gap = float(np.median(gaps)) if gaps else 0

        result = Image.fromarray(img_arr)
        logger.info(f"    边界画线(semi_bordered): {n_h_drawn} 横线, {n_v_drawn} 竖线 "
                     f"(rows={len(row_bounds)}, cols={len(col_bounds)}, "
                     f"grid=[{left},{top}~{right},{bot}])")
        return result

    def _draw_borderless_grid_lines(self, img: Image.Image, col_bounds_hint=None):
        """
        对无边框表格基于空白像素行/列检测画网格线（来自 62fcdbe）。

        Args:
            col_bounds_hint: analyze_structure 检测的列边界（可选）。当空白带检测
                显著过检（画线数 > 列数×1.3）时，用它过滤掉不在列边界附近的杂线
                （如 81386925/table_01 空白带过检画了 61 条线，实际仅 35 列）。

        Returns:
            (drawn_img, h_lines, v_lines):
                drawn_img: 画线后的图片
                h_lines: 实际画的横线 y 位置列表
                v_lines: 实际画的竖线 x 位置列表
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        img_arr = np.array(img.copy())
        EDGE_MARGIN = 5

        # === 空白行检测（完全空白） ===
        row_content = (gray < 252).sum(axis=1)
        blank_rows = np.where(row_content <= 0)[0]

        # === 画横线 ===
        h_lines_drawn = []
        if len(blank_rows) > 0:
            bands = self._group_consecutive(blank_rows)
            MIN_H_SPACING = 5
            last_y = -MIN_H_SPACING
            for band_start, band_end in bands:
                if band_end - band_start >= 1:
                    y_line = (band_start + band_end) // 2
                    if EDGE_MARGIN <= y_line < h - EDGE_MARGIN and y_line - last_y >= MIN_H_SPACING:
                        img_arr[y_line, :] = 0
                        h_lines_drawn.append(y_line)
                        last_y = y_line

        # === 空白列检测（严格空白 + 有限边缘扩展） ===
        col_content = (gray < 252).sum(axis=0)
        # 第一步：严格检测完全空白列
        blank_cols_strict = np.where(col_content <= 0)[0]
        # 第二步：将每个空白带向两侧扩展最多 MAX_EXPAND 像素，纳入近似空白列
        # 修复 JPEG 压缩导致窄空白带被误判为非空白的问题
        col_blank_threshold = max(3, int(h * 0.01))
        MAX_EXPAND = 4  # 每侧最多扩展 4px，避免链式合并
        expanded = set(blank_cols_strict.tolist())
        if len(blank_cols_strict) > 0:
            bands_strict = self._group_consecutive(blank_cols_strict)
            for bs, be in bands_strict:
                # 向左扩展（最多 MAX_EXPAND 像素）
                for x in range(bs - 1, max(-1, bs - 1 - MAX_EXPAND), -1):
                    if col_content[x] <= col_blank_threshold:
                        expanded.add(x)
                    else:
                        break
                # 向右扩展（最多 MAX_EXPAND 像素）
                for x in range(be, min(w, be + MAX_EXPAND)):
                    if col_content[x] <= col_blank_threshold:
                        expanded.add(x)
                    else:
                        break
        blank_cols = np.array(sorted(expanded), dtype=np.intp) if expanded else np.array([], dtype=np.intp)

        # === 画竖线：自适应阈值 ===
        v_lines_drawn = []
        v_threshold = 4
        # 表头保护：计算表头行（首个内容行带）每列内容，
        # 用于识别"表头文字覆盖的区域"，避免竖线切开表头。
        # 修复 ab129e6b/c3c0bcd5：不能用 h_lines_drawn[0]——当表格顶部有空白边距时，
        # 第一条横线画在顶边（如 y=5），gray[:5] 是空白，表头词检测失效，
        # 导致"缴费期间""年度\年龄"等表头被字间距空白带画的竖线切开。
        # 改用首个内容行带（第一行文字）作为表头区。
        _content_rows = np.where(row_content > 0)[0]
        if len(_content_rows) > 0:
            _hdr_top = int(_content_rows[0])
            _blank_after = [y for y in range(_hdr_top, h) if row_content[y] <= 0]
            _hdr_bot = int(_blank_after[0]) if _blank_after else min(_hdr_top + 60, h)
            # 表头区高度限制在合理范围（避免把整个数据区当表头）
            _hdr_bot = min(_hdr_bot, _hdr_top + 60)
            header_col_content = (gray[_hdr_top:_hdr_bot, :] < 200).sum(axis=0)
        else:
            header_col_content = (gray[:min(30, h // 3), :] < 200).sum(axis=0)
        # 把表头内容段按字间距合并为"表头词"：字间距（≤MERGE_GAP px）属于词内部，
        # 列间隙（>MERGE_GAP px）保持分离。竖线落在表头词内部才跳过。
        HEADER_MERGE_GAP = 8
        header_segs = []
        _in = False
        _s = 0
        for _x in range(w):
            if header_col_content[_x] > 0:
                if not _in:
                    _s = _x
                    _in = True
            elif _in:
                header_segs.append((_s, _x))
                _in = False
        if _in:
            header_segs.append((_s, w))
        header_words = []
        for _s, _e in header_segs:
            if header_words and _s - header_words[-1][1] <= HEADER_MERGE_GAP:
                header_words[-1] = (header_words[-1][0], _e)
            else:
                header_words.append((_s, _e))
        if len(blank_cols) > 0:
            bands = self._group_consecutive(blank_cols)
            widths = sorted([e - s for s, e in bands])

            # 自适应阈值：双规则
            if len(widths) >= 2:
                from collections import Counter as _Counter
                width_counts = _Counter(widths)
                distinct_widths = sorted(width_counts.keys())
                total_bands = len(widths)
                for i in range(len(distinct_widths) - 1):
                    w_lo = distinct_widths[i]
                    w_hi = distinct_widths[i + 1]
                    gap = w_hi - w_lo
                    rule_a = gap > 2 and w_lo <= 6
                    rule_b = w_hi >= w_lo * 2 and gap >= 10
                    if rule_a or rule_b:
                        n_below = sum(width_counts[dw] for dw in distinct_widths[:i + 1])
                        if n_below > total_bands * 0.5:
                            v_threshold = w_lo + 1
            else:
                distinct_widths = widths[:1] if widths else [1]

            # 画竖线
            # 表头词宽度统计（用于 MIN_V_SPACING 和 WIDE_BAND_TH）
            _col_ws = [we - ws for ws, we in header_words] if header_words else []
            _median_col_w = float(np.median(_col_ws)) if _col_ws else 50.0

            # MIN_V_SPACING 自适应：防止同一列间隙被画两条线。
            # 如 3bfd625b/t08 的 105-106 列间隙产生两条线（x=4435,4446 仅隔 11px），
            # 因为间隙中一个 4px 伪空白带把空白区切成两段。
            # 最小间距 = 中位列间距×0.35（同一间隙内的双线间距远小于列间距）。
            if col_bounds_hint and len(col_bounds_hint) >= 3:
                _col_spacings = [e - s for s, e in col_bounds_hint]
                _median_spacing = float(np.median(_col_spacings))
                MIN_V_SPACING = max(8, int(_median_spacing * 0.35))
            else:
                MIN_V_SPACING = max(8, int(_median_col_w * 1.5)) if _col_ws else 8

            # 【过检过滤】当空白带检测显著过检（候选线数 > 列数×1.3）时，
            # 用 col_bounds_hint 的列边界过滤杂线（只保留贴近列边界的线）。
            # 仅在过检时启用，避免误伤画线数与列数匹配的正常表格。
            # 【漏检保护】若表头词数显著多于 col_bounds（如 81386925：61 个表头词
            # vs col_bounds 仅 35 列），说明 col_bounds 漏检，此时不能用它过滤，
            # 否则会误删 36-61 列的正确竖线。表头词多→列多，col_bounds 不可信。
            hint_boundaries = None
            hint_tol = 10
            if col_bounds_hint and len(col_bounds_hint) >= 2:
                col_bounds_reliable = len(header_words) <= len(col_bounds_hint) * 1.3
                candidate_count = sum(1 for bs, be in bands
                                      if (be - bs) >= v_threshold and bs > 0 and be < w)
                if col_bounds_reliable and candidate_count > len(col_bounds_hint) * 1.3:
                    hint_boundaries = sorted(set(
                        [col_bounds_hint[0][0]] + [c[1] for c in col_bounds_hint]))
                    # 容差随列宽自适应：列越宽容差越大（空白带中心与列边界的偏移）
                    col_widths = [e - s for s, e in col_bounds_hint]
                    hint_tol = max(6, int(float(np.median(col_widths)) * 0.12))

            # 宽空白带阈值：超过此宽度的带可能包含多列，需 subdiv 画多条线
            # （如 81386925 的 125px 宽带含 3 个表头词，只画中心线会漏掉两侧）
            WIDE_BAND_TH = _median_col_w * 1.8

            last_x = -MIN_V_SPACING
            _img_pre_vlines = img_arr.copy()  # 快照(含横线,无竖线)，供事后擦除多余竖线
            for band_start, band_end in bands:
                band_w = band_end - band_start
                if band_w >= v_threshold:
                    # 【边距空白带过滤】贴到图像左/右边缘的空白带是表格边距，
                    # 不是列分隔，跳过（如 3bfd625b/table_04 左侧 x=0-7 空白边距
                    # 被误画竖线，把"投保年龄"列切成碎片）。
                    # 判据：空白带外侧无内容（band 起点=0 或 终点=w）。
                    if band_start <= 0 or band_end >= w:
                        continue

                    # 【宽空白带 subdiv】带宽 > WIDE_BAND_TH 时，内部可能包含多个
                    # 表头词（多列共享一个大空白区）。只画中心线会因表头词保护
                    # 跳过或只画 1 条线。改为：找带内所有表头词，在词间隙画多条线。
                    # 同时处理带左缘→第一个词的间隙（修复 81386925 词34→35缺线：
                    # 词34在带外，词35在带内，原逻辑只画带内词间线，漏了带缘→词35）。
                    if band_w > WIDE_BAND_TH:
                        words_in_band = [(ws, we) for ws, we in header_words
                                         if ws < band_end and we > band_start]
                        if len(words_in_band) >= 2:
                            # 带左缘 → 第一个词（如果第一个词不紧贴带左缘）
                            first_ws = words_in_band[0][0]
                            if first_ws - band_start > 5:
                                x_line = (band_start + first_ws) // 2
                                if (hint_boundaries is None or any(
                                        abs(x_line - b) <= hint_tol for b in hint_boundaries)):
                                    if (EDGE_MARGIN <= x_line < w - EDGE_MARGIN
                                            and x_line - last_x >= MIN_V_SPACING):
                                        img_arr[:, x_line] = 0
                                        v_lines_drawn.append(x_line)
                                        last_x = x_line
                            # 词间间隙
                            for wi in range(len(words_in_band) - 1):
                                gap_s = words_in_band[wi][1]
                                gap_e = words_in_band[wi + 1][0]
                                x_line = (gap_s + gap_e) // 2
                                if hint_boundaries is not None and not any(
                                        abs(x_line - b) <= hint_tol for b in hint_boundaries):
                                    continue
                                if (EDGE_MARGIN <= x_line < w - EDGE_MARGIN
                                        and x_line - last_x >= MIN_V_SPACING):
                                    img_arr[:, x_line] = 0
                                    v_lines_drawn.append(x_line)
                                    last_x = x_line
                            continue  # 宽带已处理，跳过后续单线逻辑

                    x_line = (band_start + band_end) // 2
                    # 【过检过滤】线不在任何列边界附近 → 杂线，跳过。
                    if hint_boundaries is not None and not any(
                            abs(x_line - b) <= hint_tol for b in hint_boundaries):
                        continue
                    # 【表头保护】竖线落在表头词（合并字间距后的连续表头内容）内部 →
                    # 会切开表头文字 → 跳过。覆盖：
                    #   ①"投保年龄"字间距 ②"保单年度\投保年龄"的细斜线"\"
                    # 真实列间隙（如"（周岁）"与"0（须出生满30日）"之间的大间隙）
                    # 不在任何表头词内，正常画线。
                    if any(ws < x_line < we for ws, we in header_words):
                        continue
                    if EDGE_MARGIN <= x_line < w - EDGE_MARGIN and x_line - last_x >= MIN_V_SPACING:
                        img_arr[:, x_line] = 0
                        v_lines_drawn.append(x_line)
                        last_x = x_line

            # 【窄列多余竖线过滤】阶梯稀疏区易过检：真列线旁 ~1/3 列宽处多出一条杂线，
            # 落在数字内部把 6 位数切开（如 292b211d 右侧 x=9253/9345 偏离 col_bounds 32px）。
            # 规则：仅当 col_bounds 可靠时，删除"偏离 col_bounds 且与相邻线间距过小"的线
            # （真列线在 col_bounds 上、间距≈中位；被挤在真列线旁的杂线间距远小于中位）。
            if (len(v_lines_drawn) >= 3 and col_bounds_hint
                    and len(col_bounds_hint) >= 3
                    and len(header_words) <= len(col_bounds_hint) * 1.15):
                _cb_bounds = sorted(set(
                    [col_bounds_hint[0][0]]
                    + [c[1] for c in col_bounds_hint]
                    + [c[0] for c in col_bounds_hint]))
                _cw = [e - s for s, e in col_bounds_hint]
                _cb_tol = max(6, int(float(np.median(_cw)) * 0.15))
                _vl = sorted(v_lines_drawn)
                _gaps = [_vl[i + 1] - _vl[i] for i in range(len(_vl) - 1)]
                _med_gap = float(np.median(_gaps)) if _gaps else 0
                _to_remove = set()
                if _med_gap > 0:
                    for i in range(1, len(_vl) - 1):  # 不动首尾(表格边缘)
                        x = _vl[i]
                        off_cb = min(abs(x - t) for t in _cb_bounds) > _cb_tol
                        if not off_cb:
                            continue
                        left_gap = _vl[i] - _vl[i - 1]
                        right_gap = _vl[i + 1] - _vl[i]
                        # 偏离 col_bounds 且被挤在相邻线旁（最小间距 < 0.6 中位）→ 杂线
                        if min(left_gap, right_gap) < 0.6 * _med_gap:
                            _to_remove.add(x)
                if _to_remove:
                    for _x in _to_remove:
                        img_arr[:, _x] = _img_pre_vlines[:, _x]  # 擦除竖线(恢复横线)
                    v_lines_drawn = [x for x in v_lines_drawn if x not in _to_remove]
                    logger.info(f"    [窄列杂线过滤] 删除 {len(_to_remove)} 条偏离 col_bounds 的过窄竖线: {sorted(_to_remove)}")

            # 【切内容竖线过滤】移除"内容紧贴两侧(±5px)"的竖线——它切穿了数字/表头。
            # 稀疏阶梯区里，数字内部/表头内部的窄缝会被当成空白列画线（如 3b6cb198 把
            # "98113.2"切成"981"|"13.2"、34821e6c 把"第11保单年度"切成"第1"|"1保单年度"）。
            # 真列线落在列间隙，两侧 ±5px 内无内容(hug≈0)；杂线两侧紧贴数字(hug≥2)。
            # 不依赖 col_bounds（部分表 col_bounds 过检不可靠），按内容判定，覆盖首尾线。
            if v_lines_drawn:
                _CUT_W = 5
                _cut_remove = set()
                for _x in v_lines_drawn:
                    _xl = max(0, _x - _CUT_W)
                    _xr = min(w, _x + _CUT_W + 1)
                    _left = (gray[:, _xl:_x] < 180).any(axis=1)
                    _right = (gray[:, _x + 1:_xr] < 180).any(axis=1)
                    if int((_left & _right).sum()) >= 2:
                        _cut_remove.add(_x)
                if _cut_remove:
                    for _x in _cut_remove:
                        img_arr[:, _x] = _img_pre_vlines[:, _x]  # 擦除竖线(恢复横线)
                    v_lines_drawn = [x for x in v_lines_drawn if x not in _cut_remove]
                    logger.info(f"    [切内容竖线过滤] 删除 {len(_cut_remove)} 条切穿数字/表头的竖线: {sorted(_cut_remove)}")

        logger.info(f"    无边框画线(borderless): {len(h_lines_drawn)} 横线, {len(v_lines_drawn)} 竖线 "
                     f"(v_threshold={v_threshold}px)")
        return Image.fromarray(img_arr), h_lines_drawn, v_lines_drawn
    

    @staticmethod
    def _group_consecutive(positions: np.ndarray) -> List[Tuple[int, int]]:
        """将连续位置分组为 (start, end) 段。"""
        if len(positions) == 0:
            return []
        bands = []
        start = int(positions[0])
        prev = int(positions[0])
        for p in positions[1:]:
            p = int(p)
            if p - prev > 1:
                bands.append((start, prev + 1))
                start = p
            prev = p
        bands.append((start, prev + 1))
        return bands


    def _trim_margins(self, img: Image.Image) -> Optional[Image.Image]:
        """裁剪四周空白，返回裁剪后的图；如果全空返回 None。"""
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        content_mask = gray < self.white_threshold

        row_has = content_mask.sum(axis=1) > 3
        col_has = content_mask.sum(axis=0) > 3
        content_rows = np.where(row_has)[0]
        content_cols = np.where(col_has)[0]

        if len(content_rows) == 0 or len(content_cols) == 0:
            return None

        pad = 5
        y0 = max(0, int(content_rows[0]) - pad)
        y1 = min(h, int(content_rows[-1]) + 1 + pad)
        x0 = max(0, int(content_cols[0]) - pad)
        x1 = min(w, int(content_cols[-1]) + 1 + pad)
        return img.crop((x0, y0, x1, y1))


# ==================== API 返回质量判定 ====================


# ==================== 合并多个 chunk 的 API 返回 ====================

import re as _re


def _clean_api_response(resp: str) -> str:
    """去掉 markdown 代码块包裹"""
    if not resp:
        return ''
    s = resp.strip()
    s = _re.sub(r'^```(?:markdown|md|html)?\s*\n', '', s)
    s = _re.sub(r'\n```\s*$', '', s.strip())
    return s.strip()


def _extract_trs(html: str) -> List[str]:
    """从 HTML 中提取所有 <tr>...</tr>"""
    return _re.findall(r'<tr\b[^>]*>.*?</tr>', html, _re.DOTALL | _re.IGNORECASE)


def _extract_tds(tr_html: str) -> List[str]:
    """从一个 <tr>...</tr> 中提取所有 <td>...</td> 或 <th>...</th>"""
    return _re.findall(r'<t[dh]\b[^>]*>.*?</t[dh]>', tr_html, _re.DOTALL | _re.IGNORECASE)


def _drop_first_td(tr_html: str) -> str:
    """去掉 tr 内的第一个 <td>/<th>（用于合并左右列段时去除重复首列）"""
    return _re.sub(r'(<tr[^>]*>)\s*<t[dh]\b[^>]*>.*?</t[dh]>',
                    r'\1', tr_html, count=1, flags=_re.DOTALL | _re.IGNORECASE)


def _align_tds(tr_html: str, expected_n: int) -> str:
    """
    把一个 <tr> 内的 <td> 数量对齐到 expected_n：
    - 多余的裁掉
    - 不足的补空 <td></td>
    """
    if expected_n <= 0:
        return tr_html
    tds = _extract_tds(tr_html)
    if len(tds) == expected_n:
        return tr_html
    if len(tds) > expected_n:
        tds = tds[:expected_n]
    else:
        tds = tds + ['<td></td>'] * (expected_n - len(tds))
    # 保留原 <tr> 属性
    m = _re.match(r'(<tr[^>]*>)', tr_html, _re.IGNORECASE)
    tr_open = m.group(1) if m else '<tr>'
    return tr_open + ''.join(tds) + '</tr>'


def _merge_horizontal_segments(seg_trs_list: List[List[str]]) -> List[str]:
    """
    水平合并多个列段的 tr 列表。

    Args:
        seg_trs_list: 每个元素是一个列段的 <tr> 列表（内部 <td> 已按需去掉首列）

    Returns:
        合并后的完整 <tr> 列表
    """
    if not seg_trs_list:
        return []
    if len(seg_trs_list) == 1:
        return seg_trs_list[0]

    max_len = max(len(trs) for trs in seg_trs_list)
    result = []
    for i in range(max_len):
        merged_tds = []
        for trs in seg_trs_list:
            if i < len(trs):
                tds = _extract_tds(trs[i])
                merged_tds.extend(tds)
        if merged_tds:
            result.append('<tr>' + ''.join(merged_tds) + '</tr>')
    return result


def merge_chunk_results(chunks_with_responses: List[Tuple[Dict, str]]) -> str:
    """
    合并多个 chunk 的 API 返回为完整表格 markdown。

    Args:
        chunks_with_responses: list of (chunk_meta, api_response_text)
            chunk_meta 字段（见 TableChunker.chunk_table 返回）：
                - order, kind, has_header, has_first_col
                - row_range: (start, end)  1-based 数据行范围
                - col_range: (start, end)  1-based 数据列范围（-1 表示全列）

    Returns:
        str: 合并后的完整 markdown（包含 <table> 主体 + 可能的顶部说明文字）
    """
    if not chunks_with_responses:
        return ''

    # 分类
    top_extras = []
    bodies = []
    for meta, resp in chunks_with_responses:
        if meta.get('kind') == 'top_extra':
            top_extras.append((meta, resp))
        else:
            bodies.append((meta, resp))

    top_extras.sort(key=lambda x: x[0]['order'])
    bodies.sort(key=lambda x: x[0]['order'])

    # 组织 body chunks：按 row_range[0] 分组（同一行段）
    body_by_row_start: Dict[int, List[Tuple[Dict, str]]] = {}
    for meta, resp in bodies:
        row_start = meta['row_range'][0]
        body_by_row_start.setdefault(row_start, []).append((meta, resp))

    # 每个行段内按 col_range[0] 排序
    for row_start in body_by_row_start:
        body_by_row_start[row_start].sort(key=lambda x: x[0]['col_range'][0])

    # 按 row_start 顺序处理，收集所有合并后的 <tr>
    all_rows_html: List[str] = []
    row_starts_sorted = sorted(body_by_row_start.keys())
    first_row_start = row_starts_sorted[0] if row_starts_sorted else None

    for row_start in row_starts_sorted:
        segments = body_by_row_start[row_start]
        # 表头行数：以本行段第一个 chunk 的 n_header_rows 为准
        # 兼容旧数据：如果 chunk 只有 has_header 无 n_header_rows，视为 1
        first_meta = segments[0][0]
        n_header = first_meta.get('n_header_rows',
                                   1 if first_meta.get('has_header') else 0)

        # 从最左 chunk 的数据行推断"实际首列 td 数"
        # 结构检测的 n_first_cols 可能过检/欠检；API 返回的 td 数才是权威
        # 用数据行 td 数中位数减去数据列数，得到 API 实际使用的首列 td 数
        inferred_first_col_tds = 0
        if first_meta.get('has_first_col') and first_meta['col_range'][0] <= 1:
            first_data_cols = max(0, first_meta['col_range'][1] - first_meta['col_range'][0] + 1)
            first_trs = _extract_trs(_clean_api_response(segments[0][1]))
            # 找 td 数 >= data_cols 的行（跳过 header/幻觉行）
            data_tds_counts = sorted([
                len(_extract_tds(tr)) for tr in first_trs
                if len(_extract_tds(tr)) >= first_data_cols
            ])
            if data_tds_counts:
                median_td = data_tds_counts[len(data_tds_counts) // 2]
                inferred_first_col_tds = max(0, median_td - first_data_cols)
        # 兜底：至少 1 个（如果 has_first_col 且推断失败）
        if first_meta.get('has_first_col') and inferred_first_col_tds == 0:
            inferred_first_col_tds = 1

        # 从最左 chunk 推断"实际 API header 行数"
        # 结构 n_header_rows 可能过检；API 用 <th> 或 <thead> 标记 header
        # 数从顶部开始连续的、含 <th> 标签的 tr 数
        def _count_leading_th_trs(trs_list):
            n = 0
            for tr in trs_list:
                if _re.search(r'<th\b', tr, _re.IGNORECASE):
                    n += 1
                else:
                    break
            return n
        first_trs_for_header = _extract_trs(_clean_api_response(segments[0][1]))
        inferred_header_rows = _count_leading_th_trs(first_trs_for_header)

        # 每个列段提取 <tr>，并按预期 tr/td 数对齐（防止 API 幻觉/漏检）
        seg_trs_list = []
        # 记录每个 chunk 被 filter 掉的行数（视作 header 已被移除）
        n_filtered_out = 0
        for i_seg, (meta, resp) in enumerate(segments):
            resp_clean = _clean_api_response(resp)
            trs = _extract_trs(resp_clean)
            trs_before_filter = len(trs)

            # 预期该 chunk 的 td 数：数据列数 + inferred_first_col_tds（该 chunk 也含 first_col）
            c0, c1 = meta['col_range']
            expected_data_cols = max(0, c1 - c0 + 1) if c0 > 0 else 0
            # 每个 chunk 图片都包含 first_col（chunker 在切时都会拼），故都要 +inferred_first_col_tds
            expected_td = expected_data_cols + (inferred_first_col_tds if meta.get('has_first_col') else 0)

            # 过滤掉 td 数远少于预期的行（API 有时多返回跨列合并的幻觉行）
            # 但保护带 <th> 的行（真 header 用 <th>+colspan 表示多列，td 数可能少）
            # 只对纯 <td> 的短行做过滤（区分：hallucination=<td colspan>vs header=<th> colspan）
            if expected_td > 3:
                filtered = []
                for tr in trs:
                    n_tds = len(_extract_tds(tr))
                    has_th = bool(_re.search(r'<th\b', tr, _re.IGNORECASE))
                    if n_tds >= expected_td * 0.3 or has_th:
                        filtered.append(tr)
                trs = filtered
            n_filtered_out = max(n_filtered_out, trs_before_filter - len(trs))

            # 对齐每行 td 数
            if expected_td > 0:
                trs = [_align_tds(t, expected_td) for t in trs]

            # 非最左列段：去掉每个 tr 前 inferred_first_col_tds 个 td
            if meta.get('has_first_col') and meta['col_range'][0] > 1 and inferred_first_col_tds > 0:
                for _ in range(inferred_first_col_tds):
                    trs = [_drop_first_td(tr) for tr in trs]
            seg_trs_list.append(trs)

        # 水平合并同一行段的各列段
        # 统一 tr 数（如过滤后仍有差异，取 min 从顶部裁）
        if len(seg_trs_list) > 1:
            min_len = min(len(trs) for trs in seg_trs_list)
            seg_trs_list = [trs[len(trs) - min_len:] for trs in seg_trs_list]
        row_htmls = _merge_horizontal_segments(seg_trs_list)

        # 非第一行段：去掉整个 header 部分（含跨列合并的多行表头），避免重复
        # 优先用 inferred_header_rows（API <th> 标记）；若=0 说明 API 全用 <td>
        # 回退到 meta.n_header（chunker 强制拼进 chunk 图的 header 行数）
        # 减去 n_filtered_out：filter 已 drop 的不再补 drop
        if row_start != first_row_start:
            base_header = inferred_header_rows if inferred_header_rows > 0 else n_header
            remaining_header = max(0, base_header - n_filtered_out)
            if remaining_header > 0:
                if len(row_htmls) > remaining_header:
                    row_htmls = row_htmls[remaining_header:]
                else:
                    row_htmls = []   # 保底：如果不够，说明该 chunk 只有 header，跳过

        all_rows_html.extend(row_htmls)

    # 组装 <table>
    if all_rows_html:
        table_html = '<table border="1">\n' + '\n'.join(all_rows_html) + '\n</table>'
    else:
        table_html = ''

    # top_extra 作为前置文本
    prefix_parts = []
    seen_text = set()
    for meta, resp in top_extras:
        cleaned = _clean_api_response(resp)
        if cleaned and cleaned not in seen_text:
            seen_text.add(cleaned)
            prefix_parts.append(cleaned)

    # 【rule5】不能遗漏任何 chunk 的文字：收集各 body chunk 中"表格外"的文字
    #（VLM 偶尔把说明/标签放在 <table>/<tr> 之外），去重后放到合并 <table> 上方
    for meta, resp in bodies:
        outside = _clean_api_response(resp)
        outside = _re.sub(r'<table\b.*?</table>', '', outside, flags=_re.S | _re.I)
        outside = _re.sub(r'<tr\b.*?</tr>', '', outside, flags=_re.S | _re.I)
        outside = _re.sub(r'<[^>]+>', '', outside).strip()
        if (outside and outside not in seen_text
                and _re.search(r'[\w\u4e00-\u9fff]', outside)):
            seen_text.add(outside)
            prefix_parts.append(outside)

    if prefix_parts:
        return '\n\n'.join(prefix_parts) + '\n\n' + table_html
    return table_html
