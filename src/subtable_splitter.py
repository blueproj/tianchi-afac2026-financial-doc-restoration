"""
Level 1.5「子表切分」(sub-table split)

针对单个 table region 内部由多个纵向堆叠子表构成的场景（典型：阶梯型现金价值表，
男性块 + 女性块 / 不同缴费年期块）。在 Level 2 之前，把一个 region 沿 y 切成 N 个子表，
每个子表独立走 Level 2；同一 region 的多个子表结果最后合并回一个 <table>。

检测信号（逐行内容量突变）：
  1. 先取数据行边界（复用 TableChunker.analyze_structure：无边框按空白行、有边框按横线）。
  2. 逐行内容量：
       - black(r)      = 排除竖向网格线后的暗像素数
       - right_edge(r) = 行内最右暗像素 x（阶梯边缘几何定义）
  3. 子表分界 = 某行内容量落到局部极小(i)，下一行(i+1)骤增回接近满宽：
       right_edge(i+1) - right_edge(i) > right_jump_ratio × 表宽
       且 right_edge(i+1) > near_full_ratio × max_right
       且 black(i+1)      > black_jump_mult × black(i)

防误切：仅在检测到明确"阶梯 reset"时切分；否则原样返回单子表（普通表零回归）。
每个子表至少 min_rows 行，避免把小计行/残行误判为分界。
"""
import logging
import numpy as np
from typing import List, Dict, Optional
from PIL import Image

logger = logging.getLogger("subtable")

Image.MAX_IMAGE_PIXELS = None


class SubTableSplitter:
    """把一个 table region 沿 y 切成多个纵向堆叠子表。"""

    def __init__(self,
                 min_rows_per_subtable: int = 5,
                 right_jump_ratio: float = 0.4,
                 min_descent_run: int = 4,
                 near_full_ratio: float = 0.6,
                 dark_threshold: int = 200):
        """
        Args:
            min_rows_per_subtable: 每个子表最少数据行数（防止误切小计/残行）
            right_jump_ratio: right_edge 跳变阈值（占表宽比例）
            min_descent_run: 分界前必须有的连续阶梯下降行数（过滤周期性满宽行）
            near_full_ratio: i+1 行 right_edge 需逼近全局最大的比例
            dark_threshold: 灰度 < 此值算暗像素
        """
        self.min_rows = min_rows_per_subtable
        self.right_jump_ratio = right_jump_ratio
        self.min_descent_run = min_descent_run
        self.near_full_ratio = near_full_ratio
        self.dark_threshold = dark_threshold

    # ==================== 主入口 ====================

    def detect_row_cut_indices(self, gray: np.ndarray, row_bounds: List,
                               w: int, h: int,
                               col_bounds: Optional[List] = None,
                               table_type: str = '') -> List[int]:
        """核心检测：给定灰度图与行/列边界，返回子表切分索引。

        返回原始 row_bounds 中的索引 i（切点在 row_bounds[i] 与 [i+1] 之间）。
        空列表表示不切分。供 Level 2 (chunk_table) 在画线后直接调用。
        仅对 bordered 表启用单元格法（网格线会掩盖内容右边缘，且其 col_bounds 最可靠）；
        borderless/semi 无边框干扰、col_bounds 偏弱，仍用像素法。
        """
        if len(row_bounds) < self.min_rows * 2:
            return []
        use_cb = col_bounds if table_type == 'bordered' else None
        black, right_edge = self._row_profiles(gray, row_bounds, w, h, use_cb)
        # 过滤空行：结构分析可能把行间空白带误检为"行"（black≈0/right_edge=0），
        # 若不剔除，会把"空行→满宽行"误判为阶梯 reset（假阳性）。
        blank_floor = max(1, int(0.003 * w))
        content_idx = [i for i in range(len(row_bounds))
                       if right_edge[i] > 0 and black[i] > blank_floor]
        if len(content_idx) < self.min_rows * 2:
            return []
        c_black = black[content_idx]
        c_right = right_edge[content_idx]
        cut_local = self._find_boundaries(c_black, c_right, w)
        # 映射回原始 row_bounds 全局索引（阶梯底部行）
        return [content_idx[i] for i in cut_local]

    def split(self, table_img: Image.Image,
              structure: Optional[Dict] = None) -> List[Dict]:
        """
        Args:
            table_img: 一个 table region 图像
            structure: 可选，预先算好的 analyze_structure 结果（复用避免重算）

        Returns:
            list of dict: {'image', 'y_range': (y0,y1), 'order'}
            未检测到分界时返回单元素列表（整图）。
        """
        w, h = table_img.size
        gray = np.array(table_img.convert('L'), dtype=np.uint8)

        # 行/列边界（复用 Level 2 的结构分析，保证一致性）
        if structure is None:
            try:
                from src.table_chunker import TableChunker
                structure = TableChunker().analyze_structure(table_img)
            except Exception:
                structure = {}
        row_bounds = self._get_row_bounds(table_img, structure)
        col_bounds = structure.get('col_bounds') if structure else None
        table_type = structure.get('table_type', '') if structure else ''
        cuts = self.detect_row_cut_indices(gray, row_bounds, w, h, col_bounds, table_type)
        if not cuts:
            logger.info(f"    [子表切分] 单子表（{len(row_bounds)}行）")
            return [self._whole(table_img)]

        subs = self._crop_subtables(table_img, row_bounds, cuts)
        ranges = ", ".join(f"{s['y_range'][0]}-{s['y_range'][1]}" for s in subs)
        logger.info(f"    [子表切分] 检测到 {len(subs)} 个子表: y={ranges}")
        return subs

    # ==================== 行边界 ====================

    @staticmethod
    def _get_row_bounds(table_img: Image.Image,
                        structure: Optional[Dict]) -> List:
        """复用 TableChunker.analyze_structure 的 row_bounds。"""
        if structure is not None and structure.get('row_bounds'):
            return structure['row_bounds']
        try:
            from src.table_chunker import TableChunker
            st = TableChunker().analyze_structure(table_img)
            return st.get('row_bounds', []) or []
        except Exception as e:
            logger.warning(f"    [子表切分] analyze_structure 失败: {e}")
            return []

    # ==================== 逐行内容量 ====================

    def _row_profiles(self, gray: np.ndarray, row_bounds: List,
                      w: int, h: int, col_bounds: Optional[List] = None):
        """计算每个数据行的 black(暗像素数) 与 right_edge(内容右边缘 x)。

        right_edge 计算：
        - **优先单元格法**（有 col_bounds 时）：用网格列把每行切成单元格，从右往左
          找最右"内部有内容(数字/文字)"的单元格，取其右边界。此法用单元格内部
          (avoid 边框内缩 3px) 判定，彻底排除竖向网格线/外边框的干扰——这是有边框
          阶梯表检测的关键（否则网格线会让 right_edge 恒为满宽，掩盖阶梯）。
        - 回退像素法（无 col_bounds）：列暗像素 >= 15%行高 的最右列。
        """
        dark = gray < self.dark_threshold
        # 全高竖线列（网格线/边框）：某列 >50% 高度为暗
        col_line = dark.mean(axis=0) > 0.5
        use_cells = col_bounds is not None and len(col_bounds) >= 3

        n = len(row_bounds)
        black = np.zeros(n, dtype=np.int64)
        right_edge = np.zeros(n, dtype=np.int64)
        for idx, (y0, y1) in enumerate(row_bounds):
            if y1 <= y0:
                continue
            band = dark[y0:y1, :].copy()
            band[:, col_line] = False  # 排除竖线
            black[idx] = int(band.sum())

            if use_cells:
                # 单元格法：从右往左找最右"内部有内容"的单元格
                re = 0
                for ci in range(len(col_bounds) - 1, -1, -1):
                    x0, x1 = col_bounds[ci]
                    yy0, yy1 = y0 + 3, y1 - 3
                    xx0, xx1 = x0 + 3, x1 - 3
                    if yy1 <= yy0 or xx1 <= xx0:
                        continue
                    cell = dark[yy0:yy1, xx0:xx1]
                    if cell.size > 0 and cell.mean() > 0.02:
                        re = int(x1)
                        break
                right_edge[idx] = re
            else:
                band_h = y1 - y0
                col_dark = band.sum(axis=0)
                thr = max(2, int(0.15 * band_h))
                content_cols = np.where(col_dark >= thr)[0]
                right_edge[idx] = int(content_cols[-1]) if len(content_cols) else 0
        return black, right_edge

    # ==================== 分界检测 ====================

    def _find_boundaries(self, black: np.ndarray, right_edge: np.ndarray,
                         w: int) -> List[int]:
        """返回分界所在的行索引 i（切点在 row[i] 与 row[i+1] 之间）。"""
        n = len(black)
        max_right = int(right_edge.max()) if n else 0
        if max_right <= 0:
            return []

        cuts: List[int] = []
        last_cut = -1  # 上一个切点后子表起始为 last_cut+1
        tol = max(1, int(0.02 * w))  # 阶梯抖动容差
        near_full = self.near_full_ratio * max_right

        def descent_run(i: int) -> int:
            """行 i 结尾的连续"阶梯递减尾巴"长度：只数**低于满宽**且非增的行。

            关键：不数满宽平台行。否则"满宽平台 + 单个孤立短行"（如 88a0acf1
            大部分行满宽、个别行突然变短）会被误判为阶梯下降。真阶梯的递减尾巴
            有几十行连续低于满宽，孤立短行只有 1 行低于满宽。
            """
            if right_edge[i] >= near_full:
                return 0
            L = 1
            j = i
            while (j > 0 and right_edge[j - 1] < near_full
                   and int(right_edge[j - 1]) >= int(right_edge[j]) - tol):
                L += 1
                j -= 1
            return L

        for i in range(n - 1):
            re_jump = int(right_edge[i + 1]) - int(right_edge[i])
            # 阶梯 reset 的本质信号：右边缘骤增回到接近满宽。
            cond_right = re_jump > self.right_jump_ratio * w
            cond_full = right_edge[i + 1] > self.near_full_ratio * max_right
            # i 必须是阶梯底（明显短于满宽）
            cond_bottom = right_edge[i] < self.near_full_ratio * max_right
            # 分界前必须有足够长的连续阶梯下降：
            # 真阶梯表下降 run 长（几十行）；周期性满宽行（如每 3 行一个合计行）
            # 的 run 只有 2-3 行，据此过滤过切（实测 00c6e7df）。
            cond_run = descent_run(i) >= self.min_descent_run
            # 极弱 black 非减校验（防噪声）
            cond_black = black[i + 1] >= black[i]
            if cond_right and cond_full and cond_bottom and cond_run and cond_black:
                # 子表最小行数约束：本段 [last_cut+1 .. i] 和剩余 [i+1 .. n-1]
                seg_len = i - last_cut
                rest_len = n - (i + 1)
                if seg_len >= self.min_rows and rest_len >= self.min_rows:
                    cuts.append(i)
                    last_cut = i
        return cuts

    # ==================== 裁切 ====================

    @staticmethod
    def _crop_subtables(table_img: Image.Image, row_bounds: List,
                        cut_row_idxs: List[int]) -> List[Dict]:
        w, h = table_img.size
        # y 切点取 row[i] 底与 row[i+1] 顶之间的中点
        y_cuts = []
        for i in cut_row_idxs:
            y0 = row_bounds[i][1]
            y1 = row_bounds[i + 1][0]
            y_cuts.append((int(y0) + int(y1)) // 2)

        ys = [0] + y_cuts + [h]
        subs = []
        for k in range(len(ys) - 1):
            top, bot = ys[k], ys[k + 1]
            sub_img = table_img.crop((0, top, w, bot))
            subs.append({'image': sub_img, 'y_range': (top, bot), 'order': k})
        return subs

    @staticmethod
    def _whole(table_img: Image.Image) -> Dict:
        w, h = table_img.size
        return {'image': table_img, 'y_range': (0, h), 'order': 0}


# ==================== 子表结果合并 ====================
