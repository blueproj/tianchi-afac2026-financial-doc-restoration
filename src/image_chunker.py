"""
图像切块模块

使用直方图投影法找到空白行/列，在空白处切分，避免切断文字。
支持两种切块模式：
1. 长文档（面条图）：垂直方向切块
2. 表格文档：水平+垂直双向切块
"""
import os
import logging
import math
import numpy as np
from PIL import Image
import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.config import (
    MAX_CHUNK_WIDTH, MAX_CHUNK_HEIGHT,
    OVERLAP_RATIO, BLANK_THRESHOLD,
    TABLE_DIRECT_SUBMIT_WIDTH, TABLE_DIRECT_SUBMIT_HEIGHT,
    TABLE_STRIP_HEIGHT, TABLE_STRIP_OVERLAP, ENABLE_TABLE_STRIP,
    TABLE_MAX_ROWS_PER_STRIP, TABLE_LINE_PADDING,
    TABLE_ROW_GAP_DARK_RATIO, TABLE_ROW_GAP_MIN_HEIGHT, TABLE_BLANK_CHUNK_THRESHOLD,
    TABLE_VLINE_KERNEL_RATIO,
    API_MAX_OUTPUT_CHARS, CHARS_PER_CELL,
    ENABLE_CONTENT_CROP, CROP_PROJECTION_THRESHOLD, CROP_MIN_CONTENT_RATIO,
    CROP_PADDING, ENABLE_LONG_RESIZE, LONG_RESIZE_SCALE
)

logger = logging.getLogger("chunker")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)


class ImageChunker:
    """基于直方图投影的图像切块器"""

    def __init__(self,
                 max_chunk_width=MAX_CHUNK_WIDTH,
                 max_chunk_height=MAX_CHUNK_HEIGHT,
                 overlap_ratio=OVERLAP_RATIO,
                 blank_threshold=BLANK_THRESHOLD,
                 use_blank_band_cut=True,
                 blank_band_min_rows=8,
                 proj_downsample_width=256,
                 table_strip_height=TABLE_STRIP_HEIGHT,
                 table_strip_overlap=TABLE_STRIP_OVERLAP,
                 table_max_rows_per_strip=TABLE_MAX_ROWS_PER_STRIP,
                 table_line_padding=TABLE_LINE_PADDING,
                 # === 双栏 / 层次切分参数（P3） ===
                 enable_two_col_split=True,
                 two_col_mode='merge',  # 'merge'=整块保留送API | 'column'=左右切分
                 two_col_central_lo=0.35,
                 two_col_central_hi=0.65,
                 two_col_row_win_frac=0.02,     # 每行中央小窗宽度占全宽比例
                 two_col_row_dark_max=0.03,     # "双栏行"：中央小窗暗率上限
                 two_col_row_side_pix=8,        # "双栏行"：左右两侧最少暗像素数
                 two_col_min_seg_h=200,         # 双栏段最少高度（<则合并到单栏）
                 two_col_smooth_h=40,           # 段落形态学平滑高度
                 two_col_uniformity_max=0.85):  # 排除表格：行暗率均匀度上限
        """
        Args:
            max_chunk_width: 单块最大宽度
            max_chunk_height: 单块最大高度
            overlap_ratio: 重叠比例（0-1），仅在被迫硬切时作为安全重叠使用
            blank_threshold: 空白行判断阈值（0-255，大于此值认为该行是空白的）
            use_blank_band_cut: 是否启用"强空白带零重叠切分"（T8）。
                True时优先在整段空白处切分，切点不跨越文字，故无需重叠，
                从源头消除接缝重复/截断；False时退回原有等分+重叠策略。
            blank_band_min_rows: 判定"强空白带"所需的最少连续空白行数
            proj_downsample_width: 计算投影时的下采样宽度（抗OOM，C3）
            table_strip_height: 表格文档等分回退切条的目标高度（像素）
            table_strip_overlap: 表格文档等分回退切条的重叠像素
            table_max_rows_per_strip: 表格CV行边界切条时每条最多包含的行数
            table_line_padding: 表格CV行边界切条时保留的上下安全边距（像素）
            enable_two_col_split: 是否在每个粗切 chunk 内再做「上下分段 + 双栏
                左右切分」的层次切分（P3，20260719）。默认 True。
            two_col_*: 双栏检测/切分的阈值，见 _split_two_columns_layered。
        """
        self.max_chunk_width = max_chunk_width
        self.max_chunk_height = max_chunk_height
        self.overlap_ratio = overlap_ratio
        self.blank_threshold = blank_threshold
        self.use_blank_band_cut = use_blank_band_cut
        self.blank_band_min_rows = blank_band_min_rows
        self.proj_downsample_width = proj_downsample_width
        self.table_strip_height = table_strip_height
        self.table_strip_overlap = table_strip_overlap
        self.table_max_rows_per_strip = max(1, table_max_rows_per_strip)
        self.table_line_padding = max(0, table_line_padding)
        self.enable_two_col_split = enable_two_col_split
        self.two_col_mode = two_col_mode
        self.two_col_central_lo = two_col_central_lo
        self.two_col_central_hi = two_col_central_hi
        self.two_col_row_win_frac = two_col_row_win_frac
        self.two_col_row_dark_max = two_col_row_dark_max
        self.two_col_row_side_pix = two_col_row_side_pix
        self.two_col_min_seg_h = two_col_min_seg_h
        self.two_col_smooth_h = two_col_smooth_h
        self.two_col_uniformity_max = two_col_uniformity_max


    def _find_cut_points(self, projection, start, end, max_chunk_size, threshold=None):
        """
        在 [start, end) 范围内找到最佳切分点

        策略：
        1. 计算需要的切块数: ceil((end-start) / max_chunk_size)
        2. 在每个目标切分位置附近搜索最近的空白行
        3. 如果找不到空白行，就在目标位置直接切

        Returns:
            list: 切分点列表，包括start和end
        """
        length = end - start
        if length <= max_chunk_size:
            return [start, end]

        # 计算需要多少块
        n_chunks = int(np.ceil(length / max_chunk_size))
        # 每块的目标大小（不含重叠）
        target_size = length / n_chunks

        threshold = threshold or self.blank_threshold
        # 找出所有空白行
        blank_indices = set(np.where(projection >= threshold)[0].tolist())

        cut_points = [start]
        for i in range(1, n_chunks):
            target_pos = start + int(i * target_size)

            # 在目标位置附近搜索最近的空白行
            search_radius = min(int(target_size * 0.3), 200)
            best_pos = target_pos
            best_dist = float('inf')

            for offset in range(-search_radius, search_radius + 1):
                pos = target_pos + offset
                if start < pos < end and pos in blank_indices:
                    dist = abs(offset)
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = pos

            # 确保切分点不与前一个切分点太近
            if best_pos - cut_points[-1] < max_chunk_size * 0.3:
                best_pos = target_pos

            cut_points.append(best_pos)

        cut_points.append(end)
        return cut_points

    def _row_projection_safe(self, img):
        """
        计算每行像素均值（水平投影）——抗OOM版本（C3）

        理论：超大图整图转RGB numpy数组内存开销巨大（2亿像素RGB约600MB，
        再做mean还要翻倍），极易OOM。投影只需相对亮度信息，因此：
        1. 先转灰度（1字节/像素，内存降为1/3）；
        2. 把宽度下采样到固定值（BILINEAR近似横向平均），使内存与原图宽度解耦；
        3. 保留原始高度，从而切分点仍能精确对应到原图的行。

        Returns:
            np.ndarray: 长度=图像高度的每行均值(0-255)
        """
        gray = img.convert('L')
        w, h = gray.size
        target_w = min(w, self.proj_downsample_width)
        if w > target_w:
            gray = gray.resize((target_w, h), Image.BILINEAR)
        arr = np.asarray(gray, dtype=np.float32)
        return arr.mean(axis=1)

    def _crop_horizontal_margins(self, img, label="document"):
        """
        裁掉左右大面积白边，只保留有效内容和安全padding。

        Returns:
            (cropped_img, offset_x, crop_box)
        """
        if not ENABLE_CONTENT_CROP:
            return img, 0, (0, 0, img.size[0], img.size[1])

        gray = img.convert('L')
        width, height = gray.size
        sample_h = min(height, 4000)
        if height > sample_h:
            step = max(1, height // sample_h)
            sample = gray.resize((width, max(1, height // step)), Image.BILINEAR)
        else:
            sample = gray

        arr = np.asarray(sample, dtype=np.uint8)
        col_means = arr.mean(axis=0)
        dark_ratio = (arr < CROP_PROJECTION_THRESHOLD).mean(axis=0)
        content_cols = np.where((col_means < CROP_PROJECTION_THRESHOLD) | (dark_ratio >= CROP_MIN_CONTENT_RATIO))[0]
        if len(content_cols) == 0:
            return img, 0, (0, 0, width, height)

        x0 = max(0, int(content_cols[0]) - CROP_PADDING)
        x1 = min(width, int(content_cols[-1]) + CROP_PADDING + 1)
        min_keep = max(200, int(width * 0.3))
        if x1 - x0 < min_keep or (x0 == 0 and x1 == width):
            return img, 0, (0, 0, width, height)

        cropped = img.crop((x0, 0, x1, height))
        logger.info(f"  {label}左右白边裁剪: x={x0}:{x1}, 宽度 {width}->{cropped.size[0]}")
        return cropped, x0, (x0, 0, x1, height)


    @staticmethod
    def _has_content(img, min_dark_ratio=0.002):
        """判断图像是否有实际内容（暗像素占比 > 阈值），用于过滤空白 chunk。"""
        gray = np.asarray(img.convert('L'), dtype=np.uint8)
        dark_ratio = (gray < 200).mean()
        return bool(dark_ratio > min_dark_ratio)

    @staticmethod
    def _position(x0, y0, x1, y1, crop_offset_x=0, col_hint='full'):
        """生成统一位置字典，兼容后续debug和二维切分。

        col_hint: 'full' | 'left' | 'right' —— 该 sub-chunk 在双栏切分里的角色，
                  拼接排序时保证同一 y 段内 left 排在 right 前面。
        """
        return {
            'x_start': int(x0 + crop_offset_x),
            'y_start': int(y0),
            'x_end': int(x1 + crop_offset_x),
            'y_end': int(y1),
            'col_hint': col_hint,
        }

    # ==================== 双栏 / 层次切分（P3 v6，20260719） ====================
    #
    # 核心思路（v6）：
    #   1. 列级白带检测：在 chunk 中央区找最宽的连续"近白列"带（dark<200 占比<2%）
    #   2. 标题保护：白带区域内有暗像素的行（标题/横线穿过）不算双栏内容行
    #   3. 形态学 closing：合并水平分隔线造成的小间隙
    #   4. 连续双栏 chunk 按列优先排序：先全部左栏（上→下），再全部右栏（上→下）
    #   5. 表格排除：形态学水平线检测 + 行暗率均匀度

    def _find_column_band(self, gray):
        """
        列级白带检测：在中央区找最宽的连续"近白列"带。
        若找不到宽白带（≥30px），退化找中央暗率谷点作为窄分隔线。

        Returns:
            (band_x0, band_x1, band_center, band_width) or None
        """
        h, w = gray.shape
        cx_lo = int(w * 0.25)
        cx_hi = int(w * 0.75)
        col_dark = (gray < 200).mean(axis=0)
        is_white = col_dark[cx_lo:cx_hi] < 0.02

        best_start, best_len = -1, 0
        cur_start = -1
        for i in range(len(is_white)):
            if is_white[i]:
                if cur_start < 0:
                    cur_start = i
            else:
                if cur_start >= 0:
                    if i - cur_start > best_len:
                        best_len = i - cur_start
                        best_start = cur_start
                    cur_start = -1
        if cur_start >= 0:
            if len(is_white) - cur_start > best_len:
                best_len = len(is_white) - cur_start
                best_start = cur_start

        min_band_width = 30
        if best_len >= min_band_width:
            x0 = cx_lo + best_start
            x1 = x0 + best_len
            return x0, x1, (x0 + x1) // 2, best_len

        # 退化：找中央区暗率最低的谷点（窄分隔线/灰线）
        central_dark = col_dark[cx_lo:cx_hi]
        if len(central_dark) < 5:
            return None
        # 滑窗平滑（5px）找最低点
        kernel = np.ones(5) / 5
        smoothed = np.convolve(central_dark, kernel, mode='valid')
        min_idx = int(np.argmin(smoothed))
        min_val = float(smoothed[min_idx])
        mean_val = float(np.mean(central_dark))
        # 谷点暗率必须显著低于均值（< 50%）
        if min_val < mean_val * 0.5 and min_val < 0.05:
            center = cx_lo + min_idx + 2
            # 以谷点为中心，向两侧扩展到暗率回升的位置
            x0 = center
            while x0 > cx_lo and col_dark[x0 - 1] < 0.03:
                x0 -= 1
            x1 = center + 1
            while x1 < cx_hi and col_dark[x1] < 0.03:
                x1 += 1
            bw = x1 - x0
            return x0, x1, center, bw

        return None

    def _find_two_col_y_range(self, gray, band_x0, band_x1):
        """
        找双栏内容的 y 范围。

        双栏内容行 = 白带区域为空 AND 左右两侧都有内容。
        标题行（居中/单侧）和后续段落（横跨白带）自然被排除。

        用形态学 closing 合并水平分隔线造成的小间隙（<50px）。
        返回第一个到最后一个内容段的完整范围（不丢中间段）。

        Returns:
            (y0, y1) or None
        """
        h, w = gray.shape
        band_region = gray[:, band_x0:band_x1]
        band_width = band_x1 - band_x0
        band_dark_per_row = (band_region < 200).sum(axis=1)

        # 左右两侧暗像素数
        left_dark = (gray[:, :band_x0] < 200).sum(axis=1)
        right_dark = (gray[:, band_x1:] < 200).sum(axis=1)

        # 双栏内容行判定（核心思路：能分左右两栏的就是目录行）
        # 条件：(a) 左右两侧都有内容；(b) 白带中心 ±30px 区域为空
        #   - 目录行：左右内容之间是宽的列间隙（>=60px），白带±30px 区域为空
        #   - 段落行：文字连续穿过白带，词间隙窄（<60px），白带±30px 区域有文字
        #   - 标题行：文字居中连成一片，白带±30px 区域有文字
        min_side = 10
        mid_c = (band_x0 + band_x1) // 2
        band_region_x0 = max(0, mid_c - 30)
        band_region_x1 = min(w, mid_c + 30)
        # 白带±30px 区域的暗像素率（向量化）
        band_region_dark = (gray[:, band_region_x0:band_region_x1] < 200).mean(axis=1)

        # 白带区域内的最大白色间隙（列间隙）
        #   目录行：条目可能延伸进白带，但条目之间仍有宽的列间隙（>20px）
        #   段落行：文字连续穿过白带，无宽间隙（<20px）
        #   用"最大白色间隙"而非"白带完全为空"，容忍条目延伸进白带
        band_region_mask = gray[:, band_region_x0:band_region_x1] < 200  # True=暗
        band_w = band_region_mask.shape[1]
        largest_white = np.zeros(h, dtype=np.int32)
        for y in range(h):
            dark_pos = np.where(band_region_mask[y])[0]
            if len(dark_pos) == 0:
                largest_white[y] = band_w  # 全白
            else:
                gaps = [int(dark_pos[0]), band_w - 1 - int(dark_pos[-1])]
                if len(dark_pos) > 1:
                    gaps.extend((np.diff(dark_pos) - 1).tolist())
                largest_white[y] = max(gaps)
        # 白带区域有宽间隙（>20px）→ 列间隙 → 目录行
        band_has_gap = largest_white > 20

        # 双栏内容行：白带区域有宽列间隙 且 左右至少一侧有内容
        #   用 OR 而非 AND：目录左右栏条目数不同时，很多行只有单侧有内容
        #   排除纯空行（左右都无内容）和段落行（白带无宽间隙）
        has_side_content = (left_dark >= min_side) | (right_dark >= min_side)
        is_content = has_side_content & band_has_gap

        # 段落行（垂直窗口上下文纠正）：
        #   单行特征会误判两类行：
        #   - 段落标点间隙恰好在白带处 → 误判为双栏
        #   - 长目录条目延伸覆盖白带 → 误判为段落
        #   用邻域窗口（±60px）的双栏比例纠正孤立误判：
        #   目录区域双栏比例高，段落区域双栏比例低
        win = 60
        is_two_col = has_side_content & band_has_gap
        cs_two = np.concatenate([[0.0], np.cumsum(is_two_col.astype(np.float64))])
        cs_ne = np.concatenate([[0.0], np.cumsum(has_side_content.astype(np.float64))])
        idx = np.arange(h)
        lo = np.maximum(0, idx - win)
        hi = np.minimum(h, idx + win + 1)
        two_cnt = cs_two[hi] - cs_two[lo]
        ne_cnt = cs_ne[hi] - cs_ne[lo]
        two_col_frac = np.divide(two_cnt, ne_cnt, out=np.zeros_like(two_cnt), where=ne_cnt > 0)
        # 段落行：有内容 且 邻域双栏比例低（<0.5，处于段落区域）
        para_rows = has_side_content & (two_col_frac <= 0.5)

        # 形态学分段
        segs = []
        y = 0
        while y < h:
            v = bool(is_content[y])
            e = y
            while e < h and bool(is_content[e]) == v:
                e += 1
            segs.append((y, e, v))
            y = e

        # closing：合并 < 50px 的非内容间隙（水平分隔线、标题行等）
        close_gap = 50
        changed = True
        while changed:
            changed = False
            new_segs = []
            i = 0
            while i < len(segs):
                s, e, d = segs[i]
                if (d and i + 2 < len(segs) and
                        not segs[i + 1][2] and
                        segs[i + 1][1] - segs[i + 1][0] < close_gap and
                        segs[i + 2][2]):
                    new_segs.append((s, segs[i + 2][1], True))
                    i += 3
                    changed = True
                else:
                    new_segs.append((s, e, d))
                    i += 1
            segs = new_segs

        # 取第一个到最后一个内容段的完整范围（不丢中间段）
        content_segs = [(s, e) for s, e, d in segs if d]
        if not content_segs:
            return None

        # "左右都有内容"判定已自然排除标题行和后续段落，直接用第一个内容段起点
        first_start = content_segs[0][0]
        start_seg_idx = 0

        # 标题段跳过：如果第一个内容段很短（<200px）且与下一段之间有大间隙（>50px），
        # 并且该段有文字（标题），则判定为标题，跳到第二段（TOC 从标题后开始）
        if len(content_segs) >= 2:
            seg0_s, seg0_e = content_segs[0]
            seg0_h = seg0_e - seg0_s
            gap_to_next = content_segs[1][0] - seg0_e
            if seg0_h < 200 and gap_to_next > 50:
                # 检查第一段是否有文字（标题）。标题可能居中，不一定在白带区域，
                # 所以检查整段宽度而非白带区域
                seg_dark = (gray[seg0_s:seg0_e, :] < 200).mean()
                if seg_dark > 0.002:
                    # 有文字 → 标题段 → 跳过，TOC 从第二段开始
                    first_start = content_segs[1][0]
                    start_seg_idx = 1

        # TOC 区域：从起始内容段开始延伸
        # 只有当间隙中出现"穿过白带的段落文字"（有内容且白带无宽间隙）才停止
        # 目录行（白带有宽间隙）不算段落，延伸应穿过它们继续
        # （para_rows 已在前面定义）
        # 注意：纯目录内容段不含"成块的段落行"；目录内的横线（1-3px 细线）虽也是
        # 段落行，但很短。只有"成块的连续段落行"（>10px，即真正的段落）才停止延伸
        def _max_para_run(arr):
            """最长连续段落行长度（区分横线细线 vs 段落粗块）"""
            mx = cur = 0
            for v in arr:
                if v:
                    cur += 1
                    if cur > mx:
                        mx = cur
                else:
                    cur = 0
            return mx

        para_run_threshold = 20
        last_end = content_segs[start_seg_idx][1]
        for i in range(start_seg_idx + 1, len(content_segs)):
            gap_s = content_segs[i - 1][1]
            gap_e = content_segs[i][0]
            # 间隙中出现成块段落文字（穿过白带）→ 这是 TOC 之后的段落，停止延伸
            if _max_para_run(para_rows[gap_s:gap_e]) > para_run_threshold:
                break
            # 下一个内容段内部含成块段落行 → 该段是段落（被 closing 合并），停止延伸
            seg_s, seg_e = content_segs[i]
            if _max_para_run(para_rows[seg_s:seg_e]) > para_run_threshold:
                break
            # 间隙纯空且下一段是纯目录（或仅含横线细线）→ 属于 TOC 内部，继续延伸
            last_end = content_segs[i][1]

        min_seg = 100
        if last_end - first_start < min_seg:
            return None

        # 校验：真目录的白带区域整体是列间隙（绝大多数有内容行的白带为空）；
        # 而文本词间隙/竖线造成的伪双栏，白带区域大部分有文字。
        # 用"TOC 区域内有内容行中白带为空的比例"过滤伪双栏（如 TNM 分期文本）。
        region = slice(first_start, last_end)
        content_mask = has_side_content[region]
        n_content = int(content_mask.sum())
        if n_content > 0:
            band_empty_mask = (band_region_dark[region] < 0.05) & content_mask
            empty_frac = float(band_empty_mask.sum()) / n_content
            if empty_frac < 0.85:
                return None

        return first_start, last_end

    @staticmethod
    def _below_is_body(chunk_img, y_split) -> bool:
        """判断 chunk 中 y_split 以下区域是否为「正文段落」（应与目录分离）。

        正文段落是 justified 全宽多行文字：整行内容横向跨度接近满宽（span>0.8）的行占比高；
        而单栏目录/名词释义的续接条目是左对齐短行、右侧大量留白，几乎没有满宽行。

        返回 True=正文（应单独成块）；False=目录类续接（应并入目录，避免"不能少"）。
        阈值经 B 榜实测：正文段落 span>0.8 行占比≈0.23，目录尾（第七/八条、释义续条）≈0.00。
        """
        import numpy as np
        h = chunk_img.size[1]
        if h - y_split < 40:
            return False
        gray = np.asarray(chunk_img.convert('L'), dtype=np.uint8)
        region = gray[y_split:h]
        rh, w = region.shape
        dark = region < 200
        has = dark.any(axis=1)
        left = np.argmax(dark, axis=1)
        right = w - 1 - np.argmax(dark[:, ::-1], axis=1)
        span = np.where(has, (right - left) / float(w), 0.0)
        return float((span > 0.8).mean()) >= 0.08

    def _analyze_chunk_columns(self, chunk_img, coarse_pos):
        """
        分析一个粗切 chunk 是否含双栏结构。

        Returns:
            dict: {
                'type': 'single' | 'two_col',
                'img': PIL.Image,
                'pos': position_dict,
                'band': (x0, x1, center, width) or None,
                'yr': (y0, y1) or None,
            }
        """
        result = dict(type='single', img=chunk_img, pos=coarse_pos,
                      band=None, yr=None)

        if not self.enable_two_col_split:
            return result

        gray = np.asarray(chunk_img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        if h < 300:
            return result

        # 先做列级白带检测 + 双栏 y 范围：这是判定"真双栏目录"的核心证据。
        # 真目录在中央有一条 >=30px 的连续纯白列间隙（_find_column_band 的宽白带路径），
        # 且 _find_two_col_y_range 用 empty_frac>=0.85 等严格条件二次校验。
        # 一旦命中这类"强目录证据"，即便存在较多横向分隔线/加粗大标题行，也不应被
        # 表格门控误杀——目录的分节横线、加粗章节标题会触发 h_line_rows 门控，
        # 导致标题未与目录分离、目录跨块无法纵向合并（见 f54920ef）。
        band = self._find_column_band(gray)
        yr = None
        left_rows = right_rows = 0.0
        if band is not None:
            yr = self._find_two_col_y_range(gray, band[0], band[1])
            if yr is not None:
                y0, y1 = yr
                left_col = gray[y0:y1, :band[0]]
                right_col = gray[y0:y1, band[1]:]
                left_rows = float(((left_col < 200).sum(axis=1) >= 10).mean()) if left_col.size else 0.0
                right_rows = float(((right_col < 200).sum(axis=1) >= 10).mean()) if right_col.size else 0.0
        # 左右两栏均有实质内容才算真双栏（各栏"有内容行占比">=0.15）。
        # 排除"单栏左对齐内容"因右侧留白形成的伪列间隙被误判为双栏：
        # 如 21473_p1 单栏条款目录（第X条 纵向堆叠、右侧全空）、21476_p4 手术定义正文；
        # 真双栏目录左右栏行占比均 >=0.27（f54920ef 右 0.32、946aeabc 右 0.36）。
        balanced = (left_rows >= 0.15 and right_rows >= 0.15)
        # 强目录证据：宽白带(>=30px) + 有效双栏 y 范围 + 左右均衡 → 可跳过表格门控
        # （目录的分节横线、加粗章节标题会触发 h_line_rows 门控，需放行，见 f54920ef）。
        strong_toc = (band is not None and band[3] >= 30 and yr is not None and balanced)

        # 门控 0：排除表格（仅在非强目录证据时生效）
        if not strong_toc:
            # 方法 A：形态学水平线检测
            table_lines = self._detect_table_row_lines(chunk_img)
            if table_lines is not None and len(table_lines) >= 3:
                lines_per_kpx = len(table_lines) * 1000.0 / max(1, h)
                if lines_per_kpx >= 0.5:
                    return result

            # 方法 B：行级横线检测（不依赖 cv2）
            # 表格行分隔线 = 暗像素占比 > 35% 的行（横线横跨大部分宽度）
            # 门槛 ≥8：TOC 分隔线通常只有 3-5 条，真表格通常 10+ 条
            row_dark_frac = (gray < 180).mean(axis=1)
            h_line_rows = int((row_dark_frac > 0.35).sum())
            if h_line_rows >= 8:
                return result

            # 门控 1：排除表格样式（行暗率均匀）
            row_dark = (gray < 240).mean(axis=1)
            if row_dark.size > 20:
                q10 = float(np.percentile(row_dark, 10))
                q90 = float(np.percentile(row_dark, 90))
                uniformity = 1.0 - min(1.0, (q90 - q10) / max(1e-6, q90))
                if uniformity >= 0.85:
                    return result

        # 需要有效的白带 + 双栏 y 范围 + 左右栏均衡（排除单栏左对齐内容误判）
        if band is None or yr is None or not balanced:
            return result

        result['type'] = 'two_col'
        result['band'] = band
        result['yr'] = yr
        return result

    def _find_blank_bands(self, projection, threshold, min_band):
        """
        找出所有"强空白带"：连续 >= min_band 行的空白区间。

        Returns:
            list of (start, end): 半开区间[start, end)，区间内均为空白行
        """
        blank = projection >= threshold
        bands = []
        start = None
        n = len(blank)
        for i in range(n):
            if blank[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= min_band:
                        bands.append((start, i))
                    start = None
        if start is not None and n - start >= min_band:
            bands.append((start, n))
        return bands

    def _select_cuts_by_bands(self, height, bands, max_h):
        """
        基于强空白带贪心选择切分边界（T8核心 + T5表格避让）

        策略：
        - 从上一切点pos出发，在 (pos, pos+max_h] 窗口内寻找强空白带中心；
        - 优先靠后60%区间内、且带宽最大的空白带作为切点：
          带越宽越可能是章节/段落间的真空白，越不可能落在表格内部，
          从而尽量避免把一个表格切成两段（T5，纯几何、无需模型）；
        - 若窗口内没有强空白带，则在 pos+max_h 处强制切分，并标记为
          非强切点（该切点两侧后续会补安全重叠）。

        Returns:
            list of (cut_y, is_strong): 不含0与height两个端点
        """
        centers = [((s + e) // 2, e - s) for (s, e) in bands]
        selected = []
        pos = 0
        guard = 0
        # T5表格避让：真章节空白带通常较宽(>=60px)，表格行间空白较窄(30-50px)。
        # 优先在宽空白带切分；若常规窗口内只有窄空白带（可能在表格内部），
        # 则向后扩展搜索宽空白带，宁可chunk略高也不切断表格。
        min_cut_band = 60
        while height - pos > max_h and guard < 100000:
            guard += 1
            window = [(c, wd) for (c, wd) in centers if pos < c <= pos + max_h]
            wide_in_window = [(c, wd) for (c, wd) in window if wd >= min_cut_band]
            if wide_in_window:
                # 常规窗口内有宽空白带：在后60%选最宽的
                near = [(c, wd) for (c, wd) in wide_in_window if c >= pos + max_h * 0.6]
                pick = max(near or wide_in_window, key=lambda x: x[1])
                cut = pick[0]
                if cut - pos < max_h * 0.3:
                    # 宽空白带距上一切点过近（产生碎块）：不直接硬切，
                    # 向后扩展搜索宽空白带，避免硬切带来的重叠重复
                    ext = [(c, wd) for (c, wd) in centers
                           if pos + max_h < c <= pos + max_h * 2 and wd >= min_cut_band]
                    if ext:
                        pick2 = min(ext, key=lambda x: abs(x[0] - (pos + max_h)))
                        selected.append((pick2[0], True))
                        cut = pick2[0]
                    else:
                        cut = pos + max_h
                        selected.append((cut, False))
                else:
                    selected.append((cut, True))
            else:
                # 常规窗口内只有窄空白带（可能在表格内部）：
                # 向后扩展搜索宽空白带，避免切断表格
                ext = [(c, wd) for (c, wd) in centers
                       if pos + max_h < c <= pos + max_h * 2 and wd >= min_cut_band]
                if ext:
                    # 扩展窗口内最靠近理想高度的宽空白带
                    pick = min(ext, key=lambda x: abs(x[0] - (pos + max_h)))
                    selected.append((pick[0], True))
                    cut = pick[0]
                elif window:
                    # 无宽空白带可用：退化为最宽窄空白带（保持不丢内容）
                    near = [(c, wd) for (c, wd) in window if c >= pos + max_h * 0.6]
                    pick = max(near or window, key=lambda x: x[1])
                    cut = pick[0]
                    if cut - pos < max_h * 0.3:
                        cut = pos + max_h
                        selected.append((cut, False))
                    else:
                        selected.append((cut, True))
                else:
                    cut = pos + max_h
                    selected.append((cut, False))
            pos = cut
        return selected

    def chunk_long_document(self, image_path):
        """
        长文档切块（垂直方向切分）

        长文档特征：宽~1500px，高30K-190Kpx

        T8切块策略：优先在"强空白带"中心切分。由于切点落在整段空白中，
        不会跨越任何文字，因此无需重叠即可保证内容不丢不重，从源头消除
        接缝处的重复/截断文本（同时改善Text Edit与Read Order）。仅在
        找不到强空白带、被迫硬切时，才对该切点两侧补安全重叠。
        （通过 use_blank_band_cut=False 可回退到原有等分+全重叠策略。）

        Returns:
            list: [(PIL.Image, (y_start, y_end)), ...]
        """
        # 使用Image.MAX_IMAGE_PIXELS解除限制
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(image_path)

        if ENABLE_LONG_RESIZE:
            w0, h0 = img.size
            new_size = (max(1, int(w0 * LONG_RESIZE_SCALE)), max(1, int(h0 * LONG_RESIZE_SCALE)))
            img = img.resize(new_size, Image.LANCZOS)
            logger.info(f"  长文档实验缩放: {w0}x{h0} -> {new_size[0]}x{new_size[1]}")

        img, crop_offset_x, _ = self._crop_horizontal_margins(img, label="长文档")

        width, height = img.size
        logger.info(f"长文档切块: {os.path.basename(image_path)}, 尺寸={width}x{height}")

        # 抗OOM的水平投影（每行均值），不整图载入RGB数组
        projection = self._row_projection_safe(img)

        if self.use_blank_band_cut:
            bands = self._find_blank_bands(projection, self.blank_threshold,
                                           self.blank_band_min_rows)
            selected = self._select_cuts_by_bands(height, bands, self.max_chunk_height)
            strong_cnt = sum(1 for _, s in selected if s)
            logger.info(f"  强空白带切分: {len(selected) + 1}块, "
                        f"强空白切点={strong_cnt}, 强制硬切={len(selected) - strong_cnt}")
            boundaries = [(0, True)] + selected + [(height, True)]
        else:
            # 回退：原有等分+全重叠策略
            cut_points = self._find_cut_points(projection, 0, height, self.max_chunk_height)
            logger.info(f"  切分为 {len(cut_points) - 1} 块, 切分点: {cut_points[:10]}...")
            boundaries = [(cp, False) for cp in cut_points]
            boundaries[0] = (0, True)
            boundaries[-1] = (height, True)

        # 生成切块：强空白切点零重叠，仅硬切点两侧补安全重叠
        chunks = []
        safety = int(self.max_chunk_height * self.overlap_ratio)
        n = len(boundaries)

        for i in range(n - 1):
            y_start, strong_start = boundaries[i]
            y_end, strong_end = boundaries[i + 1]

            # 仅在被迫硬切的内部边界补重叠；强空白切点无需重叠
            if i > 0 and not strong_start:
                y_start = max(0, y_start - safety)
            if i < n - 2 and not strong_end:
                y_end = min(height, y_end + safety)

            # 裁剪
            chunk = img.crop((0, y_start, width, y_end))

            # 如果需要缩放
            if width > self.max_chunk_width:
                new_width = self.max_chunk_width
                new_height = int(chunk.size[1] * (self.max_chunk_width / width))
                chunk = chunk.resize((new_width, new_height), Image.LANCZOS)

            chunks.append((chunk, self._position(0, y_start, width, y_end, crop_offset_x)))

        # P3 v6：双栏层次切分（列级白带 + 标题保护 + 列优先排序）
        # 第一遍：分析每个粗切 chunk 的类型
        self.last_toc_two_col = 0  # 记录本次检出的双栏(目录)粗块数，供外部识别"含目录"文档
        if self.enable_two_col_split:
            chunk_infos = []
            for c_img, c_pos in chunks:
                info = self._analyze_chunk_columns(c_img, c_pos)
                chunk_infos.append(info)

            # 第二遍：处理连续双栏 chunk 组
            fine_chunks = []
            n_double = 0
            i = 0
            while i < len(chunk_infos):
                ci = chunk_infos[i]
                if ci['type'] == 'single':
                    # 过滤空白粗块（如两个空白带之间的纯白窄条），不送 API
                    if self._has_content(ci['img']):
                        fine_chunks.append((ci['img'], ci['pos']))
                    i += 1
                else:
                    # 收集连续双栏 chunk 组
                    group_start_idx = i  # 组起始粗块索引（判定是否为文档顶部条款目录）
                    group = []
                    while i < len(chunk_infos) and chunk_infos[i]['type'] == 'two_col':
                        group.append(chunk_infos[i])
                        i += 1

                    # 门控：只有【文档顶部(起始粗块索引<=1)】且【篇幅短(<=3粗块)】的双栏组
                    # 才当作"条款目录"做 标题分离/纵向合并。两条缺一即按普通粗块原样输出：
                    # - 起始索引>1：文档中后段被判双栏的长表格（如疾病名称附表，起始>=3）；
                    # - 组>3粗块：即便始于顶部，也是长表格而非目录——真·条款目录极短（实测≤2粗块），
                    #   而"疾病名称附表"等续页长表格会从顶部起连续横跨十余粗块（如21478_p8起始=1、11粗块）。
                    # 误走目录merge会把首行切成~200px碎块、整表拼成超高巨块(如18549px)，"看不出是表格"。
                    MAX_TOC_BLOCKS = 3
                    if group_start_idx > 1 or len(group) > MAX_TOC_BLOCKS:
                        for g in group:
                            if self._has_content(g['img']):
                                fine_chunks.append((g['img'], g['pos']))
                        continue

                    # 组扩展：紧邻的 single 粗块可能是"目录延续 + 其他内容"混合块，
                    # 其自身白带检测被非目录内容干扰而失效，用组的白带位置重检其顶部目录延续
                    if self.two_col_mode == 'merge' and group:
                        ref_band = group[-1]['band']
                        while i < len(chunk_infos) and chunk_infos[i]['type'] == 'single':
                            nxt = chunk_infos[i]
                            g_gray = np.asarray(nxt['img'].convert('L'), dtype=np.uint8)
                            yr_nxt = self._find_two_col_y_range(g_gray, ref_band[0], ref_band[1])
                            nh = nxt['img'].size[1]
                            # 目录延续：检测到 TOC 且始于 chunk 上部(<30%)、有一定高度
                            if (yr_nxt is not None and yr_nxt[0] < nh * 0.3
                                    and yr_nxt[1] - yr_nxt[0] >= 100):
                                nxt_ext = dict(nxt)
                                nxt_ext['type'] = 'two_col'
                                nxt_ext['band'] = ref_band
                                nxt_ext['yr'] = yr_nxt
                                group.append(nxt_ext)
                                i += 1
                            else:
                                break
                    n_double += len(group)

                    if self.two_col_mode == 'merge':
                        # === merge 模式：标题/纯目录/后续段落 三段分离 ===
                        first_g = group[0]
                        first_img = first_g['img']
                        fw, fh = first_img.size
                        fy0, fy1 = first_g['yr']
                        first_pos = first_g['pos']
                        fcy = first_pos.get('y_start', 0) if isinstance(first_pos, dict) else 0

                        # 1) 第一个 chunk 的标题区单独切出（过滤空白块）
                        if fy0 > 10:
                            title_img = first_img.crop((0, 0, fw, fy0))
                            if self._has_content(title_img):
                                title_pos = self._position(0, fcy, fw, fcy + fy0, 0, 'full')
                                fine_chunks.append((title_img, title_pos))

                        # 2) 纯目录：第一个 chunk (fy0~) + 后续 chunk 目录区，纵向拼接。
                        #    单栏目录尾（如"第七/八条"、名词释义续条）位于 yr 之下且非正文段落，
                        #    应并入目录（延伸到块底）；仅"正文段落"(justified 全宽多行) 才单独分离。
                        first_bot = None
                        first_toc_end = fy1
                        if fh - fy1 > 10:
                            if self._below_is_body(first_img, fy1):
                                first_bot = (fy1, fh)
                            else:
                                first_toc_end = fh  # 单栏目录尾并入目录
                        toc_imgs = [first_img.crop((0, fy0, fw, first_toc_end))]
                        extra_parts = []
                        for g in group[1:]:
                            img_g = g['img']
                            w_g, h_g = img_g.size
                            gy0, gy1 = g['yr']
                            cy_g = g['pos'].get('y_start', 0) if isinstance(g['pos'], dict) else 0
                            # 目录区域：延续块顶部本就是目录（无真标题），从顶部取到 gy1，
                            # 避免标题跳过逻辑把第一个目录条目误切出去
                            g_toc_end = gy1
                            if h_g - gy1 > 10:
                                if self._below_is_body(img_g, gy1):
                                    # 下部为正文段落（如混合块中的扉页）→ 单独输出
                                    bot_img_g = img_g.crop((0, gy1, w_g, h_g))
                                    if self._has_content(bot_img_g):
                                        extra_parts.append((bot_img_g, self._position(0, cy_g + gy1, w_g, cy_g + h_g, 0, 'full')))
                                else:
                                    g_toc_end = h_g  # 单栏目录尾并入目录
                            toc_imgs.append(img_g.crop((0, 0, w_g, g_toc_end)))
                        toc_h = sum(im.size[1] for im in toc_imgs)
                        toc_w = max(im.size[0] for im in toc_imgs)
                        toc_merged = Image.new('RGB', (toc_w, toc_h), (255, 255, 255))
                        y_off = 0
                        for im in toc_imgs:
                            toc_merged.paste(im, (0, y_off))
                            y_off += im.size[1]
                        toc_pos = self._position(0, fcy + fy0, toc_w,
                                                  fcy + fy0 + toc_h, 0, 'full')
                        fine_chunks.append((toc_merged, toc_pos))

                        # 3) 第一个 chunk 的后续正文段落（仅当判定为正文时单独输出）
                        if first_bot is not None:
                            bot_img = first_img.crop((0, first_bot[0], fw, first_bot[1]))
                            if self._has_content(bot_img):
                                bot_pos = self._position(0, fcy + first_bot[0], fw, fcy + first_bot[1], 0, 'full')
                                fine_chunks.append((bot_img, bot_pos))

                        # 4) 后续 chunk 的正文部分（如混合块中的扉页），按序补出
                        for img_p, pos_p in extra_parts:
                            fine_chunks.append((img_p, pos_p))

                    else:
                        # === column 模式：左右切分 + 列优先排序 ===
                        # 1) 标题区（按 chunk 顺序）
                        for g in group:
                            img = g['img']
                            w, h = img.size
                            y0, y1 = g['yr']
                            pos = g['pos']
                            cy = pos.get('y_start', 0) if isinstance(pos, dict) else 0
                            if y0 > 10:
                                title_img = img.crop((0, 0, w, y0))
                                title_pos = self._position(0, cy, w, cy + y0, 0, 'full')
                                fine_chunks.append((title_img, title_pos))

                        # 2) 全部左栏（从上到下）
                        for g in group:
                            img = g['img']
                            w, h = img.size
                            bx0, bx1, bc, bw = g['band']
                            y0, y1 = g['yr']
                            pos = g['pos']
                            cy = pos.get('y_start', 0) if isinstance(pos, dict) else 0
                            left_img = img.crop((0, y0, bx1, y1))
                            left_pos = self._position(0, cy + y0, bx1, cy + y1, 0, 'left')
                            fine_chunks.append((left_img, left_pos))

                        # 3) 全部右栏（从上到下）
                        for g in group:
                            img = g['img']
                            w, h = img.size
                            bx0, bx1, bc, bw = g['band']
                            y0, y1 = g['yr']
                            pos = g['pos']
                            cy = pos.get('y_start', 0) if isinstance(pos, dict) else 0
                            right_img = img.crop((bx1, y0, w, y1))
                            right_pos = self._position(bx1, cy + y0, w, cy + y1, 0, 'right')
                            fine_chunks.append((right_img, right_pos))

                        # 4) 底部区（按 chunk 顺序）
                        for g in group:
                            img = g['img']
                            w, h = img.size
                            y0, y1 = g['yr']
                            pos = g['pos']
                            cy = pos.get('y_start', 0) if isinstance(pos, dict) else 0
                            if h - y1 > 10:
                                bot_img = img.crop((0, y1, w, h))
                                bot_pos = self._position(0, cy + y1, w, cy + h, 0, 'full')
                                fine_chunks.append((bot_img, bot_pos))

            if n_double > 0:
                logger.info(f"  双栏检测(v6,{self.two_col_mode}): "
                            f"{n_double}/{len(chunks)} 个粗块含双栏, "
                            f"最终 sub-chunk 数 {len(fine_chunks)}")
            self.last_toc_two_col = n_double
            chunks = fine_chunks

        return chunks

    def _detect_table_row_lines(self, img):
        """
        用 OpenCV 形态学检测表格中的水平分隔线（行边界）。

        理论：有边框的金融费率表有清晰的水平线，每条线 = 行分隔。
        用形态学开运算（宽核）提取水平线，纯 CPU、零参数、合规。
        PoC 验证：有边框表 147≈GT144、216≈GT214，几乎完美。

        Returns:
            list[int] | None: 检测到的水平线 y 坐标（排序），或 None（未检测到足够线）
        """
        if not HAS_CV2:
            return None
        try:
            Image.MAX_IMAGE_PIXELS = None
            gray = np.array(img.convert('L'))
            h, w = gray.shape
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            kernel_w = max(int(w * 0.3), 50)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)
            row_sums = h_lines.sum(axis=1)
            line_rows = np.where(row_sums > 0)[0]
            if len(line_rows) < 5:
                return None
            # 合并相邻像素行
            lines_y = []
            start = line_rows[0]
            for i in range(1, len(line_rows)):
                if line_rows[i] - line_rows[i-1] > 3:
                    lines_y.append(int((start + line_rows[i-1]) // 2))
                    start = line_rows[i]
            lines_y.append(int((start + line_rows[-1]) // 2))
            if len(lines_y) < 5:
                return None
            return lines_y
        except Exception as e:
            logger.warning(f"  CV水平线检测异常: {e}")
            return None
