"""
区域分割模块（Level 1）

将表格文档图片分割为多个独立区域（表格、文字说明），
然后每个区域再由 image_chunker 做 Level 2 的细切。

分割策略（层级式）：
1. 裁剪四周空白边
2. 水平切分（按空白行）：
   - 正常情况：在显著空白带处切分
   - 无边框表格特殊处理：根据空白带宽度分布，
     只在"异常宽"的空白带处切分（表间空白 >> 行间空白）
3. 连接表格检测（右边界突变检测）：
   - 如果水平切分后仍只有 1 块，分析每行最右侧内容像素位置
   - 连续多行右边界一致 → 同一表格
   - 右边界发生突变 → 不同表格的交界
4. 垂直切分（按空白列）：
   - 对每个水平切分结果，检查是否有左右并排的表格
   - 同样有正常模式和自适应模式
5. 递归验证：对切分结果再次检查是否需要进一步切分
"""
import os
import logging
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

logger = logging.getLogger("region_splitter")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(handler)


# ==================== 配置参数 ====================
# 空白像素判定阈值：灰度值 >= 此值认为是空白（白色）
WHITE_THRESHOLD = 252

# 水平切分参数
MIN_BLANK_BAND_HEIGHT = 3       # 最小空白带高度（像素），低于此值不认为是空白带
ADAPTIVE_GAP_RATIO = 2.0        # 自适应模式：空白带宽度 > 中位数 * 此系数 才切分

# 连接表格检测参数
EDGE_MIN_CONSECUTIVE = 5        # 最少连续行数（右边界一致才算同一区域）
EDGE_TOLERANCE = 20             # 右边界坐标容差（像素），±此范围内认为"一致"
EDGE_MIN_SHIFT = 50             # 右边界变化最小像素，才认为是不同表格

# 垂直切分参数
MIN_BLANK_BAND_WIDTH = 3        # 最小空白带宽度（像素）

# 区域最小尺寸
MIN_REGION_HEIGHT = 20          # 最小区域高度（像素），更小的丢弃
MIN_REGION_WIDTH = 20           # 最小区域宽度（像素）

# 裁剪边距
TRIM_PADDING = 10               # 裁剪时保留的安全边距（像素），避免裁掉表格边框线


class Region:
    """表示图片中的一个独立区域"""

    def __init__(self, image: Image.Image, bbox: Tuple[int, int, int, int],
                 region_type: str = 'unknown', order: int = -1):
        """
        Args:
            image: 裁剪出的区域图像
            bbox: 在原图中的坐标 (x0, y0, x1, y1)
            region_type: 'table' / 'text' / 'unknown'
            order: 阅读顺序编号（从0开始，按从上到下、从左到右排列）
        """
        self.image = image
        self.bbox = bbox
        self.region_type = region_type
        self.order = order
        self.no_merge = False  # 标记为不应被 Step 5b 合并回 table

    @property
    def width(self):
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self):
        return self.bbox[3] - self.bbox[1]

    def __repr__(self):
        return (f"Region(#{self.order} type={self.region_type}, "
                f"bbox={self.bbox}, size={self.width}x{self.height})")


class RegionSplitter:
    """层级式区域分割器"""

    def __init__(self,
                 white_threshold=WHITE_THRESHOLD,
                 min_blank_band_height=MIN_BLANK_BAND_HEIGHT,
                 adaptive_gap_ratio=ADAPTIVE_GAP_RATIO,
                 edge_min_consecutive=EDGE_MIN_CONSECUTIVE,
                 edge_tolerance=EDGE_TOLERANCE,
                 edge_min_shift=EDGE_MIN_SHIFT,
                 min_blank_band_width=MIN_BLANK_BAND_WIDTH,
                 min_region_height=MIN_REGION_HEIGHT,
                 min_region_width=MIN_REGION_WIDTH,
                 trim_padding=TRIM_PADDING):
        self.white_threshold = white_threshold
        self.min_blank_band_height = min_blank_band_height
        self.adaptive_gap_ratio = adaptive_gap_ratio
        self.edge_min_consecutive = edge_min_consecutive
        self.edge_tolerance = edge_tolerance
        self.edge_min_shift = edge_min_shift
        self.min_blank_band_width = min_blank_band_width
        self.min_region_height = min_region_height
        self.min_region_width = min_region_width
        self.trim_padding = trim_padding

    def split(self, image_path: str) -> List[Region]:
        """
        对一张表格文档图片进行层级式区域分割。

        Args:
            image_path: 图片路径

        Returns:
            list[Region]: 分割出的区域列表（按阅读顺序：从上到下、从左到右）
        """
        Image.MAX_IMAGE_PIXELS = None
        img = Image.open(image_path)
        logger.info(f"区域分割: {os.path.basename(image_path)}, 原始尺寸={img.size[0]}x{img.size[1]}")

        # Step 0: 裁剪四周空白
        trimmed_img, trim_offset = self._trim_margins(img)
        ox, oy = trim_offset
        logger.info(f"  裁剪后尺寸: {trimmed_img.size[0]}x{trimmed_img.size[1]}, 偏移=({ox},{oy})")

        # Step 1: 水平切分
        h_regions = self._split_horizontal(trimmed_img)
        logger.info(f"  水平切分: {len(h_regions)} 个区域")

        # Step 2: 对只有 1 块的情况，尝试连接表格检测（右边界突变）
        # 曾尝试扩展到所有水平区域以修复 deb8de95 类"内部两表连接"，但发现会误切
        # 5b93ec6f/425fb187 类的 header/footer 短行（右边界比数据行短，被误判为
        # 突变）。deb8de95 的主体分隔应用其他检测方法（如内部左侧细窄标签检测），
        # 不属于 Step 2 能自然覆盖的场景。故保持原逻辑不变。
        if len(h_regions) == 1:
            edge_regions = self._split_by_edge_change(h_regions[0])
            if len(edge_regions) > 1:
                h_regions = edge_regions
                logger.info(f"  右边界突变检测: 切分为 {len(h_regions)} 个区域")

        # Step 2.5: 对每个高的水平区域，检测"wide/non-wide 段落交替"结构
        # 修复 deb8de95：内部有"表 A（右边界递减）+ 表 B（跨全宽）+ 表 C（右边界
        # 递减）"三段结构，中间跨全宽段与两侧递减段之间应切开。
        # 严格保守约束（h>1500、每段≥200行、每段≥15%高度）确保不影响
        # 5b93ec6f/425fb187 类的 header+data / data+footer 结构。
        new_h_regions = []
        pattern_split_count = 0
        for h_region in h_regions:
            split_regions = self._split_by_internal_pattern(h_region)
            if len(split_regions) > 1:
                pattern_split_count += 1
            new_h_regions.extend(split_regions)
        if pattern_split_count > 0:
            logger.info(f"  内部模式切分: {pattern_split_count} 个水平区域被切分, "
                        f"总数 {len(h_regions)} → {len(new_h_regions)}")
        h_regions = new_h_regions

        # Step 3: 对每个水平区域，尝试垂直切分
        all_regions = []
        for h_region in h_regions:
            v_regions = self._split_vertical(h_region)
            if len(v_regions) > 1:
                logger.info(f"    垂直切分: bbox={h_region[:2]} → {len(v_regions)} 个子区域")
            all_regions.extend(v_regions)

        # Step 4: 构建最终 Region 对象（坐标映射回原图）+ 子图空白裁剪
        results = []
        for (sub_img, local_bbox) in all_regions:
            # local_bbox 是相对于 trimmed_img 的，需要加上 trim_offset
            x0, y0, x1, y1 = local_bbox
            orig_bbox = (x0 + ox, y0 + oy, x1 + ox, y1 + oy)

            # 对每个子图做空白裁剪（保留安全边距）
            sub_img, sub_offset = self._trim_margins(sub_img)
            # 更新 bbox
            sx, sy = sub_offset
            orig_bbox = (orig_bbox[0] + sx, orig_bbox[1] + sy,
                         orig_bbox[0] + sx + sub_img.size[0],
                         orig_bbox[1] + sy + sub_img.size[1])

            region_type = self._classify_region(sub_img)
            results.append(Region(sub_img, orig_bbox, region_type))

        # 过滤掉太小的区域和纯白/近白的空区域
        results = [r for r in results
                   if r.height >= self.min_region_height
                   and r.width >= self.min_region_width
                   and not self._is_empty_region(r.image)]

        # Step 5: 对每个 table 区域，检查顶部是否有独立说明文字，如果有则切出去
        # 循环剥离：剥离一次后可能暴露新的顶部标题（如先剥"XXX保险利益表"，
        # 再剥"保单年度"），最多循环 3 次防止无限递归
        for _strip_pass in range(3):
            expanded = []
            changed = False
            for r in results:
                if r.region_type == 'table':
                    stripped = self._strip_top_text_from_table(r)
                    if len(stripped) > 1:
                        changed = True
                    expanded.extend(stripped)
                else:
                    expanded.append(r)
            results = expanded
            if not changed:
                break

        # Step 5-left: 对每个 table 区域，检查左侧是否有窄标签列 + 大空白
        # （如"投保年龄"79px + 578px 空白 → 切出"投保年龄"为 text）
        # 判据：左侧有 >= 100px 的连续空白竖条，且其左边内容宽度 < 图宽 15%
        # 切分点：逐列分析找第一个有多行内容的 x（避免切到行号数字）
        expanded = []
        for r in results:
            if r.region_type == 'table' and r.width > 500:
                stripped = self._strip_left_label_from_table(r)
                expanded.extend(stripped)
            else:
                expanded.append(r)
        results = expanded

        # Step 5a: 对每个 table 区域，检查底部是否有"反弹段"文字（内容 xe 局部反弹）
        # 典型：deb8de95 r00 底部的"男性 保单年度末"3 行字（xe 从 ~100 反弹到 ~1700）
        expanded = []
        for r in results:
            if r.region_type == 'table':
                stripped = self._strip_bottom_rebound_text(r)
                expanded.extend(stripped)
            else:
                expanded.append(r)
        results = expanded

        # Step 5a-footnote: 对每个 table 区域，检查底部是否有"窄注释行"
        # （全宽表格 + 底部左对齐窄文字，xe 骤降）
        # 典型：8e60d5f8 底部的"注：根据实际业务需要..."
        expanded = []
        for r in results:
            if r.region_type == 'table':
                stripped = self._strip_bottom_footnote(r)
                expanded.extend(stripped)
            else:
                expanded.append(r)
        results = expanded

        # Step 5-span: 内容行跨度跳跃检测
        # 对每个 table，按空白行分割为内容行，找第一个"<70% → >=70%"的跨度跳跃
        # 跳跃前的内容行是 metadata（如"一次交清"、"投保年龄"），应该剥离
        # 只执行 1 次（不循环），避免过度剥离
        expanded = []
        for r in results:
            if r.region_type == 'table' and r.height >= 200:
                stripped = self._strip_top_by_span_jump(r)
                expanded.extend(stripped)
            else:
                expanded.append(r)
        results = expanded

        # Step 5b: 合并紧邻 table 顶部的宽 text（跨列合并单元格表头，如"被保险人"）
        # 需要原图 img 重新裁剪合并后的 table
        results = self._merge_wide_text_into_table(results, img)

        # Step 5c: 合并紧邻 table 左右侧的细长竖条 text（表格内稀疏列被误切，如全"0"列）
        results = self._merge_side_text_into_table(results, img)

        # Step 5d: 合并上下相邻、宽度相同、中间无 region 阻挡的 table
        # 典型场景 (945e8fe9)：多层 header + 数据主体被 Step 1 空白带误切成两段
        results = self._merge_adjacent_tables(results, img)

        # Step 6: 对每个 text 区域做内部水平切分（处理左中右分布的文字）
        expanded = []
        for r in results:
            if r.region_type == 'text':
                split = self._split_text_region_horizontally(r)
                expanded.extend(split)
            else:
                expanded.append(r)
        results = expanded

        # 按阅读顺序排序：从上到下为主，同一行（y 重叠）从左到右
        results = self._sort_reading_order(results)

        # 分配序号
        for i, r in enumerate(results):
            r.order = i

        logger.info(f"  最终分割: {len(results)} 个区域 "
                    f"(表格={sum(1 for r in results if r.region_type == 'table')}, "
                    f"文字={sum(1 for r in results if r.region_type == 'text')})")
        return results

    # ==================== Step 0: 裁剪四周空白 ====================

    def _trim_margins(self, img: Image.Image) -> Tuple[Image.Image, Tuple[int, int]]:
        """
        裁剪四周空白边，返回裁剪后的图像和偏移量。

        Returns:
            (trimmed_image, (offset_x, offset_y))
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 非白像素掩码
        content_mask = gray < self.white_threshold

        # 找内容边界（要求每行/列至少 3 个非白像素，避免 JPG 噪声/压缩伪影被视为内容）
        row_has_content = content_mask.sum(axis=1) > 3
        col_has_content = content_mask.sum(axis=0) > 3

        content_rows = np.where(row_has_content)[0]
        content_cols = np.where(col_has_content)[0]

        if len(content_rows) == 0 or len(content_cols) == 0:
            # 全白图
            return img, (0, 0)

        y0 = max(0, int(content_rows[0]) - self.trim_padding)
        y1 = min(h, int(content_rows[-1]) + 1 + self.trim_padding)
        x0 = max(0, int(content_cols[0]) - self.trim_padding)
        x1 = min(w, int(content_cols[-1]) + 1 + self.trim_padding)

        trimmed = img.crop((x0, y0, x1, y1))
        return trimmed, (x0, y0)

    # ==================== Step 1: 水平切分 ====================

    def _split_horizontal(self, img: Image.Image) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        按水平空白带切分图像。

        返回值: [(sub_img, (x0, y0, x1, y1)), ...] 其中坐标相对于输入 img
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 每行的非白像素数
        row_content = (gray < self.white_threshold).sum(axis=1)

        # 找空白行（非白像素数 == 0 或极少）
        # 允许少量噪声像素（<= 3px），避免单个噪点阻断切分
        is_blank_row = row_content <= 3

        # 找连续空白带
        blank_bands = self._find_continuous_bands(is_blank_row, self.min_blank_band_height)

        # 排除紧贴图片边缘的空白带（这些只是裁剪留的边距，不是内容分隔）
        blank_bands = [(s, e) for s, e in blank_bands if s > 0 and e < h]

        if not blank_bands:
            # 没有空白带，返回整图作为单个区域
            return [(img, (0, 0, w, h))]

        # 计算空白带宽度
        band_widths = [end - start for start, end in blank_bands]

        # 自适应判断：是否为无边框表格（大量等宽空白带）
        cut_bands = self._select_cut_bands(blank_bands, band_widths)

        if not cut_bands:
            return [(img, (0, 0, w, h))]

        # 按 cut_bands 切分
        regions = []
        cut_points = [0]
        for start, end in cut_bands:
            # 切分点取空白带中心
            cut_points.append((start + end) // 2)
        cut_points.append(h)

        for i in range(len(cut_points) - 1):
            y0 = cut_points[i]
            y1 = cut_points[i + 1]
            if y1 - y0 < self.min_region_height:
                # 太短的段合并到上一个区域（而非丢弃），避免遗漏文字
                if regions:
                    prev_img, prev_bbox = regions[-1]
                    merged_y0 = prev_bbox[1]
                    merged_img = img.crop((0, merged_y0, w, y1))
                    regions[-1] = (merged_img, (0, merged_y0, w, y1))
                continue
            sub_img = img.crop((0, y0, w, y1))
            regions.append((sub_img, (0, y0, w, y1)))

        return regions if regions else [(img, (0, 0, w, h))]

    def _select_cut_bands(self, blank_bands, band_widths):
        """
        选择作为切分点的空白带。

        核心逻辑：
        - 如果空白带数量少（<= 10）且宽度分布均匀 → 全部作为切分点
        - 如果空白带数量多且宽度分布有明显的两个层级 → 只在"宽"的空白带处切分
          （无边框表格：行间空白窄而均匀，表间空白明显更宽）
        """
        if not blank_bands:
            return []

        n = len(blank_bands)
        widths = np.array(band_widths)

        # 少量空白带（<= 10 个）：直接全切
        # 理由：10个以内的空白带大概率是表格/段落之间的分隔，不是单表格行间空白
        if n <= 10:
            return blank_bands

        # 多量空白带：分析宽度分布
        median_width = float(np.median(widths))
        max_width = float(np.max(widths))

        # 如果最大宽度与中位数差距不大（< 2 倍），说明空白带宽度均匀
        # 这可能是一个无边框表格的行间空白，不应该切分
        if max_width < median_width * self.adaptive_gap_ratio:
            # 空白带宽度均匀 → 可能是单个无边框表格，不切
            # 但如果空白带特别宽（绝对值大），还是要切
            if median_width < 30:
                logger.info(f"    自适应: {n}个空白带, 中位宽={median_width:.0f}px, "
                            f"最大={max_width:.0f}px → 判定为单表格行间空白，不切")
                return []
            else:
                # 中位数本身就很宽，说明确实有分隔
                return blank_bands

        # 宽度分布有两个层级：只在超过阈值的空白带处切分
        threshold = median_width * self.adaptive_gap_ratio
        cut_bands = [(s, e) for (s, e), w in zip(blank_bands, band_widths)
                     if w >= threshold]

        logger.info(f"    自适应: {n}个空白带, 中位宽={median_width:.0f}px, "
                    f"阈值={threshold:.0f}px → 选择{len(cut_bands)}个切分点")
        return cut_bands

    # ==================== Step 2: 连接表格检测（右边界突变） ====================

    def _split_by_edge_change(self, region_tuple: Tuple[Image.Image, Tuple[int, int, int, int]]) \
            -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        通过分析每行最右侧内容像素位置的突变来切分连接的表格。

        对于连在一起的多个表格（无空白行分隔），它们通常宽度不同，
        导致每行的右边界位置会发生阶梯式变化。

        保守策略：
        - 每个切分段至少占总高度 10%（避免切碎）
        - 右边界变化至少 15% 图宽（避免对同一表格内的微小宽度波动误切）
        - 最多切 5 段（一张图里超过 5 个连接表格极为罕见）
        """
        sub_img, local_bbox = region_tuple
        gray = np.array(sub_img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 区域太小不做边界检测
        min_segment_height = max(self.edge_min_consecutive * 3, int(h * 0.10))
        if h < min_segment_height * 2:
            return [region_tuple]

        content_mask = gray < self.white_threshold

        # 计算每行的最右侧内容像素 x 坐标
        right_edges = np.full(h, -1, dtype=np.int32)
        for y in range(h):
            row_content_cols = np.where(content_mask[y])[0]
            if len(row_content_cols) > 0:
                right_edges[y] = int(row_content_cols[-1])

        # 动态设置最小边界位移：至少 15% 图宽
        dynamic_min_shift = max(self.edge_min_shift, int(w * 0.15))

        cut_points = self._detect_edge_transitions(right_edges, h,
                                                    min_shift=dynamic_min_shift,
                                                    min_segment=min_segment_height)

        if not cut_points:
            return [region_tuple]

        # 最多切 5 段
        if len(cut_points) > 4:
            logger.info(f"    边界突变检测到 {len(cut_points)} 个切点，可能误判，放弃")
            return [region_tuple]

        # 按突变点切分
        x0_base, y0_base, x1_base, y1_base = local_bbox
        all_cuts = [0] + cut_points + [h]
        regions = []
        for i in range(len(all_cuts) - 1):
            cy0 = all_cuts[i]
            cy1 = all_cuts[i + 1]
            if cy1 - cy0 < self.min_region_height:
                continue
            cropped = sub_img.crop((0, cy0, w, cy1))
            bbox = (x0_base, y0_base + cy0, x1_base, y0_base + cy1)
            regions.append((cropped, bbox))

        return regions if regions else [region_tuple]

    def _detect_edge_transitions(self, right_edges: np.ndarray, h: int,
                                   min_shift: int = None,
                                   min_segment: int = None) -> List[int]:
        """
        检测右边界的阶梯式突变。

        策略：
        - 用滑动窗口计算局部右边界中位数
        - 当相邻稳定段的中位数差异 > min_shift 时，标记为突变点
        - 每段至少 min_segment 行
        """
        min_shift = min_shift or self.edge_min_shift
        min_segment = min_segment or self.edge_min_consecutive
        window = max(self.edge_min_consecutive, min_segment // 2)

        if h < window * 2:
            return []

        # 对有内容的行计算右边界的局部中位数（忽略空白行）
        smoothed = np.full(h, -1.0)
        half_w = window
        for y in range(h):
            start = max(0, y - half_w)
            end = min(h, y + half_w + 1)
            segment = right_edges[start:end]
            valid = segment[segment >= 0]
            if len(valid) >= 3:
                smoothed[y] = float(np.median(valid))

        # 分段：找到右边界稳定的连续段
        segments = []  # [(start_y, end_y, median_edge), ...]
        y = 0
        while y < h:
            if smoothed[y] < 0:
                y += 1
                continue
            # 开始一个新段
            seg_start = y
            seg_val = smoothed[y]
            # 找这个段的结束位置
            while y < h and (smoothed[y] < 0 or abs(smoothed[y] - seg_val) < self.edge_tolerance):
                if smoothed[y] >= 0:
                    seg_val = smoothed[y]  # 跟随缓慢变化
                y += 1
            seg_end = y
            if seg_end - seg_start >= min_segment:
                # 重新计算这个段的中位数
                seg_edges = right_edges[seg_start:seg_end]
                valid_edges = seg_edges[seg_edges >= 0]
                if len(valid_edges) > 0:
                    median_edge = float(np.median(valid_edges))
                    segments.append((seg_start, seg_end, median_edge))

        # 在相邻段之间，如果右边界中位数差异够大，就标记为切分点
        transitions = []
        for i in range(1, len(segments)):
            prev_start, prev_end, prev_edge = segments[i - 1]
            curr_start, curr_end, curr_edge = segments[i]
            shift = abs(curr_edge - prev_edge)
            if shift >= min_shift:
                # 切分点取两段之间的中点
                cut_y = (prev_end + curr_start) // 2
                transitions.append(cut_y)

        return transitions

    # ==================== Step 2.5: 内部 wide/non-wide 模式切分 ====================

    def _split_by_internal_pattern(self, region_tuple: Tuple[Image.Image, Tuple[int, int, int, int]]) \
            -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        在水平区域内部按"wide / non-wide 行段落"检测分隔点，切成多个子区域。

        典型场景 (deb8de95)：一个大区域内部含"递减段 + 跨全宽段 + 递减段"三段结构，
        中间跨全宽段（表 B）和两侧递减段（表 A、表 C）之间的边界应作为切分点。

        判据（严格保守，避免误切 header/footer 短行）：
        1. 区域高度 > 1500 px
        2. 每行分类：wide (x_end >= 图宽 70%) 或 non-wide (含空白行)
        3. 找连续段落，只保留长度 >= 200 行的段
        4. 必须同时存在 wide 和 non-wide 段（说明是"混合结构"）
        5. 相邻不同类型段之间才切分
        6. 切分后每段高度 >= 15% 原区域高度（双重保险）

        为什么不会误切之前 Step 2 扩展遇到的回归：
        - 5b93ec6f 的 header 只有 ~162 行 < 200 行阈值 → 段被过滤 → 不切
        - 425fb187 的 footer 短行段 < 200 行 → 段被过滤 → 不切
        """
        sub_img, local_bbox = region_tuple
        gray = np.array(sub_img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 保守约束 1：太矮不做
        if h < 1500:
            return [region_tuple]

        content_mask = gray < self.white_threshold

        # 计算每行的 x_end（最右侧非白像素的 x 坐标）
        row_x_end = np.full(h, -1, dtype=np.int32)
        for y in range(h):
            cols = np.where(content_mask[y])[0]
            if len(cols) > 0:
                row_x_end[y] = int(cols[-1])

        # 计算每行的内容像素数（用于分隔证据检测）
        row_content = content_mask.sum(axis=1)

        # 分类：wide (x_end >= 图宽 70%) 或 non-wide (含空白行)
        wide_th = int(w * 0.70)
        is_wide = row_x_end >= wide_th

        # 【关键】用滑窗平滑 is_wide 数组：让密集的 wide 行段连成一片，
        # 避免"数据行的行间空白（non-wide）打断 wide 段落"。
        # 例如 deb8de95 表 B 内部每 16 行只有 9 行 wide + 7 行行间空白，
        # 平滑后 wide 段能达到 900+ 行的连续长度。
        kernel_size = 31
        if h >= kernel_size:
            kernel = np.ones(kernel_size) / kernel_size
            wide_ratio = np.convolve(is_wide.astype(float), kernel, mode='same')
            is_wide = wide_ratio >= 0.5

        # 找连续段
        segments = []  # (start_y, end_y, is_wide)
        y = 0
        while y < h:
            t = bool(is_wide[y])
            s = y
            while y < h and bool(is_wide[y]) == t:
                y += 1
            segments.append((s, y, t))

        # 只保留长度 >= 200 行的段（保守约束 3）
        min_seg_rows = 200
        valid_segments = [(s, e, t) for s, e, t in segments if e - s >= min_seg_rows]

        # 至少 2 段，且必须同时存在 wide 和 non-wide
        if len(valid_segments) < 2:
            return [region_tuple]

        # 必须同时存在 wide 和 non-wide 段（保守约束 4）
        types_present = {t for _, _, t in valid_segments}
        if len(types_present) < 2:
            return [region_tuple]

        # 找切分点：**只在"non-wide → wide"过渡处切**
        # 理由：这类过渡表示"某种非全宽内容 → 新表格开始跨全宽"，通常是不同表格
        # 的分界（如 deb8de95 表 A 金字塔 → 表 B 阶梯表开始）。
        # 反之"wide → non-wide"过渡不切——因为可能是**同一阶梯表格的自然递减**
        # （如 deb8de95 表 B 从跨全宽逐渐变窄）。
        candidate_cuts = []  # list of (cut_y, prev_seg_start, prev_seg_end)
        for i in range(1, len(valid_segments)):
            prev_s = valid_segments[i - 1][0]
            prev_e = valid_segments[i - 1][1]
            prev_t = valid_segments[i - 1][2]
            curr_s = valid_segments[i][0]
            curr_t = valid_segments[i][2]
            if (not prev_t) and curr_t:  # non-wide → wide
                candidate_cuts.append(((prev_e + curr_s) // 2, prev_s, prev_e))

        # 【关键防误切约束】要求前段（non-wide）尾部有"内容衰减"作为分隔证据
        # 对比 deb8de95 vs ab129e6b：
        # - deb8de95 前段尾部 content ≈ 100 << 内部 avg ≈ 500 (有 3 行字/金字塔尾)
        # - ab129e6b 前段尾部 content ≈ 527 ≈ 内部 avg ≈ 530 (同一表连续数据)
        # 只有 deb8de95 有明确的"分隔证据"，才应该切。
        cut_points = []
        tail_rows = 30  # 尾部窗口大小
        for cut_y, prev_s, prev_e in candidate_cuts:
            seg_h = prev_e - prev_s
            if seg_h < tail_rows * 2:  # 段太短，跳过
                continue
            # 段内部（去掉尾部窗口后的部分）的 avg content
            inner_end = prev_e - tail_rows
            inner_avg = float(row_content[prev_s:inner_end].mean())
            # 段尾部窗口的 avg content
            tail_avg = float(row_content[inner_end:prev_e].mean())
            # 尾部 content 必须 <= 内部 × 0.5（显著衰减）才认为有分隔证据
            if inner_avg > 0 and tail_avg / inner_avg <= 0.5:
                # 切分点微调：吸附到 non-wide 段末尾附近**最长的空白带**中心。
                # 关键：向上搜索 100 行覆盖 non-wide 段末尾（含 3 行字），
                # 让切分点回退到金字塔真正结束之后的空白带，把 3 行字完整留给下段。
                snapped = self._snap_cut_to_blank_row(cut_y, row_content, w,
                                                      search_up=100, search_down=10)
                cut_points.append(snapped)
                logger.debug(f"    分隔证据: y={cut_y}→snapped={snapped}, "
                             f"prev_seg[{prev_s},{prev_e}) "
                             f"inner_avg={inner_avg:.0f} tail_avg={tail_avg:.0f} "
                             f"ratio={tail_avg/inner_avg:.2f}")

        if not cut_points:
            return [region_tuple]

        # 验证切分后每段高度 >= 15% 原区域高度（保守约束 6）
        min_seg_height = max(int(h * 0.15), 100)
        all_cuts = [0] + cut_points + [h]
        for i in range(len(all_cuts) - 1):
            if all_cuts[i + 1] - all_cuts[i] < min_seg_height:
                return [region_tuple]

        # 按切分点切分
        x0_base, y0_base, x1_base, y1_base = local_bbox
        result = []
        for i in range(len(all_cuts) - 1):
            cy0 = all_cuts[i]
            cy1 = all_cuts[i + 1]
            cropped = sub_img.crop((0, cy0, w, cy1))
            bbox = (x0_base, y0_base + cy0, x1_base, y0_base + cy1)
            result.append((cropped, bbox))

        logger.info(f"    内部模式切分: {w}x{h} → {len(result)} 段, "
                    f"切分点 y={cut_points}")
        return result

    # ==================== Step 3: 垂直切分 ====================

    def _split_vertical(self, region_tuple: Tuple[Image.Image, Tuple[int, int, int, int]]) \
            -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        按垂直空白带切分区域（检测左右并排的表格）。

        保守策略：
        - 对高区域（表格）：子区域宽度至少占原始宽度 20%（避免把标签列切出来）
        - 对矮区域（文字条）：放宽到 5%（允许切分短文字+长内容的情况）
        - 空白带宽度至少占图宽 3% 或 30px（表格内部列间距通常 < 2%）
        - 最多切 4 段（左右并排超过 4 个表格极为罕见）
        """
        sub_img, local_bbox = region_tuple
        gray = np.array(sub_img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 对矮区域（文字条，h < 200）放宽最小子区域宽度
        if h < 200:
            min_sub_width = self.min_region_width  # 仅用绝对最小值 20px
        else:
            min_sub_width = max(self.min_region_width, int(w * 0.20))

        if w < min_sub_width * 2:
            return [region_tuple]

        # 每列的非白像素数
        col_content = (gray < self.white_threshold).sum(axis=0)

        # 找空白列（允许少量噪声）
        is_blank_col = col_content <= 3

        # 找连续空白带
        blank_bands = self._find_continuous_bands(is_blank_col, self.min_blank_band_width)

        if not blank_bands:
            return [region_tuple]

        # 计算空白带宽度
        band_widths = [end - start for start, end in blank_bands]

        # 自适应选择切分点
        cut_bands = self._select_vertical_cut_bands(blank_bands, band_widths, w)

        if not cut_bands:
            return [region_tuple]

        # 按 cut_bands 切分，并验证每个子区域的宽度
        x0_base, y0_base, x1_base, y1_base = local_bbox
        cut_points = [0]
        for start, end in cut_bands:
            cut_points.append((start + end) // 2)
        cut_points.append(w)

        # 判断是否有"强分隔"空白带（宽度 >= 中位数 × 10）
        # 强分隔无条件切分，不受最小宽度限制
        all_widths = np.array(band_widths)
        median_w = float(np.median(all_widths))
        has_strong_separator = any(bw >= median_w * 10 and bw >= 50
                                   for bw in [e - s for s, e in cut_bands])

        if has_strong_separator:
            # 即使是强分隔，也要拒绝切出"细长竖条"（高宽比 > 10）——
            # 这种子区域几乎总是表格内的稀疏列（如全零列），不是独立表格。
            for i in range(len(cut_points) - 1):
                seg_width = cut_points[i + 1] - cut_points[i]
                if seg_width > 0 and h / seg_width > 10:
                    return [region_tuple]
            # 【Case 7 防护】对"矮而极宽"（w/h > 20）的水平条，拒绝切出
            # "实际内容跨度过窄"（< 100px）的孤立小段——这几乎总是稀疏跨全宽
            # header 行（左端有小段文字 + 中间大空白 + 右端一小字），
            # 不是独立表格/文字块。
            # 见 185a2337：整宽 5378×194 header 行的右段裸宽 1372px 但内容只有
            # 21px（后续 trim 后变成 21×22 的孤立小 text）。
            # 用 col_content 判断段内实际内容跨度，比裸段宽更精准。
            if w > 20 * h:
                for i in range(len(cut_points) - 1):
                    x0, x1 = cut_points[i], cut_points[i + 1]
                    seg_col_content = col_content[x0:x1]
                    seg_has_content = seg_col_content > 3
                    content_cols = np.where(seg_has_content)[0]
                    if len(content_cols) == 0:
                        # 全空段 → 拒绝切（切分点选择本身有问题）
                        return [region_tuple]
                    seg_content_width = int(content_cols[-1] - content_cols[0] + 1)
                    if seg_content_width < 100:
                        return [region_tuple]
        else:
            # 非强分隔：验证所有子区域宽度 >= min_sub_width
            valid_cut = True
            for i in range(len(cut_points) - 1):
                seg_width = cut_points[i + 1] - cut_points[i]
                if seg_width < min_sub_width:
                    valid_cut = False
                    break

            if not valid_cut:
                return [region_tuple]

        # 最多切 4 段
        if len(cut_points) - 1 > 4:
            return [region_tuple]

        regions = []
        for i in range(len(cut_points) - 1):
            x0 = cut_points[i]
            x1 = cut_points[i + 1]
            cropped = sub_img.crop((x0, 0, x1, h))
            bbox = (x0_base + x0, y0_base, x0_base + x1, y1_base)
            regions.append((cropped, bbox))

        return regions if regions else [region_tuple]

    def _select_vertical_cut_bands(self, blank_bands, band_widths, total_width):
        """
        选择垂直方向的切分空白带。

        垂直切分更保守：
        - 空白带宽度至少 3% 图宽或 30px（表格列间距通常 5-15px，表间间距 > 30px）
        - 或者空白带宽度明显大于其他空白（表格间距 vs 列间距）
        """
        if not blank_bands:
            return []

        n = len(blank_bands)
        widths = np.array(band_widths)
        median_width = float(np.median(widths))
        max_width = float(np.max(widths))

        # 最小切分宽度：3% 图宽或 30px
        min_cut_width = max(30, int(total_width * 0.03))

        # 少量空白带（<= 3 个）→ 宽度够大就切
        if n <= 3:
            return [(s, e) for (s, e), w in zip(blank_bands, band_widths) if w >= min_cut_width]

        # 多量空白带：自适应
        if max_width < median_width * self.adaptive_gap_ratio:
            # 宽度均匀 → 可能是表格列间距，不切
            return []

        # 宽度分布有两个层级：只在超过阈值的空白带处切分
        threshold = max(min_cut_width, median_width * self.adaptive_gap_ratio)
        cut_bands = [(s, e) for (s, e), w in zip(blank_bands, band_widths) if w >= threshold]
        return cut_bands

    # ==================== 辅助方法 ====================

    def _merge_wide_text_into_table(self, regions: List[Region],
                                     orig_img: Image.Image) -> List[Region]:
        """
        合并紧邻 table 顶部的宽 text 到 table（跨列合并单元格表头行）。

        合并条件：
        - text 紧邻 table 上方（y_gap <= 20px）
        - text 宽度 >= table 宽度 × 85%
        - text 和 table 的 x 范围重叠 >= table 宽度 × 85%

        对每个 table 可能循环合并多个 text（比如有连续多行合并表头）。

        Args:
            regions: Level 1 分割结果
            orig_img: 原始图片（用于重新 crop 合并后的 table 区域）
        """
        if len(regions) < 2:
            return regions

        merged_indices = set()  # 已被合并的 text 索引

        for i, r in enumerate(regions):
            if r.region_type != 'table' or i in merged_indices:
                continue

            # 循环找 table 上方紧邻的宽 text，可能有多行
            changed = True
            while changed:
                changed = False
                r_w = r.bbox[2] - r.bbox[0]
                best_j = None
                best_gap = None
                for j, other in enumerate(regions):
                    if j == i or j in merged_indices:
                        continue
                    if other.region_type != 'text':
                        continue
                    # no_merge 标记：竖线检测剥离的文字不合并回来
                    if getattr(other, 'no_merge', False):
                        continue
                    # text 必须在当前 table 顶部之上
                    if other.bbox[3] > r.bbox[1] + 5:
                        continue
                    gap = r.bbox[1] - other.bbox[3]
                    if gap > 20:
                        continue
                    # 高度检查：极薄的文字条（< 15px）不合并回去
                    # 这些通常是有边框表格顶部的metadata文字（如"女性 3年交"），
                    # 不是跨列合并表头（如"被保险人"，通常 30+px 高）
                    other_h = other.bbox[3] - other.bbox[1]
                    if other_h < 15:
                        continue
                    # 边框检查：如果 table 顶部有明显的横线（表格边框），
                    # 说明 text 是在边框外面，不应该合并回去
                    if self._has_top_border_line(r.image):
                        continue
                    # 宽度检查
                    other_w = other.bbox[2] - other.bbox[0]
                    if other_w < r_w * 0.85:
                        continue
                    # x 重叠检查
                    ox0 = max(r.bbox[0], other.bbox[0])
                    ox1 = min(r.bbox[2], other.bbox[2])
                    if ox1 - ox0 < r_w * 0.85:
                        continue
                    # 选最紧邻的
                    if best_j is None or gap < best_gap:
                        best_j = j
                        best_gap = gap

                if best_j is not None:
                    other = regions[best_j]
                    new_bbox = (min(r.bbox[0], other.bbox[0]),
                                min(r.bbox[1], other.bbox[1]),
                                max(r.bbox[2], other.bbox[2]),
                                r.bbox[3])
                    # 从原图重新 crop
                    new_img = orig_img.crop(new_bbox)
                    r.image = new_img
                    r.bbox = new_bbox
                    merged_indices.add(best_j)
                    changed = True
                    logger.info(f"    合并宽text→table: text {other.bbox} + "
                                f"table → 新 table {new_bbox}")

        return [regions[i] for i in range(len(regions)) if i not in merged_indices]

    def _has_top_border_line(self, img: Image.Image) -> bool:
        """
        检查图片顶部是否有明显的横线（表格边框）。

        在前 10 行内找密度 > 图宽 80% 的行（横线特征）。
        如果找到，说明这是个有边框的表格，顶部边框线上方的内容不应合并回来。
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
        if h < 5 or w < 50:
            return False
        content_mask = gray < self.white_threshold
        row_content = content_mask.sum(axis=1)
        # 在前 10 行内找横线
        check_rows = min(10, h)
        for y in range(check_rows):
            if row_content[y] > w * 0.8:
                return True
        return False

    def _merge_side_text_into_table(self, regions: List[Region],
                                     orig_img: Image.Image) -> List[Region]:
        """
        合并紧邻 table 左侧/右侧的细长竖条 text 到 table。

        典型场景：表格内的稀疏列（如全"0"列）被 Level 1 的垂直切分误切成独立 text。

        合并条件：
        - text 是**细长竖条**（h/w > 10）
        - text 和 table 的 y 范围**显著重叠**（>= 较小高度的 80%）
        - text 和 table 在 x 方向**相邻**（gap <= max(100, table宽度 × 5%)）

        Args:
            regions: Level 1 分割结果
            orig_img: 原始图片
        """
        if len(regions) < 2:
            return regions

        merged_indices = set()

        for i, r in enumerate(regions):
            if r.region_type != 'table' or i in merged_indices:
                continue

            # 循环找左右相邻的细长竖条
            changed = True
            while changed:
                changed = False
                r_w = r.bbox[2] - r.bbox[0]
                r_h = r.bbox[3] - r.bbox[1]

                best_j = None
                best_gap = None
                for j, other in enumerate(regions):
                    if j == i or j in merged_indices:
                        continue
                    if other.region_type != 'text':
                        continue

                    other_w = other.bbox[2] - other.bbox[0]
                    other_h = other.bbox[3] - other.bbox[1]

                    # 必须是细长竖条
                    if other_w == 0 or other_h / other_w < 10:
                        continue

                    # y 范围显著重叠
                    oy0 = max(r.bbox[1], other.bbox[1])
                    oy1 = min(r.bbox[3], other.bbox[3])
                    overlap = max(0, oy1 - oy0)
                    min_h = min(r_h, other_h)
                    if min_h == 0 or overlap < min_h * 0.8:
                        continue

                    # x 相邻（左侧或右侧）
                    if other.bbox[2] <= r.bbox[0]:
                        # text 在 table 左侧
                        gap = r.bbox[0] - other.bbox[2]
                    elif other.bbox[0] >= r.bbox[2]:
                        # text 在 table 右侧
                        gap = other.bbox[0] - r.bbox[2]
                    else:
                        continue  # x 有重叠，不是相邻

                    max_gap = max(100, int(r_w * 0.05))
                    if gap > max_gap:
                        continue

                    if best_j is None or gap < best_gap:
                        best_j = j
                        best_gap = gap

                if best_j is not None:
                    other = regions[best_j]
                    new_bbox = (min(r.bbox[0], other.bbox[0]),
                                min(r.bbox[1], other.bbox[1]),
                                max(r.bbox[2], other.bbox[2]),
                                max(r.bbox[3], other.bbox[3]))
                    new_img = orig_img.crop(new_bbox)
                    r.image = new_img
                    r.bbox = new_bbox
                    merged_indices.add(best_j)
                    changed = True
                    logger.info(f"    合并侧竖条→table: text {other.bbox} → "
                                f"新 table {new_bbox}")

        return [regions[i] for i in range(len(regions)) if i not in merged_indices]

    def _merge_adjacent_tables(self, regions: List[Region],
                                 orig_img: Image.Image) -> List[Region]:
        """
        合并上下相邻、宽度相同、中间无 region 阻挡的 table。

        典型场景 (945e8fe9)：多层 header（4493×109）+ 数据主体（4493×2377）
        被 Step 1 中间的 19 行空白带误切成两个独立 table，应合并成一个整表。

        合并条件（全部满足才合并）：
        1. 两者都是 'table' 类型
        2. **严格上下关系**：A.bottom ≤ B.top（y 无重叠）→ 排除左右并排
        3. gap = B.top - A.bottom ≤ 40 px（紧邻）
        4. 宽度差异 ≤ 图宽 × 2%（宽度相同）
        5. 左右边界差异 ≤ 图宽 × 2%（左右对齐）
        6. **A 和 B 之间无其他 region 挡在中间**
           （any region 的 y 完全在 [A.bottom, B.top] 内 且 x 与合并区间重叠）

        对比 Step 5b/5c（text→table）：本方法处理 table→table 合并。

        Args:
            regions: 前面所有 step 后的 region 列表
            orig_img: 原图（用于 crop 合并后的大区域）
        """
        if len(regions) < 2:
            return regions

        W_img = orig_img.size[0]
        tol = max(20, int(W_img * 0.02))  # 宽度/对齐容忍度
        max_gap = 40  # 上下 gap 上限

        # 循环合并直到无变化（一次合并可能引发级联合并）
        changed = True
        while changed:
            changed = False
            merge_ij = None  # (i, j) 待合并

            n = len(regions)
            for i in range(n):
                ra = regions[i]
                if ra.region_type != 'table':
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    rb = regions[j]
                    if rb.region_type != 'table':
                        continue

                    # 条件 2: A 严格在 B 上方（y 无重叠）
                    if ra.bbox[3] > rb.bbox[1]:
                        continue

                    # 条件 3: gap 合理
                    gap = rb.bbox[1] - ra.bbox[3]
                    if gap < 0 or gap > max_gap:
                        continue

                    # 条件 4: 宽度相同
                    w_a = ra.bbox[2] - ra.bbox[0]
                    w_b = rb.bbox[2] - rb.bbox[0]
                    if abs(w_a - w_b) > tol:
                        continue

                    # 条件 5: 左右对齐
                    if abs(ra.bbox[0] - rb.bbox[0]) > tol:
                        continue
                    if abs(ra.bbox[2] - rb.bbox[2]) > tol:
                        continue

                    # 条件 6: 中间无 region 阻挡
                    # 合并区间的 x 范围
                    mx0 = min(ra.bbox[0], rb.bbox[0])
                    mx1 = max(ra.bbox[2], rb.bbox[2])
                    blocked = False
                    for k in range(n):
                        if k == i or k == j:
                            continue
                        other = regions[k]
                        # y 是否完全在 A 和 B 之间
                        if other.bbox[1] < ra.bbox[3] or other.bbox[3] > rb.bbox[1]:
                            continue
                        # x 是否与合并区间重叠
                        ox0 = max(mx0, other.bbox[0])
                        ox1 = min(mx1, other.bbox[2])
                        if ox1 > ox0:
                            blocked = True
                            break
                    if blocked:
                        continue

                    # 所有条件满足，标记合并
                    merge_ij = (i, j)
                    break
                if merge_ij is not None:
                    break

            if merge_ij is not None:
                i, j = merge_ij
                ra, rb = regions[i], regions[j]
                new_bbox = (
                    min(ra.bbox[0], rb.bbox[0]),
                    ra.bbox[1],
                    max(ra.bbox[2], rb.bbox[2]),
                    rb.bbox[3],
                )
                new_img = orig_img.crop(new_bbox)
                w_a = ra.bbox[2] - ra.bbox[0]
                h_a = ra.bbox[3] - ra.bbox[1]
                w_b = rb.bbox[2] - rb.bbox[0]
                h_b = rb.bbox[3] - rb.bbox[1]
                logger.info(f"    合并相邻 table: A({w_a}x{h_a}, bbox={ra.bbox}) "
                            f"+ B({w_b}x{h_b}, bbox={rb.bbox}) "
                            f"→ 合并 {new_img.size} bbox={new_bbox}")
                # 用合并结果替换 A，删除 B
                ra.image = new_img
                ra.bbox = new_bbox
                regions = [r for k, r in enumerate(regions) if k != j]
                changed = True

        return regions

    def _strip_top_text_from_table(self, region: Region) -> List[Region]:
        """
        检查表格区域顶部是否有独立的说明文字，如果有则切出去。

        判断依据：
        - 有边框表格：竖线的最高点 y_top 就是表格主体的顶部；y_top 之上的部分
          （如果存在且内容稀疏）是独立说明文字，切出去。
        - 无边框表格：找不到竖线，用密度跳变分析代替。

        Returns:
            [text_region, table_region] 或 [region]（未切分）
        """
        img = region.image
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        # 太小的表格不做处理
        if h < 100:
            return [region]

        # ==================== Layer 0: 外框顶部横线检测（最优先） ====================
        # 有边框表格顶部通常是一条 >= 80% 图宽的横线，是最精确的表格上边界标志。
        # 命中时**强制**从横线切分，保护表格首行（如跨行合并 "保单年度末"、
        # 稀疏 header 行）不被误剥离切碎（见 Case 1）。
        y_hline = self._detect_top_horizontal_border(gray, h, w)
        if y_hline is not None:
            if y_hline < 10:
                # 表格几乎从图顶就开始，上方没有独立说明文字 → 不切
                return [region]
            # 【首行无边框防护】上方是"表格数据行"时不剥离
            # 场景：表格首行数据本身上方无横线，Layer 0 会把"列 header 下横线"
            # 当成外框顶，误将首行剥离成薄 text 条（见 d1752e16 的 6821×21）。
            # 判据：上方内容 覆盖 ≥70% 图宽 且 wide_gaps ≥4 → 多列结构 = 表格数据行
            if self._is_table_row_above_border(gray, y_hline, w):
                # 【进一步检查】上方可能同时包含"独立说明文字 + 表格首行"
                # 见 58b3cb9e：红色说明文字 y=0-30 + 空白 + 列头 y=45-75 + 横线 y=75
                # 尝试在 [0, y_hline) 内找"文字段/表格行分界"作为新的切分点
                alt_y = self._find_text_split_above_table_row(gray, y_hline, w)
                if alt_y is None:
                    logger.info(f"    Layer 0 检测到横线 y={y_hline}，"
                                f"但上方是表格数据行（跨全宽+多列），保持整表不剥离")
                    return [region]
                logger.info(f"    Layer 0 y_hline={y_hline} 上方是表格首行，"
                            f"但在其内部找到独立文字段/表格行分界 y={alt_y}，"
                            f"改用该分界切分")
                y_hline = alt_y
            # 从横线位置切：上方 = 独立说明文字，下方 = 完整表格（含首行）
            text_img = img.crop((0, 0, w, y_hline))
            text_img_trimmed, (tsx, tsy) = self._trim_margins(text_img)
            # Case 6 防护：过滤过薄（< 15px）或几乎无内容的顶部片段，
            # 避免外框上沿的少量抗锯齿像素被剥离成空 text region。
            if text_img_trimmed.size[1] < 15:
                return [region]
            top_gray_arr = np.array(text_img_trimmed.convert('L'), dtype=np.uint8)
            top_content_ratio = (top_gray_arr < self.white_threshold).mean()
            if top_content_ratio < 0.01:
                return [region]

            table_img_new = img.crop((0, y_hline, w, h))
            rx0, ry0, rx1, ry1 = region.bbox
            text_bbox = (rx0 + tsx, ry0 + tsy,
                         rx0 + tsx + text_img_trimmed.size[0],
                         ry0 + tsy + text_img_trimmed.size[1])
            table_bbox = (rx0, ry0 + y_hline, rx1, ry1)

            logger.info(f"    顶部文字剥离(外框横线): 表格{w}x{h} → "
                        f"文字{text_img_trimmed.size} + 表格{table_img_new.size} "
                        f"(y_hline={y_hline})")

            text_region = Region(text_img_trimmed, text_bbox, 'text')
            # 外框横线剥离出的 text 位于表格边框之外，不应被 Step 5b 合并回 table
            text_region.no_merge = True
            table_region = Region(table_img_new, table_bbox, 'table')
            return [text_region, table_region]

        # ==================== Layer 1-3: 原有的竖线 / 密度跳变 / 文字段检测 ====================
        # 1. 尝试用竖线检测找 y_top
        y_top = self._detect_vertical_lines_top_y(gray, h, w)
        skip_density_check = False
        from_vertical_lines = False  # 是否来自竖线检测（精确边框位置）

        if y_top is not None:
            from_vertical_lines = True
        else:
            # 无边框表格：用密度跳变分析
            y_top = self._detect_top_by_density_jump(gray, h, w)
            # Layer 2 命中后做 Layer 3 二次验证：如果水平位置分析也返回接近的
            # 切分点（±5px 内），说明该点既有密度证据又有位置证据，非常可信。
            # 此时后续走宽松阈值（min_y_top=15、跳过密度对比检查），让顶部单行
            # 小字（如 baf9a56f 的"女性"）能被剥离。
            if y_top is not None:
                y_top_v3 = self._detect_top_by_text_segments(gray, h, w)
                if y_top_v3 is not None and abs(y_top - y_top_v3) <= 5:
                    skip_density_check = True

        if y_top is None:
            # 密度跳变也失败：用"顶部左对齐说明文字段"检测。
            # 该方法基于水平位置分析，已经足够可靠，无需再做密度对比检查。
            y_top = self._detect_top_by_text_segments(gray, h, w)
            if y_top is None:
                return [region]
            skip_density_check = True

        # 2. 判断顶部空间是否合理
        # 竖线检测给出精确的表格边框位置：即使只有几行文字也应该切出来
        # Layer 3（文字段水平位置检测）也很精确：判据是"左对齐段 + 下方跨全宽段"，
        # 放宽到 15px，让顶部单行小字（如 baf9a56f 的"女性"）能被剥离。
        # Layer 2（密度跳变）判据较弱，保持 30px 更保守。
        if from_vertical_lines:
            min_y_top = 5
        elif skip_density_check:  # Layer 3 命中
            min_y_top = 15
        else:  # Layer 2（密度跳变）命中
            min_y_top = 30
        if y_top < min_y_top or y_top > h * 0.30:
            return [region]

        # 3. 分析上方 [0, y_top] 是否为稀疏文字
        top_gray = gray[:y_top]
        top_content = (top_gray < self.white_threshold).sum(axis=1)
        # 忽略横线行（密度 > 图宽 90%），它们是表格边框而非文字
        text_content = top_content[top_content < w * 0.9]
        if len(text_content) == 0:
            return [region]

        text_content_positive = text_content[text_content > 5]
        if len(text_content_positive) == 0:
            return [region]

        # 4. 密度对比：上方文字最大密度 vs 表格前段的中位密度
        table_sample_h = min(h - y_top, 500)
        table_gray = gray[y_top:y_top + table_sample_h]
        table_content = (table_gray < self.white_threshold).sum(axis=1)
        # 表格采样时同样忽略横线行
        table_content_positive = table_content[
            (table_content > 20) & (table_content < w * 0.9)
        ]
        if len(table_content_positive) == 0:
            return [region]

        table_median = float(np.median(table_content_positive))
        top_max = float(text_content_positive.max())

        # 上方最大密度 >= 表格中位密度 × 0.5 → 上方也很密集，保守不切
        # 若切分点来自"文字段水平位置"检测 或 竖线检测（精确边框位置），跳过此检查
        if not skip_density_check and not from_vertical_lines and top_max >= table_median * 0.5:
            logger.info(f"    顶部文字剥离: 上方密度过高(top_max={top_max:.0f} vs "
                        f"table_median={table_median:.0f})，保守不切")
            return [region]

        # 5. 满足条件，切分
        text_img = img.crop((0, 0, w, y_top))
        text_img_trimmed, (tsx, tsy) = self._trim_margins(text_img)
        # 剥离出的顶部 text 必须有实际内容：
        # - 高度 >= 15px（原来竖线路径下 min_h=5 会漏出 11px 的空片段，见 Case 6）
        # - content_ratio >= 1%（防止外框上沿抗锯齿噪声、稀疏空白被误剥离）
        min_h = 15 if from_vertical_lines else self.min_region_height
        if text_img_trimmed.size[1] < min_h:
            return [region]
        top_gray_arr = np.array(text_img_trimmed.convert('L'), dtype=np.uint8)
        if (top_gray_arr < self.white_threshold).mean() < 0.01:
            return [region]

        # table 直接从 y_top 开始（切分点已经在空白带下端 / 表格边框顶部），
        # 不再加 trim_padding，避免把 text 的下沿拉入 table
        table_y0 = y_top
        table_img_new = img.crop((0, table_y0, w, h))

        rx0, ry0, rx1, ry1 = region.bbox
        text_bbox = (rx0 + tsx, ry0 + tsy,
                     rx0 + tsx + text_img_trimmed.size[0],
                     ry0 + tsy + text_img_trimmed.size[1])
        table_bbox = (rx0, ry0 + table_y0, rx1, ry1)

        logger.info(f"    顶部文字剥离: 表格{w}x{h} → 文字{text_img_trimmed.size} + "
                    f"表格{table_img_new.size} (y_top={y_top}, "
                    f"密度比{top_max/table_median:.2f})")

        text_region = Region(text_img_trimmed, text_bbox, 'text')
        # 竖线检测剥离的文字不应被 Step 5b 合并回 table
        # （它在表格边框外面，不是跨列合并表头）
        if from_vertical_lines:
            text_region.no_merge = True
        table_region = Region(table_img_new, table_bbox, 'table')
        return [text_region, table_region]

    def _strip_left_label_from_table(self, region: Region) -> List[Region]:
        """
        检查表格左侧是否有窄标签列 + 大空白，如果有则切出左侧标签为 text。

        典型场景 (0f372a06 table_04)：
        - x=10-89: "投保年龄" 4个字（79px 宽）
        - x=89-667: 578px 的连续空白竖条
        - x=667-677: 行号列（年龄数字如"43","45"...）只有 9 行内容
        - x=797+: 数据列（所有行都有内容）

        判据：
        1. 左侧存在 >= 100px 的连续空白竖条（col_content <= 3）
        2. 空白竖条左侧的内容区域宽度 < 图宽的 15%
        3. 空白竖条左侧内容只出现在少数行（"投保年龄"只在1行，
           而行号列在 ~9 行）→ 通过逐列分析区分

        切分点选择：
        - 不用固定 margin（之前 -15px 会切到行号数字上）
        - 逐列扫描，找第一个有**多行内容**（>= max(3, 5%*h) 行）的 x
        - 该 x 就是表格数据的真实起始位置

        Returns:
            [text_region, table_region] 或 [region]（不剥离时）
        """
        img = region.image
        w, h = img.size
        if w < 500 or h < 100:
            return [region]

        gray = np.array(img.convert('L'), dtype=np.uint8)
        col_content = (gray < self.white_threshold).sum(axis=0)

        # 找左侧第一个 >= 100px 的连续空白竖条
        in_blank = False
        blank_start = 0
        left_blank = None
        for x in range(min(w // 2, 800)):  # 只在左半边找
            if col_content[x] <= 3:
                if not in_blank:
                    blank_start = x
                    in_blank = True
            else:
                if in_blank:
                    if x - blank_start >= 100:
                        left_blank = (blank_start, x)
                        break
                    in_blank = False
        if in_blank and min(w // 2, 800) - blank_start >= 100:
            left_blank = (blank_start, min(w // 2, 800))

        if left_blank is None:
            return [region]

        blank_x0, blank_x1 = left_blank
        label_width = blank_x0  # 空白左侧的内容宽度

        # 判据 2: 内容宽度 < 图宽 15%
        if label_width >= w * 0.15:
            return [region]

        # 判据 3: 空白带左侧的内容只出现在少数行
        # （"投保年龄"只在 1 行出现，行号列在 ~9 行出现）
        left_region = gray[:, :blank_x0]
        row_has_content = (left_region < self.white_threshold).sum(axis=1) > 3
        content_row_count = int(row_has_content.sum())
        content_ratio = content_row_count / h
        if content_ratio > 0.15:
            return [region]  # 超过 15% 的行有内容，可能是正常列

        # 执行剥离
        # 左侧标签区域
        content_rows = np.where(row_has_content)[0]
        if len(content_rows) == 0:
            return [region]
        label_y0 = max(0, int(content_rows[0]) - 3)
        label_y1 = min(h, int(content_rows[-1]) + 4)
        label_img = img.crop((0, label_y0, blank_x0, label_y1))

        # 右侧表格区域：用逐列分析找到表格数据的真实起始位置
        # 关键改进：不用 blank_x1 作为起点（blank band 可能因零星像素延伸很远）
        # 而是从 blank_x0 右侧开始，逐列扫描找第一个有**多行内容**的 x
        # "多行"定义：>= max(3, h*1%) 行有内容
        # - "投保年龄"只在 1 行 → 不算多行 → 被裁掉
        # - 行号列在 ~9 行 → 算多行 → 被保留
        # - 数据列在所有行 → 算多行 → 被保留
        min_multi_row = max(3, int(h * 0.01))

        table_start_x = blank_x1  # fallback：空白带结束位置
        for x in range(blank_x0 + 1, min(w, blank_x0 + 800)):
            # 该列有多少行有内容
            col_rows_with_content = int((gray[:, x] < self.white_threshold).sum())
            if col_rows_with_content >= min_multi_row:
                # 找到表格数据起始，左边留 5px 安全空白
                table_start_x = max(blank_x0 + 1, x - 5)
                break

        logger.info(f"    左侧标签剥离: 空白带 x=[{blank_x0},{blank_x1}], "
                     f"表格起始 x={table_start_x}, "
                     f"min_multi_row={min_multi_row}")

        table_img = img.crop((table_start_x, 0, w, h))

        x0, y0, x1, y1 = region.bbox
        text_region = Region(
            image=label_img,
            bbox=(x0, y0 + label_y0, x0 + blank_x0, y0 + label_y1),
            region_type='text'
        )
        table_region = Region(
            image=table_img,
            bbox=(x0 + table_start_x, y0, x1, y1),
            region_type='table'
        )

        lw, lh = label_img.size
        tw, th = table_img.size
        logger.info(f"    左侧标签剥离: 表格{w}x{h} → "
                    f"标签({lw}x{lh}) + 表格({tw}x{th}) "
                    f"(空白带 x=[{blank_x0},{blank_x1}])")
        return [text_region, table_region]

    def _strip_top_by_span_jump(self, region: Region) -> List[Region]:
        """
        按内容行跨度跳跃检测表格顶部 metadata 并剥离。

        算法：
        1. 按空白行将表格分割为"内容行"段
        2. 计算每个内容行的黑色像素跨度（最右-最左）/ 图宽
        3. 找第一个"<70% → >=70%"的跨度跳跃
        4. 在跳跃处的空白带切分（前面是 metadata，后面是数据表）
        5. 只在前 15% 高度内搜索

        典型场景：
        - d8b59365: 居中标题(19%) + 元数据(62%) → 列头(94%)，跳跃在行[2]→[3]
        - 3792d522: "男性"(1%) → 数据(100%)，跳跃在行[0]→[1]
        - 88684b6b: "投保年龄"(1%) → 数据(99%)，跳跃在行[1]→[2]

        安全性：
        - 阶梯型表格(1829aea8)第一行 span=100%>=70%，不会触发
        - 只在表格前 15% 高度搜索

        Returns:
            [text_region, table_region] 或 [region]
        """
        img = region.image
        w, h = img.size
        if h < 200:
            return [region]

        gray = np.array(img.convert('L'), dtype=np.uint8)
        search_limit = min(int(h * 0.15), 300)

        # 按空白行分割为内容行段
        row_content = (gray < self.white_threshold).sum(axis=1)
        in_content = False
        content_rows = []
        seg_start = 0
        for y in range(search_limit):
            if row_content[y] > 3:
                if not in_content:
                    seg_start = y
                    in_content = True
            else:
                if in_content:
                    content_rows.append((seg_start, y))
                    in_content = False
        if in_content:
            content_rows.append((seg_start, search_limit))

        if len(content_rows) < 2:
            return [region]

        # 计算每个内容行的跨度
        spans = []
        for ys, ye in content_rows:
            seg = gray[ys:ye, :]
            cols_with_content = np.where((seg < self.white_threshold).sum(axis=0) > 0)[0]
            if len(cols_with_content) > 0:
                span = (int(cols_with_content[-1]) - int(cols_with_content[0])) / w
            else:
                span = 0
            spans.append(span)

        # 找第一个 <70% → >=70% 的跳跃
        for i in range(1, len(content_rows)):
            if spans[i - 1] < 0.70 and spans[i] >= 0.70:
                # 验证：低 span 的内容行高度必须 >= 5px（排除边框线）
                prev_height = content_rows[i - 1][1] - content_rows[i - 1][0]
                if prev_height < 5:
                    continue  # 太矮，可能是边框线，跳过这个跳跃点
                # 【夹心表头防护】跳跃点之前的所有内容行必须都是窄的（纯 metadata 前缀）。
                # 若更上方已出现"宽内容行"（span>=70% 且 高度>=5px，排除细边框线），
                # 说明表格内容从顶部就开始了，当前窄行只是夹在宽行之间的表头结构
                # （如 4c9e11b2：列号"0..69"行(97%,9px) + "总榜单"(1%) + 数据行(98%)），
                # 不应作为 metadata→table 跳跃点剥离。
                # 注意：必须排除细横线（高<5px，如 0cd74f08 的蓝色边框线），否则会把
                # "边框线 + 投保年龄标签" 误判为夹心表头，导致标签无法被剥离。
                has_wide_content_above = any(
                    spans[j] >= 0.70 and (content_rows[j][1] - content_rows[j][0]) >= 5
                    for j in range(i - 1)
                )
                if has_wide_content_above:
                    return [region]
                # 找到跳跃！切分点在两个内容行之间的空白带
                prev_end = content_rows[i - 1][1]
                cur_start = content_rows[i][0]
                y_cut = (prev_end + cur_start) // 2

                # 安全检查：切分点不能太深（前 15%）
                if y_cut > h * 0.15:
                    return [region]

                # 切分
                text_img = img.crop((0, 0, w, y_cut))
                text_img_trimmed, (tsx, tsy) = self._trim_margins(text_img)
                if text_img_trimmed.size[1] < 10:
                    return [region]

                table_img_new = img.crop((0, y_cut, w, h))
                rx0, ry0, rx1, ry1 = region.bbox
                text_bbox = (rx0 + tsx, ry0 + tsy,
                             rx0 + tsx + text_img_trimmed.size[0],
                             ry0 + tsy + text_img_trimmed.size[1])
                table_bbox = (rx0, ry0 + y_cut, rx1, ry1)

                logger.info(f"    顶部文字剥离(跨度跳跃): 表格{w}x{h} → "
                            f"文字{text_img_trimmed.size} + 表格{table_img_new.size} "
                            f"(y_cut={y_cut}, span {spans[i-1]*100:.0f}%→{spans[i]*100:.0f}%)")

                text_region = Region(text_img_trimmed, text_bbox, 'text')
                text_region.no_merge = True
                table_region = Region(table_img_new, table_bbox, 'table')
                return [text_region, table_region]

        return [region]

    def _strip_bottom_rebound_text(self, region: Region) -> List[Region]:
        """
        检测表格底部是否有"内容反弹段"（xe 局部反弹的独立文字），如果有则剥离。
    
        典型场景 (deb8de95 r00_table)：
        - 上方是表 A 金字塔尾部（xe 递减到 ~100）
        - 底部有 3 行左对齐文字（"男性 保单年度末"），xe 反弹到 ~1700
        - 该 3 行字是表格外的元数据标签，应作为独立 text 剥离
    
        算法：
        1. 从底部枚举窗口大小 [30, 50, 80, 100] 行
        2. 判据（严格保守）：
           - 底部窗口 avg xe ∈ [500, 图宽 × 60%]（比金字塔尾窄段宽，但不到 wide）
           - 底部段 avg xe / 上方相邻 100 行 avg xe ≥ 3（**关键：显著反弹**）
           - 底部段 avg xs < 图宽 20%（左对齐）
        3. 切分点 = 窗口起点，然后**吸附到附近的严格空白行**（避免切到文字上）
        """
        img = region.image
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape
    
        if h < 500:
            return [region]
    
        content_mask = gray < self.white_threshold
        row_content = content_mask.sum(axis=1)
        row_x_end = np.full(h, -1, dtype=np.int32)
        row_x_start = np.full(h, -1, dtype=np.int32)
        for y in range(h):
            cols = np.where(content_mask[y])[0]
            if len(cols) > 0:
                row_x_start[y] = int(cols[0])
                row_x_end[y] = int(cols[-1])
    
        # 从底部向上枚举窗口大小，找到"反弹段"
        for bottom_h in [30, 50, 80, 100]:
            if bottom_h >= h // 3:
                break
            bottom_start = h - bottom_h
            # 底部窗口的 avg xe（忽略 -1 的空行）
            bottom_xe = row_x_end[bottom_start:h]
            valid = bottom_xe[bottom_xe >= 0]
            if len(valid) < bottom_h * 0.5:
                continue
            bottom_avg_xe = float(valid.mean())
    
            # 底部 avg xe 必须在合理范围（不太窄，也不达 wide）
            if bottom_avg_xe < 500 or bottom_avg_xe >= w * 0.60:
                continue
    
            # 上方相邻 100 行的 avg xe（作为对比基线）
            upper_end = bottom_start
            upper_start = max(0, upper_end - 100)
            upper_xe = row_x_end[upper_start:upper_end]
            upper_valid = upper_xe[upper_xe >= 0]
            if len(upper_valid) < 30:
                continue
            upper_avg_xe = float(upper_valid.mean())
    
            # 关键约束：底部 avg xe / 上方 >= 3（显著反弹）
            if upper_avg_xe <= 0 or bottom_avg_xe / upper_avg_xe < 3:
                continue
    
            # 底部 avg xs < 图宽 20%（左对齐）
            bottom_xs = row_x_start[bottom_start:h]
            valid_xs = bottom_xs[bottom_xs >= 0]
            if len(valid_xs) == 0:
                continue
            bottom_avg_xs = float(valid_xs.mean())
            if bottom_avg_xs >= w * 0.20:
                continue
    
            # 满足所有条件，剥离
            # 切分点：窗口起点，然后向上吸附到严格空白行（避免切到 3 行字上部的文字）
            y_split_raw = bottom_start
            y_split = self._snap_cut_to_blank_row(y_split_raw, row_content, w,
                                                   search_up=20, search_down=3)
    
            table_img = img.crop((0, 0, w, y_split))
            text_img = img.crop((0, y_split, w, h))
            text_img_trimmed, (tsx, tsy) = self._trim_margins(text_img)
    
            if text_img_trimmed.size[1] < 15:
                continue
            text_gray = np.array(text_img_trimmed.convert('L'), dtype=np.uint8)
            if (text_gray < self.white_threshold).mean() < 0.01:
                continue
    
            rx0, ry0, rx1, ry1 = region.bbox
            table_bbox = (rx0, ry0, rx1, ry0 + y_split)
            text_bbox = (rx0 + tsx, ry0 + y_split + tsy,
                         rx0 + tsx + text_img_trimmed.size[0],
                         ry0 + y_split + tsy + text_img_trimmed.size[1])
    
            logger.info(f"    底部反弹段剥离: 表格{w}x{h} → 表格({w},{y_split}) + "
                        f"文字{text_img_trimmed.size} "
                        f"(y_split_raw={y_split_raw}→{y_split}, "
                        f"bottom_xe={bottom_avg_xe:.0f}, "
                        f"upper_xe={upper_avg_xe:.0f}, "
                        f"ratio={bottom_avg_xe / upper_avg_xe:.1f})")
    
            table_region = Region(table_img, table_bbox, 'table')
            text_region = Region(text_img_trimmed, text_bbox, 'text')
            # 底部反弹段位于表格边框之外，不应被 Step 5b 合并回 table
            text_region.no_merge = True
            return [table_region, text_region]
    
        return [region]

    def _strip_bottom_footnote(self, region: Region) -> List[Region]:
        """
        检测全宽有边框表格底部的独立注释行（footnote）并剥离。

        典型场景 (8e60d5f8)：
        - 表格主体全宽（xe ≈ 图宽），底部有完整边框横线
        - 底边框下方有一行左对齐窄文字（如"注：根据实际业务需要..."）
        - 注释 xe 远小于表格宽度（xe 骤降），应作为独立 text 剥离

        与 _strip_bottom_rebound_text 的区别：
        - rebound: 表格尾部窄 + 下方文字宽（xe 反弹，ratio>=3）
        - footnote: 表格全宽 + 下方文字窄（xe 骤降）

        算法：
        1. 在底部 search_h 范围内，从下往上找"窄左对齐文字段"
           （xe < w×0.4 且 xs < w×0.2）
        2. 确认该窄段上方紧邻的是全宽表格内容（xe > w×0.7，含边框横线）
        3. 在窄段顶部（xe 骤降处）切分，吸附到附近空白行

        Returns:
            [table_region, text_region] 或 [region]
        """
        img = region.image
        gray = np.array(img.convert('L'), dtype=np.uint8)
        h, w = gray.shape

        if h < 300:
            return [region]

        content_mask = gray < self.white_threshold
        row_content = content_mask.sum(axis=1)

        row_x_end = np.full(h, -1, dtype=np.int32)
        row_x_start = np.full(h, -1, dtype=np.int32)
        for y in range(h):
            cols = np.where(content_mask[y])[0]
            if len(cols) > 0:
                row_x_start[y] = int(cols[0])
                row_x_end[y] = int(cols[-1])

        # 只在底部 search_h 范围内搜索
        search_h = min(150, h // 4)
        bottom_start = h - search_h

        narrow_xe_th = w * 0.40   # 窄行：xe 小于图宽 40%
        left_xs_th = w * 0.20     # 左对齐：xs 小于图宽 20%
        wide_xe_th = w * 0.70     # 全宽行：xe 大于图宽 70%

        # 状态机从底部往上扫描：跳过底部空白 → 收集窄行 → 遇宽行停止
        footnote_rows = []
        phase = 'skip_blank'
        for y in range(h - 1, bottom_start - 1, -1):
            xe = row_x_end[y]
            xs = row_x_start[y]
            is_blank = (xe < 0)
            is_narrow_left = (0 <= xs < left_xs_th and 0 <= xe < narrow_xe_th)

            if phase == 'skip_blank':
                if is_blank:
                    continue
                elif is_narrow_left:
                    footnote_rows.append(y)
                    phase = 'collect_narrow'
                else:
                    # 底部直接是全宽表格行，没有 footnote
                    return [region]
            else:  # collect_narrow
                if is_narrow_left:
                    footnote_rows.append(y)
                elif is_blank:
                    continue  # footnote 与表格间的间隙，继续往上探测
                else:
                    break  # 遇到全宽表格行，footnote 结束

        if not footnote_rows:
            return [region]

        footnote_top = min(footnote_rows)
        footnote_bottom = max(footnote_rows)

        # footnote 垂直跨度至少 8px（排除表格底边抗锯齿噪声，如 afa837b9 仅 4px）
        if footnote_bottom - footnote_top < 8:
            return [region]

        # footnote 至少要有 3 行有效内容（排除单行抗锯齿噪声）
        footnote_content_rows = [y for y in range(footnote_top, footnote_bottom + 1)
                                 if 0 <= row_x_end[y] < narrow_xe_th]
        if len(footnote_content_rows) < 3:
            return [region]

        # 确认 footnote 上方紧邻全宽表格内容（至少 3 行 xe > w×0.7）
        upper_check_start = max(0, footnote_top - 40)
        upper_xe = row_x_end[upper_check_start:footnote_top]
        upper_wide_count = int((upper_xe > wide_xe_th).sum())
        if upper_wide_count < 3:
            # 上方不是全宽表格，不构成"表格+注释"结构，不剥离
            return [region]

        # 切分点：footnote_top（xe 骤降处），吸附到附近空白行
        y_split = self._snap_cut_to_blank_row(footnote_top, row_content, w,
                                              search_up=8, search_down=2)
        if y_split <= upper_check_start or y_split >= h - 5:
            y_split = footnote_top

        table_img = img.crop((0, 0, w, y_split))
        text_img = img.crop((0, y_split, w, h))
        text_img_trimmed, (tsx, tsy) = self._trim_margins(text_img)

        # 切出的文字至少 16px 高（一行正常文字约 20-30px，排除抗锯齿薄条）
        if text_img_trimmed.size[1] < 16:
            return [region]
        text_gray = np.array(text_img_trimmed.convert('L'), dtype=np.uint8)
        if (text_gray < self.white_threshold).mean() < 0.005:
            return [region]

        rx0, ry0, rx1, ry1 = region.bbox
        table_bbox = (rx0, ry0, rx1, ry0 + y_split)
        text_bbox = (rx0 + tsx, ry0 + y_split + tsy,
                     rx0 + tsx + text_img_trimmed.size[0],
                     ry0 + y_split + tsy + text_img_trimmed.size[1])

        logger.info(f"    底部注释剥离: 表格{w}x{h} → 表格({w},{y_split}) + "
                    f"文字{text_img_trimmed.size} "
                    f"(footnote y={footnote_top}-{footnote_bottom})")

        table_region = Region(table_img, table_bbox, 'table')
        text_region = Region(text_img_trimmed, text_bbox, 'text')
        # 底部注释位于表格边框之外，不应被 Step 5b 合并回 table
        text_region.no_merge = True
        return [table_region, text_region]

    def _detect_top_by_density_jump(self, gray: np.ndarray, h: int, w: int) -> Optional[int]:
        """
        无边框表格的顶部检测：找密度跳变点。

        策略：从上往下扫描，找到第一个"高密度行"的位置，然后往上退到最近的空白带
        中心，避免切分点落在文字中间。
        - 高密度：>= 图宽 5%
        - 连续多行高密度才算表格开始
        - 只在上部 30% 内检测

        Returns:
            int: 切分 y 坐标（落在空白带内）；None 表示未找到
        """
        max_search_y = int(h * 0.30)
        row_content = (gray < self.white_threshold).sum(axis=1)

        high_density_threshold = int(w * 0.05)
        min_consecutive = 5

        # 从上往下扫描，找第一个连续高密度段的起点
        consecutive = 0
        table_start_y = None
        for y in range(max_search_y):
            # 忽略横线（>90% 图宽）
            if row_content[y] > w * 0.9:
                continue
            if row_content[y] >= high_density_threshold:
                consecutive += 1
                if consecutive >= min_consecutive:
                    table_start_y = y - min_consecutive + 1
                    break
            else:
                consecutive = 0

        if table_start_y is None:
            return None

        # 上方必须有稀疏文字才切
        top_content = row_content[:table_start_y]
        sparse_rows = np.where((top_content > 5) & (top_content < high_density_threshold))[0]
        if len(sparse_rows) < 3:
            return None

        # 关键：往上找最近的**真正空白带**（穿过稀疏内容行），
        # 因为紧邻表格的稀疏行可能属于表格（如表头），应该留给表格。
        # 从 table_start_y 向上扫描，跳过任何非空白行，直到找到空白行为止。
        blank_end = None
        for y in range(table_start_y - 1, max(0, table_start_y - 100), -1):
            if row_content[y] <= 3:
                blank_end = y + 1  # 空白带的下边界（不含）
                break

        if blank_end is None:
            # 100px 内没找到任何空白带，只能返回原 table_start_y
            return table_start_y

        # 从空白结束位置往上继续找空白带的顶部
        blank_start = blank_end - 1
        for y in range(blank_end - 2, max(0, blank_end - 30), -1):
            if row_content[y] <= 3:
                blank_start = y
            else:
                break

        # 空白带太短（< 3 行）不采用，回退到 table_start_y
        if blank_end - blank_start < 3:
            return table_start_y

        # 切分点取空白带**结束位置**（而非中心），确保切分点在空白带之内的最下端。
        # 这样 table 区域从紧邻文字/表头的位置开始，避免下方文字（尤其是表头字体
        # 反锯齿的上边缘）被误切到 text 区域里。
        return blank_end

    def _detect_top_by_text_segments(self, gray: np.ndarray, h: int, w: int) -> Optional[int]:
        """
        通过分析顶部内容段的**水平位置**识别说明文字。

        判据：**说明文字段的内容集中在图的左侧**（x_start < 20% 图宽 且 x_end < 50% 图宽）。
        - 表格内的表头/数据行通常跨越更宽的水平范围（右侧或全宽都有内容）。
        - 顶部说明文字通常左对齐或居中，不会跨越全宽。

        从上往下扫描内容段，找连续的"说明文字段"，切分点在其后。

        Returns:
            int: 切分 y 坐标（表格开始处），None 表示未找到
        """
        max_search_y = int(h * 0.30)
        content_mask = gray < self.white_threshold
        row_content = content_mask.sum(axis=1)

        x_start_th = w * 0.20  # 内容起点必须在左侧 20% 以内
        x_end_th = w * 0.50    # 内容终点必须在左侧 50% 以内

        # 扫描内容段
        segs = []  # (y_start, y_end, x_start, x_end)
        in_content = False
        seg_start = 0
        for y in range(min(h, max_search_y)):
            is_content = row_content[y] > 3
            if is_content and not in_content:
                seg_start = y
                in_content = True
            elif not is_content and in_content:
                seg_mask = content_mask[seg_start:y]
                col_sum = seg_mask.sum(axis=0)
                content_cols = np.where(col_sum > 0)[0]
                if len(content_cols) > 0:
                    segs.append((seg_start, y,
                                 int(content_cols[0]), int(content_cols[-1])))
                seg_start = y
                in_content = False

        if not segs:
            return None

        # 从上往下扫描，连续的"左侧对齐段"计为说明文字
        text_seg_count = 0
        last_text_end = 0
        next_seg_start = None
        for ys, ye, xs, xe in segs:
            if xs < x_start_th and xe < x_end_th:
                text_seg_count += 1
                last_text_end = ye
            else:
                next_seg_start = ys
                break

        if text_seg_count < 1:
            return None

        # 切分点：下一段（表格开始）的起点。如果找不到下一段，用最后说明文字段的结尾
        if next_seg_start is not None:
            return next_seg_start
        return last_text_end

    def _split_text_region_horizontally(self, region: Region) -> List[Region]:
        """
        对文字区域做水平切分：如果内容左右分布（中间有大段空白），
        切成多个更小的文字区域。

        保守策略：
        - 只处理宽高比 > 10 的细长文字条（普通字符行才可能"左右分布"）
        - 最小空白宽度：max(80px, 图宽 5%) —— 避免误把字间距当分隔
        - 切完每段宽度 >= max(50px, 图宽 3%) —— 避免单字被切开

        Returns:
            [region] 或 [region1, region2, ...]（切分后的多个区域）
        """
        img = region.image
        w, h = img.size

        # 只处理细长文字条（宽高比 > 10）
        if h == 0 or w / h < 10:
            return [region]
        if w < 400:
            return [region]

        gray = np.array(img.convert('L'), dtype=np.uint8)
        col_content = (gray < self.white_threshold).sum(axis=0)
        is_blank_col = col_content <= 3

        # 最小空白宽度
        min_gap_width = max(80, int(w * 0.05))
        # 最小子段宽度
        min_seg_width = max(50, int(w * 0.03))

        bands = self._find_continuous_bands(is_blank_col, min_gap_width)
        # 排除紧贴左右边缘的空白带
        bands = [(s, e) for s, e in bands if s > 0 and e < w]
        if not bands:
            return [region]

        # 切分点：空白带中心
        cut_points = [0]
        for s, e in bands:
            cut_points.append((s + e) // 2)
        cut_points.append(w)

        # 验证每段宽度都 >= min_seg_width
        for i in range(len(cut_points) - 1):
            if cut_points[i + 1] - cut_points[i] < min_seg_width:
                return [region]

        rx0, ry0, rx1, ry1 = region.bbox
        new_regions = []
        for i in range(len(cut_points) - 1):
            cx0 = cut_points[i]
            cx1 = cut_points[i + 1]
            sub_img = img.crop((cx0, 0, cx1, h))
            sub_img_trimmed, (sx, sy) = self._trim_margins(sub_img)
            if sub_img_trimmed.size[0] < 20 or sub_img_trimmed.size[1] < 10:
                continue
            if self._is_empty_region(sub_img_trimmed):
                continue
            new_bbox = (rx0 + cx0 + sx, ry0 + sy,
                        rx0 + cx0 + sx + sub_img_trimmed.size[0],
                        ry0 + sy + sub_img_trimmed.size[1])
            new_regions.append(Region(sub_img_trimmed, new_bbox, 'text'))

        return new_regions if len(new_regions) > 1 else [region]

    def _is_table_row_above_border(self, gray: np.ndarray, y_hline: int, w: int) -> bool:
        """
        判断横线上方 [0, y_hline) 的内容是否为"表格数据行"（首行无独立边框的情况）。

        场景：表格的第一行数据（列 header 或第一行内容）本身**上方没有横线**，
        只有下方有横线。此时 Layer 0 会把这条"列 header 下横线"当成外框顶，
        误将首行数据剥离成 text（见 d1752e16 的 6821×21 text 条）。

        判据：横线上方的内容同时满足以下两条 → 是表格数据行，不应剥离
        1. **横向覆盖率 ≥ 70% 图宽**：表格数据行会跨越几乎整个宽度
        2. **"宽空白列间隔"数量 ≥ 4**：段间空白必须够宽（≥ 图宽 × 0.1% 或至少 5px）
           才算真正的列分隔——中文文字字符间距（1-3px）不算，
           表格数字列间距（≥ 10px）才算。

        与之对比（这两类应正常剥离）：
        - 独立说明文字（如"某某表"、"女性"）：宽度小，不满足第 1 条
        - 跨列合并表头（如"被保险人"）：跨全宽但**单段连续**，不满足第 2 条
        - 中文长文本行（如 58b3cb9e "保险期间：...单位：人民币元"）：
          跨全宽但字符间距太小，wide_gap 数量 << 4，也不满足第 2 条

        Returns:
            True: 是表格数据行，应保持整表不剥离
            False: 不是表格行，走原剥离逻辑
        """
        if y_hline <= 0 or y_hline > gray.shape[0]:
            return False
        top_gray = gray[:y_hline]
        # 每列的内容像素数（垂直投影）
        col_content = (top_gray < self.white_threshold).sum(axis=0)
        # "有内容"的列：至少含 1 个非白像素
        has_content = col_content > 0
        content_cols = np.where(has_content)[0]
        if len(content_cols) == 0:
            return False

        # 横向覆盖率：最左内容列到最右内容列 / 图宽
        x_start = int(content_cols[0])
        x_end = int(content_cols[-1])
        coverage = (x_end - x_start + 1) / float(w)
        if coverage < 0.70:
            return False

        # 找连续内容段
        segments = []  # [(start, end), ...]
        in_seg = False
        seg_start = 0
        for i in range(len(has_content)):
            if has_content[i] and not in_seg:
                seg_start = i
                in_seg = True
            elif not has_content[i] and in_seg:
                segments.append((seg_start, i))
                in_seg = False
        if in_seg:
            segments.append((seg_start, len(has_content)))

        # 只统计"宽空白间隔"（gap ≥ 图宽 × 0.1% 或至少 5px）
        # 这是关键：中文字符间的小空白（1-3px）不算，避免中文文字行被误判为表格行
        wide_gap_th = max(5, int(w * 0.001))
        wide_gaps = 0
        for i in range(1, len(segments)):
            gap = segments[i][0] - segments[i - 1][1]
            if gap >= wide_gap_th:
                wide_gaps += 1

        return wide_gaps >= 4

    def _find_text_split_above_table_row(self, gray: np.ndarray, y_hline: int, w: int) -> Optional[int]:
        """
        Layer 0 判定"上方是表格首行"时，进一步在 [0, y_hline) 内寻找
        "独立文字段 → 表格行"的分界，返回分界作为新切分点。

        典型场景 (58b3cb9e)：
        - y=0-30 红色说明文字（"保险期间：...  单位：人民币元"）
          特征：**两块内容 + 中间大空白**，max_gap 远大于其他 gap
        - y=30-75 列头行"35 36...70"
          特征：**均匀多列**，所有 gap 差不多大小
        - 两者在 y 方向上**紧邻无空白间隔**，靠 y 空白带无法区分
        - 但**行内 gap 分布**完全不同，逐行分析可区分

        算法（逐行分析）：
        1. 用 5 行滑窗扫描每 y 位置
        2. 计算该 y 位置的 col_content 段和段间 gaps
        3. 分类为：
           - 'blank': 段数 < 2（几乎无内容）
           - 'meta':  wide gap 段数 < 4，或 max_gap/median_gap ≥ 10（异常大 gap）
           - 'table': wide gap 段数 ≥ 4，且 max_gap/median_gap < 10（均匀分布）
        4. 找 meta → table 转折点作为切分点

        Returns:
            切分点 y 坐标；找不到时返回 None（保持原逻辑不剥离）
        """
        wide_gap_th = max(5, int(w * 0.001))
        window = 5

        def classify_row(y: int) -> str:
            """分析 y 位置附近 5 行滑窗的行类型"""
            lo = max(0, y - window)
            hi = min(y_hline, y + 1)
            if hi <= lo:
                return 'blank'
            col_content_win = (gray[lo:hi] < self.white_threshold).sum(axis=0)
            has_content_win = col_content_win > 0
            segments = []
            in_seg = False
            seg_start = 0
            for i in range(len(has_content_win)):
                if has_content_win[i] and not in_seg:
                    seg_start = i
                    in_seg = True
                elif not has_content_win[i] and in_seg:
                    segments.append((seg_start, i))
                    in_seg = False
            if in_seg:
                segments.append((seg_start, len(has_content_win)))

            if len(segments) < 2:
                return 'blank'

            gaps = [segments[i][0] - segments[i - 1][1] for i in range(1, len(segments))]
            wide_gaps = [g for g in gaps if g >= wide_gap_th]

            # 段数太少（少于 4 列）→ 元数据
            if len(wide_gaps) < 4:
                return 'meta'

            # 计算 max_gap / median_gap
            max_gap = max(wide_gaps)
            median_gap = float(np.median(wide_gaps))
            # 异常大 gap（比中位数大 10 倍）→ 元数据（两块内容+大空白）
            if median_gap > 0 and max_gap / median_gap >= 10:
                return 'meta'

            return 'table'

        # 从上到下逐 y 分类，找 meta → table 转折点
        prev_non_blank = None
        for y in range(window, y_hline):
            cur = classify_row(y)
            if cur == 'blank':
                continue
            if prev_non_blank == 'meta' and cur == 'table':
                # 转折点：当前 y 是 table 起点，切分点在 y - 1（属于 meta）之后
                # 用 y - window // 2 更靠近真实分界（滑窗中心）
                candidate = y - window // 2
                if candidate < 10:
                    prev_non_blank = cur
                    continue
                # 二次验证（用逐行分类，不用合并投影）：
                # candidate 之上必须有多个 meta 行（说明确实是元数据文字段）。
                # 不能用 `_is_table_row_above_border`（合并投影）——红字多行合并后
                # 字符间小空白被填满，会误判为多列表格行。
                meta_count = sum(
                    1 for yy in range(max(window, candidate - 40), candidate)
                    if classify_row(yy) == 'meta'
                )
                if meta_count >= 5:
                    logger.info(f"    行类型分界 meta→table: y={candidate}, "
                                f"meta_count={meta_count}")
                    return candidate
            prev_non_blank = cur

        return None

    def _detect_top_horizontal_border(self, gray: np.ndarray, h: int, w: int) -> Optional[int]:
        """
        检测表格外边框顶部横线的 y 坐标（Step 5 顶部剥离的最高优先级 Layer）。

        有边框表格的顶部通常是一条清晰的横向长线。检测到该横线后，
        剥离位置**强制**等于横线 y 坐标，这样能保护表格首行（如跨行合并的
        "保单年度末"、稀疏 header 行）不被误剥离切碎（见 Case 1）。

        策略：
        - OpenCV 形态学（水平核）+ Otsu 二值化
        - 核宽度 = max(w × 0.6, 100)，确保只识别真正的长横线
        - 判定阈值：单行连续横线像素 ≥ w × 0.8（覆盖 80% 图宽）
        - 只在图片上部 y < h × 30% 范围内搜索

        Returns:
            int: 最高的横线 y 坐标；None 表示未检测到
        """
        if not HAS_CV2:
            return None
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 水平核宽度：60% 图宽（允许两端有小间断的长横线也能被检出）
            kernel_w = max(int(w * 0.6), 100)
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=1)

            # 每行的横线像素数
            row_line_count = (h_lines > 0).sum(axis=1)

            # 只在图片上部 30% 内搜索
            max_search = max(30, int(h * 0.30))
            # 判定为"有效横线"的最小宽度阈值：80% 图宽（严格约束，排除内部行分隔线）
            min_line_width = int(w * 0.8)

            segment = row_line_count[:max_search]
            valid_rows = np.where(segment >= min_line_width)[0]
            if len(valid_rows) == 0:
                return None

            # 返回最高的横线 y 坐标（即表格外框顶部）
            return int(valid_rows[0])
        except Exception as e:
            logger.warning(f"  顶部横线检测异常: {e}")
            return None

    def _detect_vertical_lines_top_y(self, gray: np.ndarray, h: int, w: int) -> Optional[int]:
        """
        检测竖线的最高 y 坐标（即表格边框顶部）。

        Returns:
            int: 竖线最高点 y 坐标；None 表示未检测到足够竖线
        """
        if not HAS_CV2:
            return None
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            # 竖线核：高度用 min(h*0.15, 100)，避免要求竖线穿越整个表格
            kernel_h = max(min(int(h * 0.15), 100), 30)
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_h))
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=1)

            # 每列的最高非零 y 位置
            has_line = v_lines > 0
            col_line_count = has_line.sum(axis=0)  # 每列竖线像素数
            # 只考虑真正的竖线列（像素数 >= kernel_h）
            valid_cols = np.where(col_line_count >= kernel_h)[0]

            if len(valid_cols) < 2:
                return None

            # 对每个 valid 列，找最高的 y
            tops = []
            for c in valid_cols:
                col_pixels = np.where(has_line[:, c])[0]
                if len(col_pixels) > 0:
                    tops.append(int(col_pixels[0]))

            if len(tops) < 2:
                return None

            # 取所有竖线顶点的中位数（避免个别噪声）
            y_top = int(np.median(tops))
            return y_top
        except Exception as e:
            logger.warning(f"  竖线最高点检测异常: {e}")
            return None

    def _sort_reading_order(self, regions: List[Region]) -> List[Region]:
        """
        按阅读顺序排列区域：从上到下为主，同一行从左到右。

        判定"同一行"：两个区域的 y 范围有显著重叠（重叠 >= 较小区域高度的 50%）。

        算法：
        1. 按 y_center 排序
        2. 分组：y 重叠的归为同一行
        3. 行内按 x 排序
        4. 行间按 y 排序
        """
        if len(regions) <= 1:
            return regions

        # 按 y_center 初步排序
        regions_sorted = sorted(regions, key=lambda r: (r.bbox[1] + r.bbox[3]) / 2)

        # 分组为"行"
        rows = []
        current_row = [regions_sorted[0]]

        for r in regions_sorted[1:]:
            # 检查与当前行的 y 重叠
            prev = current_row[-1]
            # 计算重叠区间
            overlap_y0 = max(r.bbox[1], prev.bbox[1])
            overlap_y1 = min(r.bbox[3], prev.bbox[3])
            overlap_height = max(0, overlap_y1 - overlap_y0)
            min_height = min(r.height, prev.height)

            if min_height > 0 and overlap_height >= min_height * 0.5:
                # y 重叠显著，归为同一行
                current_row.append(r)
            else:
                # 新起一行
                rows.append(current_row)
                current_row = [r]

        rows.append(current_row)

        # 行内按 x 排序，行间按行的最小 y 排序
        result = []
        for row in rows:
            row_sorted = sorted(row, key=lambda r: r.bbox[0])  # 按 x0 从左到右
            result.extend(row_sorted)

        return result

    def _is_empty_region(self, img: Image.Image) -> bool:
        """
        判断区域是否为空白（几乎没有内容像素）。
        内容像素占比 < 0.5% 认为是空区域。
        """
        gray = np.array(img.convert('L'), dtype=np.uint8)
        content_ratio = (gray < self.white_threshold).mean()
        return content_ratio < 0.005

    @staticmethod
    def _snap_cut_to_blank_row(y_cut: int, row_content: np.ndarray, w: int,
                                search_up: int = 100, search_down: int = 10,
                                strict_blank_ratio: float = 0.02,
                                min_band: int = 3) -> int:
        """
        将切分点微调到附近**最长的连续空白带**中心（连续 ≥ min_band 行
        content ≤ 图宽 × strict_blank_ratio）。避免切分位置落在文字/数字中间。

        关键设计：
        - 找**连续多行**的空白带（不是单个低 content 行）—— 避免吸附到
          "文字段内部的行间空白"（只 1-2 行 content 略低）
        - 优先向上搜索更远范围（100 行）—— 因为 non-wide 段末尾可能延伸到
          "过渡内容+3 行字"很远，切分点应回退到金字塔真正结束的空白带
        - 若找到多个空白带，选**最长**的（大概是段间分隔）；同长度选靠近 y_cut 的

        Args:
            y_cut: 粗略切分点
            row_content: 每行的非白像素数
            w: 图宽（用于计算严格空白阈值）
            search_up: 向上搜索行数（默认 100，覆盖 non-wide 段尾部）
            search_down: 向下搜索行数
            strict_blank_ratio: 严格空白阈值 = 图宽 × 该比例（默认 2%）
            min_band: 空白带最小连续行数（默认 3）

        Returns:
            调整后的 y 坐标；找不到合适空白带时返回原 y_cut
        """
        h = len(row_content)
        strict_th = max(3, int(w * strict_blank_ratio))

        lo = max(0, y_cut - search_up)
        hi = min(h, y_cut + search_down)
        if hi <= lo:
            return y_cut

        # 找扫描范围内所有连续空白带（长度 ≥ min_band）
        bands = []
        y = lo
        while y < hi:
            if row_content[y] <= strict_th:
                band_start = y
                while y < hi and row_content[y] <= strict_th:
                    y += 1
                if y - band_start >= min_band:
                    bands.append((band_start, y))
            else:
                y += 1

        if not bands:
            return y_cut

        # 优先选**最长**的空白带（段间分隔通常最宽），同长度选靠近 y_cut 的
        best = max(bands, key=lambda b: (b[1] - b[0],
                                          -abs((b[0] + b[1]) // 2 - y_cut)))
        return (best[0] + best[1]) // 2

    @staticmethod
    def _find_continuous_bands(is_blank: np.ndarray, min_length: int) -> List[Tuple[int, int]]:
        """
        找出连续 True 的区间（长度 >= min_length）。

        Args:
            is_blank: bool 数组
            min_length: 最小连续长度

        Returns:
            list of (start, end): 半开区间 [start, end)
        """
        bands = []
        n = len(is_blank)
        start = None
        for i in range(n):
            if is_blank[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if i - start >= min_length:
                        bands.append((start, i))
                    start = None
        if start is not None and n - start >= min_length:
            bands.append((start, n))
        return bands

    def _classify_region(self, img: Image.Image) -> str:
        """
        简单分类区域类型：table 或 text。

        启发式规则：
        - 包含较多水平线 / 竖线 → table
        - 高度很小（< 100px）且内容稀疏 → text
        - 否则 → table（默认当表格处理更安全）
        """
        w, h = img.size

        # 高度很小的大概率是文字说明
        if h < 80:
            return 'text'

        # 宽度很小的也可能是文字
        if w < 100:
            return 'text'

        # 内容密度检查
        gray = np.array(img.convert('L'), dtype=np.uint8)
        content_ratio = (gray < self.white_threshold).mean()

        # 内容极少 → text
        if content_ratio < 0.01:
            return 'text'

        # 高度很小 + 内容不多 → text
        if h < 150 and content_ratio < 0.05:
            return 'text'

        # 默认当表格
        return 'table'


# ==================== 便捷函数 ====================
