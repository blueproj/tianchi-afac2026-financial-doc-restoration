"""
结果拼接去重模块

功能：
1. 将多个切块的API返回结果拼接为完整文档
2. 处理重叠区域的内容去重
3. 保持阅读顺序正确
"""
import os
import re
import logging
import difflib

logger = logging.getLogger("merger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)


class ResultMerger:
    """切块结果拼接器"""

    @staticmethod
    def clean_api_result(text):
        """清理API返回结果中的markdown代码块包裹"""
        if not text:
            return text
        text = text.strip()
        # 去除开头的 ```markdown 或 ```md
        text = re.sub(r'^```(?:markdown|md)?\s*\n', '', text)
        # 去除结尾的 ```
        text = re.sub(r'\n```\s*$', '', text)
        return text.strip()

    def __init__(self, similarity_threshold=0.5, min_overlap_chars=20,
                 dedup_paragraph_threshold=1.0, enable_dedup_long_lines=False,
                 enable_merge_paragraphs=True, overlap_dedup=True):
        """
        Args:
            similarity_threshold: 文本相似度阈值，高于此值才认为是重叠内容
            min_overlap_chars: 最小重叠字符数
            dedup_paragraph_threshold: 段落去重阈值，默认1.0禁用（避免误删合法重复）
            enable_dedup_long_lines: 是否启用行级精确去重，默认False
            enable_merge_paragraphs: 是否启用段落合并（移除文本块间空行），默认True
                FinixDoc的段落边界与GT不一致，合并减少块数可降低编辑距离
            overlap_dedup: 长文档垂直拼接时是否做重叠去重，默认True（兼容旧重叠切块）。
                当切块器采用强空白带零重叠切分时应设为False：切块间无重叠，
                任何去重都是误删（会错误移除相似的非重叠内容，尤其表格行）。
        """
        self.similarity_threshold = similarity_threshold
        self.min_overlap_chars = min_overlap_chars
        self.dedup_paragraph_threshold = dedup_paragraph_threshold
        self.enable_dedup_long_lines = enable_dedup_long_lines
        self.enable_merge_paragraphs = enable_merge_paragraphs
        self.overlap_dedup = overlap_dedup

    def merge_vertical(self, chunk_results, force_dedup=False):
        """
        垂直拼接（长文档场景：从上到下拼接）

        Args:
            chunk_results: list of str, 每个切块的API返回结果（按顺序）
            force_dedup: 强制启用重叠去重（表格文档仍用重叠切块，需去重）

        Returns:
            str: 拼接后的完整Markdown
        """
        if not chunk_results:
            return ""

        if len(chunk_results) == 1:
            return chunk_results[0]

        # 清理每个chunk的API返回结果
        chunk_results = [self.clean_api_result(r) for r in chunk_results]

        # 零重叠切分（overlap_dedup=False）时切块间无重叠，直接拼接。
        # 避免重叠去重误删相似的非重叠内容（尤其结构相似的表格行）。
        if not (self.overlap_dedup or force_dedup):
            parts = [r for r in chunk_results if r and r.strip()]
            return '\n'.join(parts)

        merged = chunk_results[0]
        for i in range(1, len(chunk_results)):
            merged = self._merge_two_vertical(merged, chunk_results[i])
            logger.debug(f"拼接块 {i}/{len(chunk_results)}, 当前长度={len(merged)}")

        return merged

    def _find_char_overlap(self, text1, text2, search_chars=3000, min_match=50):
        """
        字符级重叠检测：在text1末尾和text2开头寻找最长公共子串

        Returns:
            int: text2中重叠结束位置（0表示未找到）
        """
        tail = text1[-search_chars:] if len(text1) > search_chars else text1
        head = text2[:search_chars] if len(text2) > search_chars else text2

        matcher = difflib.SequenceMatcher(None, tail, head)
        match = matcher.find_longest_match(0, len(tail), 0, len(head))

        if match.size >= min_match:
            # 检查匹配是否在text1末尾附近（允许50字符误差）
            end_in_tail = match.a + match.size
            if end_in_tail >= len(tail) - 50:
                # 找到重叠，计算text2中的结束位置
                overlap_end = match.b + match.size
                # 向后找到下一个换行，避免截断行
                remaining = text2[overlap_end:]
                nl = remaining.find('\n')
                if nl >= 0:
                    return overlap_end + nl + 1
                return overlap_end

        return 0

    def _merge_two_vertical(self, text1, text2):
        """
        拼接两个垂直相邻的文本块（text1在上，text2在下）

        策略：
        1. 先用行级相似度匹配（主方法）
        2. 如果行级匹配失败，用字符级最长公共子串匹配（fallback）
        3. 如果都失败，直接拼接（不插入空行）
        """
        if not text1:
            return text2
        if not text2:
            return text1

        # 将文本按行分割
        lines1 = text1.strip().split('\n')
        lines2 = text2.strip().split('\n')

        # 搜索窗口大小
        search_window = min(80, len(lines1), len(lines2))

        best_match_len = 0  # 最佳匹配的行数
        best_similarity = 0

        # 尝试不同的重叠长度（行级匹配）
        for overlap_len in range(1, search_window + 1):
            # text1末尾overlap_len行 vs text2开头overlap_len行
            tail = lines1[-overlap_len:]
            head = lines2[:overlap_len]

            # 计算相似度
            similarity = self._lines_similarity(tail, head)

            if similarity >= self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_len = overlap_len

        if best_match_len > 0:
            # 行级匹配成功，去除text2开头的重复部分
            logger.debug(f"  行级去重: 移除{best_match_len}行重复内容, 相似度={best_similarity:.2f}")
            merged_lines = lines1 + lines2[best_match_len:]
        else:
            # 行级匹配失败，尝试字符级匹配
            overlap_end = self._find_char_overlap(text1.strip(), text2.strip())
            if overlap_end > 0:
                logger.debug(f"  字符级去重: 移除text2前{overlap_end}字符重叠内容")
                return text1.strip() + '\n' + text2.strip()[overlap_end:]
            else:
                # 两种方法都失败，检查是否有部分行重叠
                if lines1 and lines2:
                    last_line = lines1[-1].strip()
                    first_line = lines2[0].strip()
                    if last_line and first_line:
                        if last_line in first_line and len(last_line) > 5:
                            merged_lines = lines1 + [first_line.replace(last_line, '', 1).strip()] + lines2[1:]
                        elif first_line in last_line and len(first_line) > 5:
                            merged_lines = lines1[:-1] + [last_line.replace(first_line, '', 1).strip()] + lines2
                        else:
                            # 不插入空行，直接拼接
                            merged_lines = lines1 + lines2
                    else:
                        merged_lines = lines1 + lines2
                else:
                    merged_lines = lines1 + lines2

        return '\n'.join(merged_lines)


    def _merge_horizontal(self, chunks):
        """
        水平拼接（左右相邻的块）

        主要处理表格的拼接：
        - 如果左右两块都是表格行，尝试合并为完整表格
        - 否则按文本拼接
        """
        if not chunks:
            return ""
        if len(chunks) == 1:
            return chunks[0]

        merged = chunks[0]
        for i in range(1, len(chunks)):
            merged = self._merge_two_horizontal(merged, chunks[i])
        return merged

    def _merge_two_horizontal(self, text1, text2):
        """
        水平拼接两个块

        对于表格：如果text1末尾有未闭合的<table>，text2开头有<table>的续接部分，
        则尝试合并表格行
        """
        if not text1:
            return text2
        if not text2:
            return text1

        # 检查是否是表格的左右拼接
        # text1末尾的表格行 和 text2开头的表格行 可以合并
        lines1 = text1.strip().split('\n')
        lines2 = text2.strip().split('\n')

        # 尝试表格行合并
        # 如果text1最后几行和text2前几行都是<tr>行，可能可以合并
        merged_lines = []

        # 简化处理：对于水平拼接，检查是否有表格行可以合并
        # 如果text1末尾是 <tr>...</tr> 且 text2开头也是 <tr>...</tr> 的延续
        # 这种情况比较复杂，先用简单的文本拼接
        i = 0
        j = 0

        # 找到text1中最后一个完整的表格
        last_table_start = -1
        last_table_end = -1
        for idx, line in enumerate(lines1):
            if '<table' in line:
                last_table_start = idx
            if '</table>' in line:
                last_table_end = idx

        # 找到text2中第一个表格
        first_table_start = -1
        first_table_end = -1
        for idx, line in enumerate(lines2):
            if '<table' in line and first_table_start == -1:
                first_table_start = idx
            if '</table>' in line and first_table_start != -1 and first_table_end == -1:
                first_table_end = idx
                break

        # 如果两边都有表格，尝试合并
        if (last_table_start != -1 and last_table_end == -1 and
                first_table_start != -1 and first_table_end == -1):
            # text1的表格没有闭合，text2的表格是延续
            # 合并表格部分
            merged_lines = lines1[:last_table_start]
            # 合并表格行
            table_lines = lines1[last_table_start:] + lines2[first_table_start:]
            # 去重重叠行
            table_lines = self._dedup_table_rows(table_lines)
            merged_lines.extend(table_lines)
            # 加入text2中表格之后的内容
            if first_table_end != -1 and first_table_end + 1 < len(lines2):
                merged_lines.extend(lines2[first_table_end + 1:])
        else:
            # 简单拼接，检查行级去重
            merged_lines = lines1 + lines2
            # 去除可能的重复行
            merged_lines = self._dedup_consecutive_lines(merged_lines)

        return '\n'.join(merged_lines)

    def _dedup_table_rows(self, lines):
        """去除表格中的重复行"""
        seen = set()
        result = []
        for line in lines:
            # 标准化行用于比较
            key = re.sub(r'\s+', '', line)
            if key not in seen:
                seen.add(key)
                result.append(line)
        return result

    def _dedup_consecutive_lines(self, lines):
        """去除连续重复行"""
        if not lines:
            return []
        result = [lines[0]]
        for line in lines[1:]:
            if line != result[-1]:
                result.append(line)
        return result

    def _lines_similarity(self, lines1, lines2):
        """
        计算两组行的相似度

        使用SequenceMatcher计算ratio
        """
        text1 = '\n'.join(lines1)
        text2 = '\n'.join(lines2)

        if len(text1) == 0 or len(text2) == 0:
            return 0.0

        # 使用difflib的SequenceMatcher
        matcher = difflib.SequenceMatcher(None, text1, text2)
        return matcher.ratio()

    def _dedup_long_lines(self, text, min_len=40):
        """
        行级精确去重：移除在文本中重复出现的长行（>min_len字符）

        策略：
        1. 遍历所有行，对长度>min_len的非标题非表格行计算hash
        2. 如果hash已存在（精确重复），移除后面的行
        3. 跳过标题行(#开头)和表格行(<table/<tr/<td)
        """
        lines = text.split('\n')
        seen_hashes = set()
        result = []
        removed = 0

        for line in lines:
            stripped = line.strip()
            # 跳过空行、标题、表格
            if (not stripped or
                stripped.startswith('#') or
                '<table' in stripped.lower() or
                '<tr' in stripped.lower() or
                '<td' in stripped.lower() or
                '<th' in stripped.lower() or
                stripped.startswith('```')):
                result.append(line)
                continue

            if len(stripped) >= min_len:
                # 使用行内容的前60字符作为hash key（容忍尾部差异）
                key = stripped[:60]
                if key in seen_hashes:
                    removed += 1
                    if removed <= 3:
                        logger.debug(f"  行级去重: 移除重复行: {stripped[:50]}...")
                    continue  # 跳过重复行
                seen_hashes.add(key)

            result.append(line)

        if removed > 0:
            logger.info(f"  行级去重: 共移除{removed}行重复内容")
        return '\n'.join(result)

    def _dedup_paragraphs(self, text, similarity_threshold=0.85):
        """
        段落级去重：去除重复的段落（由重叠区域检测失败导致）

        策略：
        1. 按空行分割成段落
        2. 对每个段落，检查是否与之前的段落高度相似
        3. 如果相似度超过阈值，移除后面的段落
        """
        lines = text.split('\n')
        paragraphs = []  # [(start_line, end_line, text), ...]

        current_lines = []
        current_start = 0
        in_table = False

        for i, line in enumerate(lines):
            # HTML表格作为一个整体段落，不拆分
            if '<table' in line.lower():
                in_table = True
            if in_table:
                current_lines.append(line)
                if '</table>' in line.lower():
                    in_table = False
                continue

            if line.strip() == '':
                # 空行 = 段落边界
                if current_lines:
                    para_text = '\n'.join(current_lines).strip()
                    if para_text:
                        paragraphs.append((current_start, i, para_text))
                    current_lines = []
                current_start = i + 1
            else:
                if not current_lines:
                    current_start = i
                current_lines.append(line)

        # 最后一个段落
        if current_lines:
            para_text = '\n'.join(current_lines).strip()
            if para_text:
                paragraphs.append((current_start, len(lines), para_text))

        if len(paragraphs) <= 1:
            return text

        # 段落去重
        kept_indices = []
        kept_texts = []

        for idx, (start, end, para_text) in enumerate(paragraphs):
            is_dup = False
            # 跳过标题行（标题可能合法重复）
            is_heading = para_text.startswith('#')
            # 跳过表格（表格去重由其他逻辑处理）
            is_table = '<table' in para_text.lower()

            if not is_heading and not is_table and len(para_text) > 20:
                for prev_text in kept_texts[-30:]:  # 只检查最近30个段落，避免太慢
                    if prev_text.startswith('#') or '<table' in prev_text.lower():
                        continue
                    sim = difflib.SequenceMatcher(None, para_text, prev_text).ratio()
                    if sim >= similarity_threshold:
                        is_dup = True
                        logger.debug(f"  段落去重: sim={sim:.2f}, 移除 {para_text[:40]}...")
                        break

            if not is_dup:
                kept_indices.append(idx)
                kept_texts.append(para_text)

        # 重建文本
        result_lines = []
        for idx in kept_indices:
            start, end, para_text = paragraphs[idx]
            result_lines.append(para_text)
            result_lines.append('')  # 段落间空行

        return '\n'.join(result_lines).strip()

    def _table_col_count(self, table_html):
        """估算表格列数（考虑 colspan），用于判断相邻表格是否可合并。

        修复：老实现只算首个非空行的 <td> 数量，忽略 colspan：
        `<tr><td colspan=4>xxx</td></tr>` 会被算成 1 列，与后面正常的 4 列行
        差异过大导致 _merge_adjacent_tables 拒绝合并。

        新做法：扫描前 5 行，取「按 colspan 加和的最大列数」作为估计值——
        跨列的分组表头行不会拉低这个估计，主数据行的真实列数会体现出来。
        """
        rows = re.findall(r'<tr[^>]*>.*?</tr>', table_html, flags=re.DOTALL | re.IGNORECASE)
        widths = []
        for row in rows[:5]:
            cells_attrs = re.findall(r'<t[dh]([^>]*)>', row, flags=re.IGNORECASE)
            if not cells_attrs:
                continue
            total = 0
            for attrs in cells_attrs:
                cs = re.search(r'colspan\s*=\s*["\']?(\d+)', attrs, flags=re.IGNORECASE)
                total += int(cs.group(1)) if cs else 1
            if total > 0:
                widths.append(total)
        return max(widths) if widths else 0

    def _merge_adjacent_tables(self, text):
        """
        结构感知合并相邻表格块（长文档：chunker 从 y 方向把一张大表切成两张的补救）。

        改进（20260719）：
        1. `_table_col_count` 考虑 colspan，避免分组表头行拉低列数。
        2. 若 gap 里有实质文字（如"髓样癌（所有年龄组）"这类子分组小标题），
           把它作为一整行 `<tr><td colspan=N>gap</td></tr>` 插入到合并后的表中，
           不丢掉这个语义信息。
        3. 中间 gap 允许放宽到 ≤ 50 字符（老版 ≤ 30）——覆盖 8-15 字的分组标题。
        """
        pattern = re.compile(
            r'(<table[^>]*>.*?</table>)(\s*(?:[^\n<]{0,50}\n\s*){0,3})(<table[^>]*>.*?</table>)',
            flags=re.DOTALL | re.IGNORECASE
        )
        guard = 0
        start_pos = 0
        while guard < 1000:
            guard += 1
            match = pattern.search(text, start_pos)
            if not match:
                break
            left, gap, right = match.group(1), match.group(2), match.group(3)
            left_cols = self._table_col_count(left)
            right_cols = self._table_col_count(right)
            gap_text = re.sub(r'\s+', '', gap)
            can_merge = (
                left_cols > 0 and right_cols > 0 and
                abs(left_cols - right_cols) <= 1 and
                len(gap_text) <= 50
            )
            if not can_merge:
                # 从 left 的结尾之后继续找下一对，不要卡死在这里
                start_pos = match.start(1) + len(left)
                continue
            # 构造合并结果
            merged_left = re.sub(r'</table>\s*$', '', left, flags=re.IGNORECASE)
            merged_right = re.sub(r'^\s*<table[^>]*>', '', right, flags=re.IGNORECASE)
            insert = ''
            if gap_text:
                # gap 里有实质文字，作为跨列 header 行插入
                col_span = max(left_cols, right_cols)
                # 清理 gap 里的换行，保留可见文字
                gap_clean_line = re.sub(r'\s+', ' ', gap).strip()
                insert = f'\n<tr><td colspan={col_span}>{gap_clean_line}</td></tr>\n'
            merged = merged_left + insert + merged_right
            text = text[:match.start()] + merged + text[match.end():]
            # 合并后从 merged 起点继续扫，允许链式合并 3+ 张
            start_pos = match.start()
        return text
    
    def _remove_heading_marks(self, text):
        """
        将标题转换为纯文本（移除#前缀）
        
        表格文档GT没有标题，FinixDoc可能输出标题。
        移除#前缀但保留文本内容，避免产生不匹配的heading块。
        """
        lines = text.split('\n')
        result = []
        for line in lines:
            m = re.match(r'^(#{1,6})\s+(.*)', line)
            if m:
                result.append(m.group(2))  # 保留文本，移除#前缀
            else:
                result.append(line)
        return '\n'.join(result)
    
    def _dedup_consecutive_table_rows(self, text):
        """
        移除连续完全相同的表格行（去除切条重叠导致的重复行）。

        仅删除相邻且规范化后完全一致的单行 <tr>...</tr>：
        费率表相邻行内容各不相同，完全一致的相邻行几乎必为切条重叠副本，
        故精确去重安全（不会误删相似但不同的行）。
        """
        lines = text.split('\n')
        result = []
        prev_key = None
        for line in lines:
            s = line.strip()
            if s.startswith('<tr') and s.endswith('</tr>'):
                key = re.sub(r'\s+', '', s)
                if key == prev_key:
                    continue  # 跳过与上一行完全相同的表格行
                prev_key = key
                result.append(line)
            else:
                if s:
                    prev_key = None
                result.append(line)
        return '\n'.join(result)

    def post_process_table_doc(self, text):
        """
        表格文档专用后处理
        
        1. 去除markdown代码块包裹
        2. 移除标题标记（GT表格文档无标题）
        3. 修复截断的表格（API复读/token耗尽导致未闭合）
        4. 合并相邻表格
        5. 列数对齐（每行补齐到表头列数）
        6. 去除多余空行
        """
        # 去除markdown代码块包裹
        text = re.sub(r'^```(?:markdown|md)?\s*\n', '', text.strip())
        text = re.sub(r'\n```\s*$', '', text.strip())
        text = text.strip()
        
        # 移除标题标记（表格文档GT无标题）
        text = self._remove_heading_marks(text)

        # 修复截断的表格
        text = self._fix_truncated_tables(text)
        
        # 合并相邻表格
        text = self._merge_adjacent_tables(text)

        # 列数对齐（仅当差异不大时补齐，避免过度膨胀）
        # text = self._align_table_columns(text)  # 暂禁用：对超宽表反而增加编辑距离

        # 截断复读行：API复读会导致一行有上百个<td>，超过参考列数的截断
        text = self._trim_overlong_rows(text)

        # 去除切条重叠导致的连续重复表格行
        text = self._dedup_consecutive_table_rows(text)

        # 方向二：全角数字/拉丁字母/罗马数字→半角（对齐GT官方约定）
        text = re.sub(r'[０-９]', lambda m: chr(ord(m.group(0)) - 0xFEE0), text)
        text = re.sub(r'[Ａ-Ｚａ-ｚ]', lambda m: chr(ord(m.group(0)) - 0xFEE0), text)
        roman_full_to_half = {
            'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV',
            'Ⅴ': 'V', 'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII',
            'Ⅸ': 'IX', 'Ⅹ': 'X', 'Ⅺ': 'XI', 'Ⅻ': 'XII',
        }
        text = ''.join(roman_full_to_half.get(ch, ch) for ch in text)
        
        # 去除连续空行（最多保留一个）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 去除行尾空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)
        
        # 去除开头和结尾的空行
        text = text.strip()
        
        return text

    def _fix_truncated_tables(self, text):
        """
        修复截断的表格：有<table>但无</table>的片段。

        策略：找到最后一个完整的</tr>，截断后面的不完整内容，补上</table>。
        如果连一个完整<tr>都没有，整块置为空表格。
        """
        # 找所有<table>的位置
        table_starts = [m.start() for m in re.finditer(r'<table[^>]*>', text, re.IGNORECASE)]
        table_ends = [m.end() for m in re.finditer(r'</table>', text, re.IGNORECASE)]

        if not table_starts:
            return text

        # 检查每个<table>是否有配对的</table>
        result_parts = []
        last_pos = 0
        for i, start in enumerate(table_starts):
            # 找这个<table>对应的</table>
            matching_end = None
            for end_pos in table_ends:
                if end_pos > start:
                    matching_end = end_pos
                    break

            if matching_end is not None:
                # 有配对，跳过
                continue

            # 没有配对的</table> → 截断修复
            # 找<table>标签结束位置
            tag_end = text.index('>', start) + 1
            table_content = text[tag_end:]

            # 找最后一个完整的</tr>
            last_tr_end = table_content.rfind('</tr>')
            if last_tr_end == -1:
                last_tr_end = table_content.rfind('</TR>')

            if last_tr_end >= 0:
                # 截取到最后一个完整</tr>，补</table>
                fixed_content = table_content[:last_tr_end + 5]  # +5 for '</tr>'
                fixed = text[:tag_end] + fixed_content + '\n</table>'
                text = fixed + text[len(text):]  # 替换整个后续（因为是最后一个未闭合的table）
            else:
                # 连一个完整tr都没有，直接闭合
                text = text[:tag_end] + '</table>'

        return text

    def _align_table_columns(self, text):
        """
        对齐表格列数：每行<td>/<th>数补齐到该表格的最大列数。
        （当前暂禁用）
        """
        def align_single_table(table_html):
            # 提取所有<tr>行
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', table_html, re.DOTALL | re.IGNORECASE)
            if not rows:
                return table_html

            # 统计每行的列数
            def count_cells(row_content):
                return len(re.findall(r'<t[dh][^>]*>', row_content, re.IGNORECASE))

            col_counts = [count_cells(r) for r in rows]
            if not col_counts:
                return table_html

            # 参考列数：取第一行（通常是表头）或最大值
            ref_cols = col_counts[0] if col_counts[0] > 0 else max(col_counts)
            if ref_cols <= 0:
                return table_html

            # 逐行补齐
            def pad_row(match):
                row_content = match.group(1)
                n = count_cells(row_content)
                if n < ref_cols:
                    padding = '<td></td>' * (ref_cols - n)
                    # 在</tr>前插入
                    row_content = row_content.rstrip() + padding
                return '<tr>' + row_content + '</tr>'

            result = re.sub(
                r'<tr[^>]*>(.*?)</tr>',
                pad_row,
                table_html,
                flags=re.DOTALL | re.IGNORECASE
            )
            return result

        # 对每个完整的<table>...</table>做列对齐
        def process_table(match):
            return align_single_table(match.group(0))

        text = re.sub(
            r'<table[^>]*>.*?</table>',
            process_table,
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        return text

    def _trim_overlong_rows(self, text):
        """
        截断复读行：API复读导致一行内<td>远超正常列数（如2083列 vs 正常36列）。

        策略：
        1. 确定参考列数（前几行的中位数）
        2. 超过参考列数 1.5 倍的行，截取前 ref_cols 个完整 cell
        """
        def trim_single_table(table_html):
            tr_pattern = re.compile(r'<tr[^>]*>(.*?)</tr>', re.DOTALL | re.IGNORECASE)
            matches = list(tr_pattern.finditer(table_html))
            if not matches:
                return table_html

            def count_cells(content):
                return len(re.findall(r'<t[dh]', content, re.IGNORECASE))

            col_counts = [count_cells(m.group(1)) for m in matches]
            if not col_counts:
                return table_html

            # 参考列数：取最小的非零列数（复读行一定是异常大的那个）
            non_zero = [c for c in col_counts if c > 0]
            if not non_zero:
                return table_html
            ref_cols = min(non_zero)
            if ref_cols <= 0:
                ref_cols = 1

            threshold = max(ref_cols + 5, int(ref_cols * 1.5))

            # 从后往前替换超长行（避免位移问题）
            result = table_html
            for m in reversed(matches):
                n = count_cells(m.group(1))
                if n > threshold:
                    # 快速截断：找第 ref_cols 个 </td> 或 </th> 的位置
                    content = m.group(1)
                    close_tags = list(re.finditer(r'</t[dh]>', content, re.IGNORECASE))
                    if len(close_tags) >= ref_cols:
                        cut_pos = close_tags[ref_cols - 1].end()
                        trimmed = content[:cut_pos]
                        replacement = '<tr>' + trimmed + '</tr>'
                        result = result[:m.start()] + replacement + result[m.end():]

            return result

        def process_table(match):
            return trim_single_table(match.group(0))

        text = re.sub(
            r'<table[^>]*>.*?</table>',
            process_table,
            text,
            flags=re.DOTALL | re.IGNORECASE
        )
        return text
    
    # 匹配应该作为新块开头的行模式（保留前面的空行）
    _NEW_BLOCK_PATTERNS = [
        r'^\d+\.',           # 1. 2. 等
        r'^\d+\)',           # 1) 2) 等
        r'^[\(（]\d+[\)）]', # (1) （1） 等
        r'^[\(（][一二三四五六七八九十百千万]+[\)）]',  # (一) （一） 等
        r'^[一二三四五六七八九十]+、',  # 一、 二、 等
        r'^第[一二三四五六七八九十\d]+[条章节款项]',     # 第一条 等
        r'^[-*•◆○●◇◇]',     # bullet points
        r'^注[：:]',           # 注：
        r'^说明[：:]',         # 说明：
        r'^附[：:]',           # 附：
        r'^备注[：:]',         # 备注：
    ]
    
    def _merge_text_paragraphs(self, text):
        """
        段落合并：移除连续文本块之间的空行（保留标题前后空行）
        
        训练集测试结论：
        - all-merge（移除所有文本间空行）= 147块, R=17.2
        - no-merge（保留所有空行）= 373块, R=5.6
        - selective-merge（选择性保留）= 364块, R=7.5
        - all-merge效果最好，因为减少块数=减少最低编辑距离
        
        不影响Text Score，因为评测器的normalize_text会移除空行
        """
        lines = text.split('\n')
        result = []
        prev_was_text = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # 标题行
            if re.match(r'^#{1,6}\s', stripped):
                result.append(line)
                prev_was_text = False
            # 表格行
            elif '<table' in stripped.lower() or '<tr' in stripped.lower():
                result.append(line)
                prev_was_text = False
            # 空行
            elif not stripped:
                # 如果前一行是文本，且下一行也是文本，则跳过空行（合并段落）
                if prev_was_text and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line and not re.match(r'^#{1,6}\s', next_line) and '<table' not in next_line.lower():
                        continue  # 跳过空行，合并段落
                result.append(line)
                prev_was_text = False
            else:
                result.append(line)
                prev_was_text = True
        
        return '\n'.join(result)
    
    def post_process(self, text):
        """
        后处理：清理拼接结果中的常见问题
        """
        # 去除markdown代码块包裹 (```markdown ... ```)
        text = re.sub(r'^```(?:markdown|md)?\s*\n', '', text.strip())
        text = re.sub(r'\n```\s*$', '', text.strip())
        text = text.strip()

        # 行级精确去重（去除重叠区域检测失败导致的重复长行）
        if self.enable_dedup_long_lines:
            text = self._dedup_long_lines(text)

        # 段落级去重（去除重叠区域检测失败导致的重复段落）
        if self.dedup_paragraph_threshold < 1.0:
            text = self._dedup_paragraphs(text, similarity_threshold=self.dedup_paragraph_threshold)

        # 合并相邻表格（跨chunk拆分的表格）
        text = self._merge_adjacent_tables(text)

        # 段落合并（减少块数，改善Read Order Score）
        if self.enable_merge_paragraphs:
            text = self._merge_text_paragraphs(text)

        # 去除连续空行（最多保留一个）
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除行首行尾多余空白
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        text = '\n'.join(lines)

        # 去除开头和结尾的空行
        text = text.strip()

        return text

    def merge(self, chunk_results, positions=None, doc_type='long'):
        """
        通用拼接接口

        Args:
            chunk_results: list of str, 按顺序的切块结果
            positions: 位置信息列表（可选），含 row_idx/col_idx/is_extra
            doc_type: 'long' 或 'table'

        Returns:
            str: 拼接后的完整文本
        """
        if doc_type == 'long':
            result = self.merge_vertical(chunk_results)
            return self.post_process(result)
        else:
            # 表格文档：一维纵向合并 + 表格后处理
            result = self.merge_vertical(chunk_results)
            return self.post_process_table_doc(result)


    def merge_horizontal_table_strips(self, segments):
        """
        水平合并同一行的多个列段。

        将左段和右段的 <tr> 按行号对齐，拼接各段的 <td>/<th> 单元格。
        """
        if not segments:
            return ''
        if len(segments) == 1:
            return segments[0]

        # 提取每段的 <tr> 行
        def extract_tr_rows(html_text):
            rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html_text or '', flags=re.DOTALL | re.IGNORECASE)
            return rows

        all_segment_rows = [extract_tr_rows(seg) for seg in segments]
        max_rows = max(len(rows) for rows in all_segment_rows) if all_segment_rows else 0

        if max_rows == 0:
            # 没有 <tr>，直接拼接文本
            return '\n'.join(s for s in segments if s and s.strip())

        # 按行号对齐拼接
        merged_rows = []
        for i in range(max_rows):
            cells = ''
            for seg_rows in all_segment_rows:
                if i < len(seg_rows):
                    cells += seg_rows[i]
            merged_rows.append(f'<tr>{cells}</tr>')

        # 重建表格
        return '<table border="1">\n' + '\n'.join(merged_rows) + '\n</table>'
