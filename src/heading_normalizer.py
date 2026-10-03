"""
标题级别后处理模块

功能：
1. 自动检测FinixDoc-VL输出的标题级别偏移量
2. 修正标题级别以匹配GT约定
3. 归一化标题文本（去除多余空格等）
4. 确保相同编号模式的标题使用一致级别

核心原理：
- FinixDoc-VL输出的标题级别通常从1或2开始
- GT的标题级别因文档而异（1, 2, 3等）
- 通过分析标题编号模式和层级结构，推断正确的级别偏移量

启发式规则：
1. 单一级别且起始为2 → GT可能起始为1，偏移-1
2. 多级别且起始为1，顶级标题用中文数字+顿号 → GT可能起始为2，偏移+1
3. 多级别且起始为2 → GT可能起始为1，偏移-1
4. 单一级别且起始为1 → GT可能起始为1，偏移0
5. 多级别且起始为1，顶级标题用阿拉伯数字+点 → GT可能起始为1，偏移0
"""
import re
import logging

logger = logging.getLogger("heading_normalizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.addHandler(handler)


def detect_heading_pattern(content):
    """
    检测标题文本的编号模式
    
    Returns:
        str: 模式名称 ('cn_num', 'ar_num', 'paren_ar', 'paren_cn', 'other')
    """
    content = content.strip()
    if re.match(r'[\u4e00-\u9fff]+[、]', content):
        return 'cn_num'  # 中文数字+顿号，如"十七、"
    elif re.match(r'\d+\.', content):
        return 'ar_num'  # 阿拉伯数字+点，如"40."
    elif re.match(r'[\(（]\d+[\)）]', content):
        return 'paren_ar'  # 括号+阿拉伯数字，如"(1)"
    elif re.match(r'[\(（][\u4e00-\u9fff]+[\)）]', content):
        return 'paren_cn'  # 括号+中文数字，如"(五十三)"
    elif re.match(r'\d+\)', content):
        return 'ar_paren'  # 数字+右括号，如"1)"
    else:
        return 'other'


def extract_headings(text):
    """
    从文本中提取所有标题行
    
    Returns:
        list of (level, content, line_index)
    """
    headings = []
    lines = text.split('\n')
    for i, line in enumerate(lines):
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            headings.append((level, content, i))
    return headings


def normalize_heading_text(text):
    """
    归一化标题文本

    对齐官方口径：GT 全部为半角标注，API 也不会返回全角。
    因此只做单向的"全角→半角"清理，不再执行任何"半角→全角"转换。
    """
    # 全角罗马数字→半角映射（用于修正API偶尔输出的Ⅰ/Ⅱ/…）
    roman_full_to_half = {
        'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV',
        'Ⅴ': 'V', 'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII',
        'Ⅸ': 'IX', 'Ⅹ': 'X', 'Ⅺ': 'XI', 'Ⅻ': 'XII',
    }

    lines = text.split('\n')
    result = []
    for line in lines:
        m = re.match(r'^(#{1,6}\s+)(.*)', line)
        if m:
            prefix = m.group(1)
            content = m.group(2).strip()

            # 移除注册号信息：'条款 注册号：C000078...' → '条款'
            content = re.sub(r'\s*注册号[：:]\S+', '', content)

            # 1. 去除数字编号后的空格：'40. 严重' → '40.严重', '1.1 合同' → '1.1合同'
            content = re.sub(r'(\d+(?:\.\d+)+\.?)\s+(?=[^\d\s])', r'\1', content)
            content = re.sub(r'(\d+\.)\s+(?=[^\d\s])', r'\1', content)

            # 2. 不做括号方向转换，保留API原始输出

            # 3. 全角罗马数字→半角
            content = ''.join(roman_full_to_half.get(ch, ch) for ch in content)

            # 4. 全角数字/拉丁字母 → 半角
            content = re.sub(r'[\uff10-\uff19]', lambda mm: chr(ord(mm.group(0)) - 0xFEE0), content)
            content = re.sub(r'[\uff21-\uff3a\uff41-\uff5a]', lambda mm: chr(ord(mm.group(0)) - 0xFEE0), content)

            # 5. 统一prefix为一个空格
            prefix = re.sub(r'\s+', ' ', prefix)
            result.append(prefix + content)
        else:
            result.append(line)
    return '\n'.join(result)


def merge_multiline_headings(text):
    """
    合并多行标题
    
    FinixDoc有时将一个标题输出为多行：
    # 安盛天平个人交通工具意外伤害保险
    (2022版) (互联网专属) 条款
    
    GT中这是单个标题行。合并为：
    # 安盛天平个人交通工具意外伤害保险 (2022版) (互联网专属) 条款
    
    规则：`#`标题行后紧跟的非空非`#`行，如果不以句号/分号结尾且较短，合并
    """
    lines = text.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            # 这是一个标题行，检查后续行是否是续行
            heading_prefix = m.group(1)
            heading_content = m.group(2).strip()
            
            # 收集续行
            j = i + 1
            while j < len(lines):
                next_line = lines[j].strip()
                # 续行条件：非空、非标题、不以句号/分号结尾、较短
                if (next_line and
                    not re.match(r'^#{1,6}\s', next_line) and
                    not next_line.startswith('```') and
                    not next_line.startswith('<table') and
                    len(next_line) < 80 and
                    not next_line.endswith('。') and
                    not next_line.endswith('；') and
                    not next_line.endswith('.') and
                    # 续行通常以括号或文字开头
                    (next_line.startswith('(') or 
                     next_line.startswith('（') or
                     re.match(r'^[\u4e00-\u9fff]', next_line) or
                     re.match(r'^[A-Z]', next_line))):
                    heading_content += ' ' + next_line
                    j += 1
                else:
                    break
            
            result.append(heading_prefix + ' ' + heading_content)
            i = j
        else:
            result.append(line)
            i += 1
    
    return '\n'.join(result)


def detect_and_fix_missing_headings(text):
    """
    检测并修复缺失的#标记
    
    FinixDoc有时不输出#标记（特别是每个chunk的第一个标题）：
    (四十七)严重自身免疫性肝炎
    
    GT中这是标题：# (四十七)严重自身免疫性肝炎
    
    策略：
    1. 收集文档中已有的标题模式及其级别
    2. 扫描非标题行，如果匹配已知模式且像标题，添加#标记
    """
    lines = text.split('\n')
    
    # 第一遍：收集已有的标题模式 -> 级别
    pattern_levels = {}
    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            level = len(m.group(1))
            content = m.group(2).strip()
            pattern = detect_heading_pattern(content)
            if pattern != 'other':
                if pattern not in pattern_levels:
                    pattern_levels[pattern] = {}
                if level not in pattern_levels[pattern]:
                    pattern_levels[pattern][level] = 0
                pattern_levels[pattern][level] += 1
    
    if not pattern_levels:
        return text
    
    # 对每种模式，找出众数级别
    pattern_target = {}
    for pattern, level_counts in pattern_levels.items():
        best_level = max(level_counts, key=level_counts.get)
        max_count = level_counts[best_level]
        for level in sorted(level_counts.keys()):
            if level_counts[level] == max_count:
                best_level = level
                break
        pattern_target[pattern] = best_level
    
    # 第二遍：检测缺失的#标记
    result = []
    fixes = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # 跳过已有的标题行
        if re.match(r'^#{1,6}\s', stripped):
            result.append(line)
            continue
        
        # 跳过空行、表格、代码块
        if (not stripped or 
            stripped.startswith('```') or 
            stripped.startswith('<table') or
            stripped.startswith('<tr')):
            result.append(line)
            continue
        
        # 检测是否像标题
        pattern = detect_heading_pattern(stripped)
        if pattern in pattern_target:
            # 额外条件：短行、不以标点结尾
            is_short = len(stripped) < 60
            no_punctuation = (not stripped.endswith('。') and 
                             not stripped.endswith('；') and
                             not stripped.endswith('.') and
                             not stripped.endswith('：'))
            
            # 检查下一行是否是空行或标题
            next_is_blank_or_heading = False
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if (not next_line or 
                    re.match(r'^#{1,6}\s', next_line) or
                    next_line.startswith('```')):
                    next_is_blank_or_heading = True
            elif i == len(lines) - 1:
                next_is_blank_or_heading = True
            
            # 检查上一行是否是空行/标题/以标点结尾
            prev_is_valid = False
            if i > 0:
                prev_line = lines[i - 1].strip()
                if (not prev_line or 
                    re.match(r'^#{1,6}\s', prev_line) or
                    prev_line.endswith('。') or
                    prev_line.endswith('；') or
                    prev_line.endswith('：') or
                    prev_line.endswith(':') or
                    prev_line.endswith('.')):
                    prev_is_valid = True
            else:
                prev_is_valid = True
            
            if is_short and no_punctuation and prev_is_valid:
                target_level = pattern_target[pattern]
                result.append('#' * target_level + ' ' + stripped)
                fixes += 1
                continue
        
        result.append(line)
    
    if fixes > 0:
        logger.info(f"修复了{fixes}个缺失的#标记")
    
    return '\n'.join(result)


def remove_duplicate_title(text):
    """
    检测并移除重复标题
    
    FinixDoc经常在L1标题后，又在L2输出相同标题+额外内容（如注册号），
    导致后续所有标题级别被推高一级。
    
    策略：如果前两个标题文本相似（一个是另一个的前缀），
    且第二个级别更高，则移除第二个标题，并将后续标题级别-1。
    """
    headings = extract_headings(text)
    if len(headings) < 2:
        return text
    
    h1_level, h1_content, h1_idx = headings[0]
    h2_level, h2_content, h2_idx = headings[1]
    
    # 检查是否是重复标题
    if h2_level > h1_level:
        # 文本相似度检查：一个是另一个的前缀，或前20字符相同
        is_dup = False
        if h1_content and h2_content:
            # 去除空格后比较
            c1 = re.sub(r'\s+', '', h1_content)
            c2 = re.sub(r'\s+', '', h2_content)
            if len(c1) > 10 and len(c2) > 10:
                if c1 in c2 or c2 in c1:
                    is_dup = True
                elif c1[:20] == c2[:20]:
                    is_dup = True
        
        if is_dup:
            # 移除第二个标题行
            lines = text.split('\n')
            if h2_idx < len(lines):
                lines[h2_idx] = ''  # 替换为空行
            text = '\n'.join(lines)
            
            # 将所有级别 > h1_level 的标题下移一级
            lines = text.split('\n')
            result = []
            removed = 0
            for line in lines:
                m = re.match(r'^(#{1,6})\s+(.*)', line)
                if m:
                    level = len(m.group(1))
                    if level > h1_level:
                        new_level = max(1, level - 1)
                        result.append('#' * new_level + ' ' + m.group(2))
                        removed += 1
                    else:
                        result.append(line)
                else:
                    result.append(line)
            
            if removed > 0:
                logger.info(f"移除重复标题，{removed}个标题级别-1")
            
            return '\n'.join(result)
    
    return text


def _ar_num_depth(content):
    """ar_num 编号的点号深度：'1.'→1, '1.1'→2, '1.1.1'→3"""
    m = re.match(r'(\d+(?:\.\d+)*)', content.strip())
    if m:
        return len(m.group(1).rstrip('.').split('.'))
    return 1


def normalize_heading_levels_per_pattern(text):
    """
    基于编号模式 + 文档结构分配标题级别（V9，2026-07-22 依据100训练GT重标定）

    统计依据（finixdocbench_huge_long_100 的 100 份 GT）：
    - GT 几乎不使用 L4+（仅约 1.2%）→ 层级统一封顶 L3；
      修掉旧版 ar3→L4、paren_ar→L4 等系统性过深错误。
    - other（无编号）：文档标题（首个标题 或 长度>10）→ L1；其余章节名(总则/释义…)→ L2
      （GT 中非标题 other 多为 L2，占比高于 L1）。
    - cn_num（一、）→ L2（GT 73%）。
    - paren_ar/ar_paren（(1)/1)）→ L3（GT 79%，与基准无关）。
    - paren_cn（（一））→ 若文档含更外层体系(cn_num/ar_num/文档标题)则 L3，否则 L2。
    - ar_num（十进制）按点号深度递进：level = ar_base + (depth-1)，
      ar_base = 2（文档含 cn_num 或文档标题，正文整体下沉一级）否则 1。

    说明：残余误差主要来自"每文档绝对基准"的固有歧义（同一编号在不同文档处于不同层级），
    纯规则无法进一步区分；实测本方案在训练集上标题层级准确率 46%→52%，
    read_order 16.59→17.36（两项均提升，且不做整体 rebase 以免破坏已正确的相对结构）。
    """
    headings = extract_headings(text)
    if not headings:
        return text

    patterns_present = set(detect_heading_pattern(c) for _, c, _ in headings)
    has_cn = 'cn_num' in patterns_present
    has_ar = 'ar_num' in patterns_present

    # 文档标题检测：other模式且内容较长，或首标题为other且后接ar_num
    has_doc_title = False
    for level, content, idx in headings:
        if detect_heading_pattern(content) == 'other' and len(content) > 10:
            has_doc_title = True
            break
    if (not has_doc_title and headings
            and detect_heading_pattern(headings[0][1]) == 'other' and has_ar):
        has_doc_title = True

    # ar_num 基准级；paren_cn 是否有更外层父级
    ar_base = 2 if (has_cn or has_doc_title) else 1
    paren_cn_level = 3 if (has_cn or has_ar or has_doc_title) else 2

    lines = text.split('\n')
    result = []
    changes = 0
    seen = False
    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if not m:
            result.append(line)
            continue
        content = m.group(2).strip()
        pattern = detect_heading_pattern(content)
        is_first = not seen
        seen = True
        if pattern == 'ar_num':
            target_level = ar_base + (_ar_num_depth(content) - 1)
        elif pattern == 'cn_num':
            target_level = 2
        elif pattern in ('paren_ar', 'ar_paren'):
            target_level = 3
        elif pattern == 'paren_cn':
            target_level = paren_cn_level
        else:  # other
            target_level = 1 if (is_first or len(content) > 10) else 2

        # GT 几乎不用 L4+：统一封顶 L3
        target_level = min(max(target_level, 1), 3)

        if len(m.group(1)) != target_level:
            changes += 1
        result.append('#' * target_level + ' ' + content)

    if changes > 0:
        logger.info(f"V9 per-pattern级别归一化：修改了{changes}个标题的级别"
                    f"（ar_num基准L{ar_base}, paren_cn L{paren_cn_level}, 封顶L3）")

    return '\n'.join(result)


def downgrade_text_item_headings(text):
    """
    将被错误标记为标题的文本项降级为普通文本

    FinixDoc 有时把正文中的编号项输出为标题，但它们本应是普通文本行。
    典型对比：
    - 误判为标题: # (1)心功能衰竭程度达到纽约心脏病学会...；（长、以分号结尾 → 应是正文）
    - 真正标题:   #### (1) 白血病（短、无标点 → 保留为标题）

    判断条件：
    - paren_ar/ar_paren 模式且内容较长（>30字符）→ 降级为文本
    - paren_ar/ar_paren 模式且以句号/分号结尾 → 降级为文本
    - 短的（<=30字符）且不以标点结尾 → 保留为标题
    """
    headings = extract_headings(text)
    if not headings:
        return text
    
    lines = text.split('\n')
    result = []
    downgraded = 0
    for line in lines:
        m = re.match(r'^(#{1,6})\s+(.*)', line)
        if m:
            content = m.group(2).strip()
            pattern = detect_heading_pattern(content)
            
            should_downgrade = False
            
            if pattern in ('paren_ar', 'ar_paren'):
                is_long = len(content) > 30
                has_punctuation = (content.endswith('。') or 
                                   content.endswith('；') or 
                                   content.endswith('：') or
                                   content.endswith('.') or
                                   content.endswith(';'))
                if is_long or has_punctuation:
                    should_downgrade = True
            
            if should_downgrade:
                result.append(content)  # 移除#前缀，保留为文本
                downgraded += 1
            else:
                result.append(line)
        else:
            result.append(line)
    
    if downgraded > 0:
        logger.info(f"降级了{downgraded}个文本项标题为普通文本")
    
    return '\n'.join(result)


def normalize_text_content(text):
    """
    V7新增：归一化文本内容（非标题行）
    
    对齐官方口径：GT 全部为半角，API 也不会返回全角。因此仅执行"全角→半角"
    单向清理，同时保留GT里显著存在的空格删除模式。
    """
    # 全角罗马数字→半角映射
    roman_full_to_half = {
        'Ⅰ': 'I', 'Ⅱ': 'II', 'Ⅲ': 'III', 'Ⅳ': 'IV',
        'Ⅴ': 'V', 'Ⅵ': 'VI', 'Ⅶ': 'VII', 'Ⅷ': 'VIII',
        'Ⅸ': 'IX', 'Ⅹ': 'X', 'Ⅺ': 'XI', 'Ⅻ': 'XII',
    }

    lines = text.split('\n')
    result = []
    for line in lines:
        # 跳过标题行（标题由normalize_heading_text处理）
        if re.match(r'^#{1,6}\s', line):
            result.append(line)
            continue

        # 跳过表格行
        if '<table' in line.lower() or '<tr' in line.lower() or '<td' in line.lower() or '<th' in line.lower():
            result.append(line)
            continue

        content = line

        # 1. 全角罗马数字→半角
        content = ''.join(roman_full_to_half.get(ch, ch) for ch in content)

        # 2. 移除中文字符之间的空格
        content = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', content)
        content = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', content)

        # 3. 括号编号后去除空格：(1) 过去 → (1)过去
        content = re.sub(r'([\)）])\s+([\u4e00-\u9fff])', r'\1\2', content)

        # 4. 中文括号编号前去除空格： text (1) → text(1)
        content = re.sub(r'([\u4e00-\u9fff])\s+([\(（]\d)', r'\1\2', content)

        # 5. 全角数字/拉丁字母→半角
        content = re.sub(r'[\uff10-\uff19]', lambda m: chr(ord(m.group(0)) - 0xFEE0), content)
        content = re.sub(r'[\uff21-\uff3a\uff41-\uff5a]', lambda m: chr(ord(m.group(0)) - 0xFEE0), content)

        result.append(content)

    return '\n'.join(result)


def strip_heading_in_table_cells(text):
    """剔除表格单元格内的 Markdown 标题标记。

    FinixDoc 有时把表格内的分节标题（如"中症疾病"）输出为
    `<td colspan=2>## 中症疾病</td>`，标题层级 `##` 混进了单元格文本。
    GT 中表格单元格从不含 `#`（实测 100 份 GT = 0 处），且会污染 table TEDS 的
    单元格文本比对。此处剔除紧跟在 <td>/<th> 开标签之后的 `#{1,6}\\s+` 标题标记，
    仅保留文字。
    """
    return re.sub(r'(<t[dh][^>]*>)\s*#{1,6}\s+', r'\1', text)


def post_process_headings(text, use_pattern_norm=True):
    """
    标题后处理主函数 - V7
    
    流程：
    1. 合并多行标题
    2. 修复缺失的#标记
    3. 降级文本项标题（FinixDoc错误标记的编号项）
    4. 移除重复标题
    5. V7: 基于编号模式直接分配标题级别（含文档标题检测）
    6. 归一化标题文本（去空格等）
    7. V7新增: 归一化文本内容（罗马数字、标点、空格）
    
    Args:
        text: FinixDoc-VL API输出的Markdown文本
        use_pattern_norm: 保留参数兼容性
    
    Returns:
        str: 后处理后的文本
    """
    # 0. 剔除加粗标记 **（GT 几乎不使用；经训练集验证移除后 read_order 略升、text 不变）
    text = text.replace('**', '')

    # 0b. 剔除表格单元格内的标题标记（<td>## 中症疾病</td> → <td>中症疾病</td>）
    text = strip_heading_in_table_cells(text)

    # 1. 合并多行标题
    text = merge_multiline_headings(text)
    
    # 2. 修复缺失的#标记
    text = detect_and_fix_missing_headings(text)
    
    # 3. 降级文本项标题（如(1)心功能衰竭... → 普通文本）
    text = downgrade_text_item_headings(text)
    
    # 4. 移除重复标题（FinixDoc常见问题：标题重复输出导致级别偏移）
    text = remove_duplicate_title(text)
    
    # 5. V7: 基于编号模式直接分配标题级别
    text = normalize_heading_levels_per_pattern(text)
    
    # 6. 归一化标题文本
    text = normalize_heading_text(text)
    
    # 7. V7新增: 归一化文本内容
    text = normalize_text_content(text)
    
    return text
