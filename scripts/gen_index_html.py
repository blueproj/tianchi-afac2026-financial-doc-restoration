"""
生成 pipeline 输出目录的汇总 index.html，方便点击跳转查看各样本结果。

用法:
  python scripts/gen_index_html.py --dir output/pipeline_B
"""
import os
import json
import re
import html as _html
import argparse
from datetime import datetime


def render_final_md(md: str) -> str:
    """把 final.md 渲染为可视 HTML：<table> 原样保留（浏览器直接渲染），
    其余文本按行转 <p>。"""
    parts = re.split(r'(<table[^>]*>.*?</table>)', md, flags=re.S | re.I)
    body = []
    for part in parts:
        if re.match(r'\s*<table', part, re.I):
            body.append(f'<div class="tblwrap">{part}</div>')
        else:
            for line in part.split('\n'):
                line = line.strip()
                if line:
                    body.append(f'<p>{_html.escape(line)}</p>')
    return '\n'.join(body)


def write_final_view(sid_dir: str, sid: str, image: str):
    """在样本目录内生成 final_view.html（渲染现有 final.md，不重跑 pipeline）。"""
    fp = os.path.join(sid_dir, 'final.md')
    try:
        md = open(fp, encoding='utf-8').read()
    except Exception:
        return
    page = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<title>final.md 渲染 - {sid}</title><style>
body{{font-family:-apple-system,'Segoe UI',sans-serif;margin:20px;background:#fafafa}}
h2{{color:#333;margin-bottom:2px}} p{{margin:4px 0;color:#333}}
.tblwrap{{overflow-x:auto;margin:12px 0;border:1px solid #e0e0e0;border-radius:4px}}
table{{border-collapse:collapse}}
td,th{{border:1px solid #bbb;padding:2px 7px;font-size:11px;white-space:nowrap;text-align:center}}
table tr:first-child{{background:#eef6ff;font-weight:500}}
a.back{{color:#1a73e8;text-decoration:none;font-size:13px}}
</style></head><body>
<a class="back" href="../index.html">&larr; 返回汇总</a>
<h2>{sid} — final.md 渲染结果</h2>
<p style="color:#888;font-size:12px">{_html.escape(image)}</p><hr>
{render_final_md(md)}
</body></html>"""
    with open(os.path.join(sid_dir, 'final_view.html'), 'w', encoding='utf-8') as f:
        f.write(page)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='pipeline输出目录')
    args = parser.parse_args()

    pipeline_dir = args.dir
    samples = []

    for sid in sorted(os.listdir(pipeline_dir)):
        sid_dir = os.path.join(pipeline_dir, sid)
        if not os.path.isdir(sid_dir):
            continue
        summary_path = os.path.join(sid_dir, 'summary.json')
        info = {'sid': sid, 'doc_type': '?', 'markdown_len': 0, 'elapsed_s': 0,
                'image': '', 'n_regions': 0, 'n_tables': 0, 'n_chunks': 0}
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                d = json.load(f)
            info['doc_type'] = d.get('doc_type', '?')
            info['markdown_len'] = d.get('markdown_len', 0)
            info['elapsed_s'] = d.get('elapsed_s', 0)
            info['image'] = d.get('image', '')
            if info['doc_type'] == 'table':
                info['n_regions'] = d.get('n_regions', 0)
                info['n_tables'] = d.get('n_tables', 0)
            else:
                info['n_chunks'] = d.get('n_chunks', 0)
        info['has_report'] = os.path.exists(os.path.join(sid_dir, 'report.html'))
        info['has_final'] = os.path.exists(os.path.join(sid_dir, 'final.md'))
        if info['has_final']:
            write_final_view(sid_dir, sid, info['image'])
        samples.append(info)

    n_table = sum(1 for s in samples if s['doc_type'] == 'table')
    n_long = sum(1 for s in samples if s['doc_type'] == 'long')
    total_time = sum(s['elapsed_s'] for s in samples)
    total_md = sum(s['markdown_len'] for s in samples)

    # 生成 HTML
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>B榜 Pipeline 结果汇总 - {os.path.basename(pipeline_dir)}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
h1 {{ color: #333; margin-bottom: 5px; }}
.stats {{ background: #fff; padding: 15px 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
.stats span {{ margin-right: 24px; font-size: 14px; }}
.stats .num {{ font-weight: bold; color: #1a73e8; font-size: 16px; }}
.filter-bar {{ margin-bottom: 15px; }}
.filter-bar button {{ padding: 6px 16px; margin-right: 8px; border: 1px solid #ddd; border-radius: 4px; background: #fff; cursor: pointer; font-size: 13px; }}
.filter-bar button.active {{ background: #1a73e8; color: #fff; border-color: #1a73e8; }}
table {{ width: 100%; border-collapse: collapse; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th {{ background: #f0f0f0; padding: 10px 12px; text-align: left; font-size: 13px; font-weight: 600; position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-top: 1px solid #eee; font-size: 13px; }}
tr:hover {{ background: #f8f9ff; }}
.type-table {{ color: #d93025; font-weight: 500; }}
.type-long {{ color: #1a73e8; font-weight: 500; }}
a {{ color: #1a73e8; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
.btn {{ display: inline-block; padding: 3px 10px; border-radius: 3px; font-size: 12px; margin-right: 4px; }}
.btn-report {{ background: #e8f0fe; color: #1a73e8; }}
.btn-final {{ background: #e6f4ea; color: #137333; }}
.btn-view {{ background: #e0f7fa; color: #00796b; }}
.btn-dir {{ background: #fef7e0; color: #b06000; }}
.md-len {{ font-family: monospace; }}
</style>
</head>
<body>
<h1>B榜 Pipeline 结果汇总</h1>
<p style="color:#666; margin-top:0;">目录: {pipeline_dir} | 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>

<div class="stats">
  <span>总样本: <span class="num">{len(samples)}</span></span>
  <span>表格: <span class="num">{n_table}</span></span>
  <span>长文档: <span class="num">{n_long}</span></span>
  <span>总耗时: <span class="num">{total_time:.0f}s ({total_time/60:.1f}min)</span></span>
  <span>总输出: <span class="num">{total_md/1024:.0f} KB</span></span>
</div>

<div class="filter-bar">
  <button class="active" onclick="filterType('all')">全部 ({len(samples)})</button>
  <button onclick="filterType('table')">表格 ({n_table})</button>
  <button onclick="filterType('long')">长文档 ({n_long})</button>
</div>

<table id="main-table">
<thead>
<tr>
  <th>#</th>
  <th>Sample ID</th>
  <th>类型</th>
  <th>文件名</th>
  <th>Regions/Chunks</th>
  <th>输出(字符)</th>
  <th>耗时(s)</th>
  <th>操作</th>
</tr>
</thead>
<tbody>
"""

    for i, s in enumerate(samples, 1):
        type_cls = f"type-{s['doc_type']}"
        type_label = '表格' if s['doc_type'] == 'table' else '长文档'
        detail = f"{s['n_regions']}区/{s['n_tables']}表" if s['doc_type'] == 'table' else f"{s['n_chunks']} chunks"
        fname = s.get('image', '')[:40]
        if len(s.get('image', '')) > 40:
            fname += '...'

        links = ''
        if s['has_report']:
            links += f'<a class="btn btn-report" href="{s["sid"]}/report.html" target="_blank">Report</a>'
        if s['has_final']:
            links += f'<a class="btn btn-view" href="{s["sid"]}/final_view.html" target="_blank">渲染MD</a>'
            links += f'<a class="btn btn-final" href="{s["sid"]}/final.md" target="_blank">Final.md</a>'
        links += f'<a class="btn btn-dir" href="{s["sid"]}/" target="_blank">目录</a>'

        html += f"""<tr data-type="{s['doc_type']}">
  <td>{i}</td>
  <td><strong>{s['sid']}</strong></td>
  <td class="{type_cls}">{type_label}</td>
  <td title="{s.get('image','')}">{fname}</td>
  <td>{detail}</td>
  <td class="md-len">{s['markdown_len']:,}</td>
  <td>{s['elapsed_s']:.1f}</td>
  <td>{links}</td>
</tr>
"""

    html += """</tbody>
</table>

<script>
function filterType(type) {
  document.querySelectorAll('.filter-bar button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('#main-table tbody tr').forEach(tr => {
    if (type === 'all' || tr.dataset.type === type) {
      tr.style.display = '';
    } else {
      tr.style.display = 'none';
    }
  });
}
</script>
</body>
</html>
"""

    out_path = os.path.join(pipeline_dir, 'index.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"生成完成: {out_path}")
    print(f"  {len(samples)} 个样本 (table={n_table}, long={n_long})")


if __name__ == '__main__':
    main()
