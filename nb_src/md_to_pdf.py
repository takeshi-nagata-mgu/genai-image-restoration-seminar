"""Markdown のガイドを PDF にする（受講者が開きやすいように）。
使い方: python md_to_pdf.py <入力.md> <出力.pdf> [相対リンクの基準 URL]
Markdown → HTML（python-markdown）→ Microsoft Edge のヘッドレス印刷で PDF。Windows 用。
基準 URL（例 https://github.com/<アカウント>/<リポジトリ>/blob/main/）を渡すと、相対リンクを GitHub 上のファイルへのリンクにする
（渡さないと、PDF 内のリンクは作業用の一時フォルダを指してしまう）。
"""
import os
import subprocess
import sys
import tempfile

import markdown

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'BIZ UDPGothic', 'Yu Gothic', sans-serif; font-size: 10.5pt; line-height: 1.6; color: #1F2329; }
h1 { font-size: 18pt; border-bottom: 3px solid #E39B2D; padding-bottom: 4px; }
h2 { font-size: 14pt; border-left: 6px solid #1E8C8C; padding-left: 8px; margin-top: 22px; }
h3 { font-size: 12pt; color: #17202B; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; page-break-inside: avoid; }
th { background: #17202B; color: #fff; text-align: left; }
th, td { border: 1px solid #D5DAE1; padding: 4px 6px; vertical-align: top; }
tr:nth-child(even) td { background: #F2F4F7; }
code { font-family: Consolas, 'BIZ UDGothic', monospace; background: #F2F4F7; padding: 0 3px; font-size: 9pt; }
pre { background: #1E232B; color: #E6E9EE; padding: 8px 10px; border-radius: 6px; white-space: pre-wrap; word-break: break-all; page-break-inside: avoid; }
pre code { background: none; color: inherit; padding: 0; }
blockquote { border-left: 4px solid #E39B2D; margin: 8px 0; padding: 4px 12px; background: #FFF4E0; }
hr { border: none; border-top: 1px solid #D5DAE1; margin: 16px 0; }
a { color: #1E6FB8; }
"""


def convert(src, dst, base_url=None):
    text = open(src, encoding="utf-8").read()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    title = os.path.splitext(os.path.basename(src))[0]
    base = f"<base href='{base_url}'>" if base_url else ""
    html = f"<!doctype html><html lang='ja'><head><meta charset='utf-8'>{base}<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>"
    with tempfile.TemporaryDirectory() as td:
        h = os.path.join(td, "doc.html")
        open(h, "w", encoding="utf-8").write(html)
        out = os.path.abspath(dst)
        subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={out}",
                        "file:///" + h.replace("\\", "/")], check=True, capture_output=True)
    print("wrote", dst)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
