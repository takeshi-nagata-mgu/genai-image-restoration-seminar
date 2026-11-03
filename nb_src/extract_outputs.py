"""実行済みノートブックから画像出力とテキスト出力を取り出す（スライド用素材・結果確認用）。
使い方: python extract_outputs.py 03 05   → slides/figs/nb03_c012_0.png などに保存し、テキスト出力を表示
"""
import base64
import glob
import os
import sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "notebooks_executed")
DST = os.path.join(HERE, "..", "..", "slides", "figs")
os.makedirs(DST, exist_ok=True)

keys = sys.argv[1:]
for path in sorted(glob.glob(os.path.join(SRC, "*.ipynb"))):
    name = os.path.basename(path)
    if keys and not any(name.startswith(k) for k in keys):
        continue
    nb = nbformat.read(path, as_version=4)
    print(f"===== {name}")
    for ci, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        k = 0
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                print(f"[c{ci:03d}] ERROR {out.get('ename')}: {out.get('evalue')[:300]}")
            if out.get("output_type") == "stream" and out.get("name") == "stdout":
                print(f"[c{ci:03d}] " + out["text"].strip()[:1500])
            data = out.get("data", {})
            if "image/png" in data:
                fn = f"nb{name[:2]}_c{ci:03d}_{k}.png"
                with open(os.path.join(DST, fn), "wb") as f:
                    f.write(base64.b64decode(data["image/png"]))
                print(f"[c{ci:03d}] image -> {fn}")
                k += 1
