"""nb_src/*.py（# %% 形式）から notebooks/*.ipynb を生成する。

使い方:  python build_notebooks.py
  - "# %% [markdown]" で始まるセルは Markdown セル（各行の先頭 "# " を除去）
  - "# %%" で始まるセルはコードセル
  - "# %% [setup]" は共通セットアップ (_common_setup.py) に置換される
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "notebooks")


def to_source(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return [l + "\n" for l in lines[:-1]] + [lines[-1]] if lines else []


def parse(path, common):
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        if kind == "markdown":
            lines = [l[2:] if l.startswith("# ") else l.lstrip("#") for l in buf]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": to_source(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                          "outputs": [], "source": to_source(list(buf))})

    with open(path, encoding="utf-8") as f:
        for line in f.read().splitlines():
            if line.startswith("# %%"):
                flush()
                buf = []
                tag = line[4:].strip()
                if tag == "[markdown]":
                    kind = "markdown"
                elif tag == "[setup]":
                    kind = "code"
                    buf = list(common)
                else:
                    kind = "code"
                continue
            buf.append(line)
    flush()
    for i, c in enumerate(cells):
        c["id"] = f"cell-{i:03d}"
    return cells


def main():
    with open(os.path.join(HERE, "_common_setup.py"), encoding="utf-8") as f:
        common = f.read().splitlines()
    os.makedirs(OUT, exist_ok=True)
    for src in sorted(glob.glob(os.path.join(HERE, "[0-9][0-9]_*.py"))):
        nb = {
            "cells": parse(src, common),
            "metadata": {
                "accelerator": "GPU",
                "colab": {"provenance": [], "gpuType": "T4"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        name = os.path.splitext(os.path.basename(src))[0] + ".ipynb"
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
        print("wrote", name, len(nb["cells"]), "cells")


if __name__ == "__main__":
    main()
