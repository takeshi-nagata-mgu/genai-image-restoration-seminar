"""ノートブックをローカルで実行テストする（結果は notebooks_executed/ に保存）。
使い方: python run_test.py 01 03 ...   （番号省略時は全部）
キャッシュ類はすべて D:\\seminar_env 配下に置く。
"""
import glob
import os
import sys
import time

os.environ.setdefault("HF_HOME", r"D:\seminar_env\hf_cache")
os.environ.setdefault("TORCH_HOME", r"D:\seminar_env\torch_cache")

import nbformat
from nbclient import NotebookClient

HERE = os.path.dirname(os.path.abspath(__file__))
NB = os.path.join(HERE, "..", "notebooks")
OUT = os.path.join(HERE, "..", "notebooks_executed")
RUN_DIR = r"D:\seminar_env"   # weights/ はここに作られる

os.makedirs(OUT, exist_ok=True)
keys = sys.argv[1:]
for path in sorted(glob.glob(os.path.join(NB, "*.ipynb"))):
    name = os.path.basename(path)
    if keys and not any(name.startswith(k) for k in keys):
        continue
    nb = nbformat.read(path, as_version=4)
    t = time.time()
    client = NotebookClient(nb, timeout=3600, kernel_name="python3",
                            resources={"metadata": {"path": RUN_DIR}})
    try:
        client.execute()
        status = "OK"
    except Exception as e:
        status = "FAIL: " + str(e).splitlines()[0][:300]
        print(str(e)[-3000:])
    nbformat.write(nb, os.path.join(OUT, name))
    print(f"{name}: {status} ({time.time() - t:.0f}s)", flush=True)
