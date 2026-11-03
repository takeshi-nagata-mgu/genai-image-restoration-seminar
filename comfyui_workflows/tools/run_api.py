"""API 形式のワークフローを ComfyUI サーバーに送って実行する（バッチ処理の例）。
使い方:
  python run_api.py upscale_realesrgan.json                      # そのまま実行
  python run_api.py upscale_realesrgan.json --image a.png b.png   # ノード "1"（LoadImage）の画像を差し替えて順に実行
  オプション --server http://127.0.0.1:8188
画像は ComfyUI の input フォルダに置いておくこと。結果は output フォルダに保存される。
"""
import argparse
import json
import time
import urllib.error
import urllib.request


def run(server, wf, timeout=1800):
    req = urllib.request.Request(server + "/prompt", data=json.dumps({"prompt": wf}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    except urllib.error.HTTPError as e:
        return False, e.read().decode()[:2000]
    t = time.time()
    while time.time() - t < timeout:
        h = json.load(urllib.request.urlopen(f"{server}/history/{pid}"))
        if pid in h:
            st = h[pid]["status"]
            if st["status_str"] == "error":
                msg = [m for m in st["messages"] if m[0] == "execution_error"]
                return False, json.dumps(msg[0][1].get("exception_message") if msg else st, ensure_ascii=False)[:2000]
            files = [i["filename"] for o in h[pid]["outputs"].values() for i in o.get("images", [])]
            return True, f"{time.time() - t:.0f}s {files}"
        time.sleep(1)
    return False, "timeout"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("workflow")
    ap.add_argument("--image", nargs="*")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    wf = json.load(open(a.workflow, encoding="utf-8"))
    for img in (a.image or [None]):
        if img:
            wf["1"]["inputs"]["image"] = img
        ok, msg = run(a.server, wf)
        print(("OK  " if ok else "ERR ") + (img or a.workflow), msg)
