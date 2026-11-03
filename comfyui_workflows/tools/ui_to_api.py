"""ComfyUI の UI 形式ワークフロー（画面で保存した .json）を API 形式に変換する。
起動中の ComfyUI の /object_info を使って、ウィジェット値を入力名に対応づける。

使い方: python ui_to_api.py workflow.json workflow_api.json [--server http://127.0.0.1:8188]
"""
import argparse
import json
import urllib.request

CONTROL_VALUES = {"fixed", "increment", "decrement", "randomize"}
WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}


def is_widget(spec):
    t = spec[0]
    return isinstance(t, list) or t in WIDGET_TYPES


def convert(ui, info):
    links = {l[0]: l for l in ui["links"]}
    nodes = {n["id"]: n for n in ui["nodes"]}

    def resolve(link_id):
        """Reroute ノードをたどって実際の出力元 (node_id, slot) を返す"""
        _, src, slot = links[link_id][:3]
        while nodes[src]["type"] == "Reroute":
            src, slot = links[nodes[src]["inputs"][0]["link"]][1:3]
        return src, slot

    api = {}
    for n in ui["nodes"]:
        if n["type"] in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode") or n.get("mode", 0) in (2, 4):
            continue
        spec = info[n["type"]]["input"]
        order = list(spec.get("required", {}).items()) + list(spec.get("optional", {}).items())
        linked = {i["name"]: i["link"] for i in n.get("inputs", []) if i.get("link") is not None}
        widgets = list(n.get("widgets_values") or [])
        if isinstance(n.get("widgets_values"), dict):           # 一部ノードは dict 形式
            widgets = []
        inputs, wi = {}, 0
        for name, sp in order:
            if name in linked:
                src, slot = resolve(linked[name])
                inputs[name] = [str(src), slot]
                if is_widget(sp) and any(i["name"] == name and "widget" in i for i in n.get("inputs", [])):
                    wi += 1                                        # 入力に変換されたウィジェットも値を持つ
                    if wi < len(widgets) and widgets[wi] in CONTROL_VALUES:
                        wi += 1
                continue
            if not is_widget(sp) or wi >= len(widgets):
                continue
            inputs[name] = widgets[wi]
            wi += 1
            if wi < len(widgets) and isinstance(widgets[wi], str) and widgets[wi] in CONTROL_VALUES:
                wi += 1                                            # seed の後の control_after_generate
        if isinstance(n.get("widgets_values"), dict):
            inputs.update({k: v for k, v in n["widgets_values"].items() if k in dict(order) and k not in inputs})
        api[str(n["id"])] = {"class_type": n["type"], "inputs": inputs, "_meta": {"title": n.get("title", n["type"])}}
    return api


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--server", default="http://127.0.0.1:8188")
    a = ap.parse_args()
    info = json.load(urllib.request.urlopen(a.server + "/object_info"))
    ui = json.load(open(a.src, encoding="utf-8"))
    json.dump(convert(ui, info), open(a.dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("wrote", a.dst)
