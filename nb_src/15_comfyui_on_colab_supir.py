# %% [markdown]
# # 15. ComfyUI on Colab：SUPIR による高品質な画像復元（有料枠向け）
#
# **セミナー対応章**：第3章（1c）・実践ガイド　｜　**推奨ランタイム**：**A100 ＋ ハイメモリ（有料）**／L4 ＋ ハイメモリでも可　｜　**所要時間**：初回 約20〜30分（モデル 約12GB のダウンロードを含む）
#
# SUPIR は SDXL を土台にした大型の復元モデルで、**GPU メモリ 12GB 以上・システム RAM 32GB 以上** が必要です。
# 無料枠（T4・RAM 約12GB）では **RAM 不足で停止** するため、有料枠で実行してください。
#
# このノートブックでは：
# 1. Colab 上に **ComfyUI をインストール**（画面は使わず、裏で動かす）
# 2. SUPIR のカスタムノードとモデルをダウンロード
# 3. **API 形式のワークフロー** を Python から送って実行（ComfyUI の自動化・バッチ処理の方法）
#
# > **ローカルPC で ComfyUI を起動済みの場合**：下の `COMFY_URL` に `"http://127.0.0.1:8188"` を設定すると、インストールを飛ばして手元の ComfyUI で実行できます
# > （SUPIR ノード・モデルは手元の ComfyUI に入れておいてください）。
#
# ### ⚠ 有料枠を使う前に（重要）
# - ［ランタイム］→［ランタイムのタイプを変更］で **A100 GPU**（または L4）を選び、**「ハイメモリ」をオン**
# - 接続後、右上の［リソース］で **コンピューティングユニットの消費レートと残高** を確認
# - 使い終わったら **［ランタイム］→［ランタイムを接続解除して削除］**（接続している間は CU が減り続けます）
# - **SUPIR は非商用ライセンス** です。業務利用の前にライセンスを確認してください

# %% [setup]

# %%
COMFY_URL = os.environ.get("COMFY_URL")   # ローカルの ComfyUI を使う場合は "http://127.0.0.1:8188" などを設定
COMFY_DIR = "/content/ComfyUI" if IN_COLAB else os.path.abspath("ComfyUI")

try:
    import psutil
    ram = psutil.virtual_memory().total / 1024**3
    print(f"システムRAM: {ram:.0f} GB")
    if COMFY_URL is None and ram < 30:
        print("⚠ RAM が不足しています。［ランタイムのタイプを変更］で「ハイメモリ」をオンにしてください。")
except ImportError:
    pass

# %% [markdown]
# ## 1. ComfyUI と SUPIR ノードのインストール（Colab・初回のみ 約3分）

# %%
import json, shutil

def sh(cmd, cwd=None):
    print("$", cmd)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

if COMFY_URL is None:
    if not os.path.exists(COMFY_DIR):
        sh(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY_DIR}")
        sh(f"{sys.executable} -m pip install -q -r requirements.txt", cwd=COMFY_DIR)
    nodes_dir = os.path.join(COMFY_DIR, "custom_nodes")
    if not os.path.exists(os.path.join(nodes_dir, "ComfyUI-SUPIR")):
        sh("git clone --depth 1 https://github.com/kijai/ComfyUI-SUPIR", cwd=nodes_dir)
        sh(f"{sys.executable} -m pip install -q -r ComfyUI-SUPIR/requirements.txt", cwd=nodes_dir)
    # 講座の劣化作成ノード（縮小＋ノイズ、ガウスぼかし）を書き出す
    sem = os.path.join(nodes_dir, "seminar_nodes")
    os.makedirs(sem, exist_ok=True)
    with open(os.path.join(sem, "__init__.py"), "w", encoding="utf-8") as f:
        f.write('''
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class SRDegradationNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
            "scale_factor": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05}),
            "multiple_of": ("INT", {"default": 16, "min": 1, "max": 64}),
            "noise_sigma": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
            "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "enabled": (["true", "false"],)}}
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "target_width", "target_height")
    FUNCTION = "degrade"
    CATEGORY = "image"
    def degrade(self, image, scale_factor, multiple_of, noise_sigma, noise_seed, enabled):
        B, H, W, C = image.shape
        if enabled == "false":
            return (image, W * 4, H * 4)
        nw = max(multiple_of, round(W * scale_factor / multiple_of) * multiple_of)
        nh = max(multiple_of, round(H * scale_factor / multiple_of) * multiple_of)
        x = TF.resize(image.permute(0, 3, 1, 2), [nh, nw], interpolation=TF.InterpolationMode.BICUBIC, antialias=True).clamp(0, 1)
        if noise_sigma > 0:
            g = torch.Generator(device=x.device).manual_seed(int(noise_seed) % 2147483647)
            x = (x + torch.randn(x.shape, generator=g, device=x.device) * noise_sigma / 255.0).clamp(0, 1)
        return (x.permute(0, 2, 3, 1).contiguous(), nw * 4, nh * 4)

class CleanGaussianBlurNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",),
            "kernel_size": ("INT", {"default": 5, "min": 3, "max": 25, "step": 2}),
            "sigma": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1})}}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_blur"
    CATEGORY = "image/filters"
    def apply_blur(self, image, kernel_size, sigma):
        k = kernel_size + (kernel_size % 2 == 0)
        C = image.shape[-1]
        g = torch.arange(k, device=image.device, dtype=image.dtype) - (k - 1) / 2
        g = torch.exp(-g**2 / (2 * sigma**2)); g = g / g.sum()
        w = (g[:, None] * g[None]).expand(C, 1, k, k).contiguous()
        x = F.conv2d(image.permute(0, 3, 1, 2), w, groups=C, padding=k // 2).clamp(0, 1)
        return (x.permute(0, 2, 3, 1).contiguous(),)

NODE_CLASS_MAPPINGS = {"SRDegradationNode": SRDegradationNode, "CleanGaussianBlurNode": CleanGaussianBlurNode}
''')
    print("インストール完了")

# %% [markdown]
# ## 2. モデルのダウンロード（Colab・初回のみ 約10〜20分、合計 約12GB）
# - `sd_xl_base_1.0.safetensors`（SDXL 本体, 6.9GB）
# - `SUPIR-v0Q_fp16.safetensors`（SUPIR, 約2.7GB。v0Q は汎用、v0F は軽度の劣化向け）

# %%
from huggingface_hub import hf_hub_download

if COMFY_URL is None:
    ckpt = os.path.join(COMFY_DIR, "models", "checkpoints")
    for repo, fn in [("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
                     ("Kijai/SUPIR_pruned", "SUPIR-v0Q_fp16.safetensors")]:
        if not os.path.exists(os.path.join(ckpt, fn)):
            t = time.time()
            hf_hub_download(repo, fn, local_dir=ckpt)
            print(f"{fn}: {time.time()-t:.0f}s")
    print(os.listdir(ckpt))

# %% [markdown]
# ## 3. ComfyUI サーバーを裏で起動

# %%
import urllib.request, urllib.parse

if COMFY_URL is None:
    log = open(os.path.join(COMFY_DIR, "comfyui.log"), "w")
    server = subprocess.Popen([sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188"],
                              cwd=COMFY_DIR, stdout=log, stderr=subprocess.STDOUT)
    COMFY_URL = "http://127.0.0.1:8188"
    for _ in range(120):
        try:
            urllib.request.urlopen(COMFY_URL + "/system_stats", timeout=2); break
        except Exception:
            time.sleep(2)
print("ComfyUI:", json.load(urllib.request.urlopen(COMFY_URL + "/system_stats"))["system"]["comfyui_version"])

# %% [markdown]
# > **（参考）ComfyUI の画面を Colab で開く**：次のセルのコメントを外して実行すると、別タブで ComfyUI の画面を操作できます。
# > 配布の `sr_SUPIR.json` などをドラッグ＆ドロップすれば、ローカルと同じように使えます。

# %%
# from google.colab.output import serve_kernel_port_as_window
# serve_kernel_port_as_window(8188)

# %% [markdown]
# ## 4. API でワークフローを実行する関数
# ComfyUI の画面で **Workflow → Export (API)** で保存した JSON（＝ノードIDごとの入力値の辞書）を `/prompt` に送ると実行されます。
# バッチ処理（フォルダ内の全画像を処理など）はこの方法で自動化できます。

# %%
def upload_image(pil_img, name):
    """画像を ComfyUI の input フォルダに送る（/upload/image API）"""
    import io, uuid
    buf = io.BytesIO(); pil_img.save(buf, format="PNG")
    boundary = uuid.uuid4().hex
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\n"
            f"Content-Type: image/png\r\n\r\n").encode() + buf.getvalue() + \
           f"\r\n--{boundary}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request(COMFY_URL + "/upload/image", data=body,
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    return json.load(urllib.request.urlopen(req))["name"]

def run_workflow(workflow, timeout=1800):
    """ワークフローを実行し、保存された画像を {ノードID: [PIL画像, ...]} で返す"""
    req = urllib.request.Request(COMFY_URL + "/prompt", data=json.dumps({"prompt": workflow}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req))["prompt_id"]
    t = time.time()
    while time.time() - t < timeout:
        hist = json.load(urllib.request.urlopen(f"{COMFY_URL}/history/{pid}"))
        if pid in hist:
            st = hist[pid].get("status", {})
            if st.get("status_str") == "error":
                raise RuntimeError(json.dumps(st.get("messages"), ensure_ascii=False)[:2000])
            outs = {}
            for nid, o in hist[pid]["outputs"].items():
                for im in o.get("images", []):
                    q = urllib.parse.urlencode({"filename": im["filename"], "subfolder": im["subfolder"], "type": im["type"]})
                    import io
                    outs.setdefault(nid, []).append(Image.open(io.BytesIO(urllib.request.urlopen(f"{COMFY_URL}/view?{q}").read())))
            print(f"完了: {time.time()-t:.0f}s")
            return outs
        time.sleep(2)
    raise TimeoutError("ComfyUI の処理がタイムアウトしました")

# %% [markdown]
# ## 5. SUPIR ワークフロー（API 形式）
# 配布の `sr_SUPIR.json`（講師が論文の図4 の作成に使ったもの）を API 形式に変換したものです。
# 正解画像 → **劣化（1/4 縮小＋ノイズ σ=15）** → ×4 拡大 → SUPIR で復元、の流れです。
# （元のワークフローは大きな画像用に最初に 1/2 縮小していますが、ここでは 512px の画像を使うので縮小なし＝ scale_factor 1.0 にしています）

# %%
SUPIR_WORKFLOW = {
 "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}},
 "53": {"class_type": "SRDegradationNode", "inputs": {"image": ["1", 0], "scale_factor": 1.0, "multiple_of": 16, "noise_sigma": 0, "noise_seed": 11, "enabled": "true"}},
 "2": {"class_type": "SRDegradationNode", "inputs": {"image": ["53", 0], "scale_factor": 0.25, "multiple_of": 16, "noise_sigma": 15, "noise_seed": 111, "enabled": "true"}},
 "77": {"class_type": "CleanGaussianBlurNode", "inputs": {"image": ["2", 0], "kernel_size": 5, "sigma": 0.1}},
 "41": {"class_type": "ImageScaleBy", "inputs": {"image": ["77", 0], "upscale_method": "lanczos", "scale_by": 4}},
 "40": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base_1.0.safetensors"}},
 "42": {"class_type": "SUPIR_model_loader_v2", "inputs": {"model": ["40", 0], "clip": ["40", 1], "vae": ["40", 2],
        "supir_model": "SUPIR-v0Q_fp16.safetensors", "fp8_unet": False, "diffusion_dtype": "auto", "high_vram": False}},
 "43": {"class_type": "SUPIR_first_stage", "inputs": {"SUPIR_VAE": ["42", 1], "image": ["41", 0], "use_tiled_vae": True,
        "encoder_tile_size": 512, "decoder_tile_size": 512, "encoder_dtype": "auto"}},
 "44": {"class_type": "SUPIR_conditioner", "inputs": {"SUPIR_model": ["42", 0], "latents": ["43", 2],
        "positive_prompt": "high quality, detailed, sharp, 8k", "negative_prompt": "blurry, noise, low quality, artifacts"}},
 "45": {"class_type": "SUPIR_sample", "inputs": {"SUPIR_model": ["42", 0], "latents": ["43", 2], "positive": ["44", 0], "negative": ["44", 1],
        "seed": 1234, "steps": 30, "cfg_scale_start": 4.0, "cfg_scale_end": 1.0, "EDM_s_churn": 5, "s_noise": 1.003, "DPMPP_eta": 1.0,
        "control_scale_start": 1.0, "control_scale_end": 1.0, "restore_cfg": -1.0, "keep_model_loaded": True,
        "sampler": "RestoreDPMPP2MSampler", "sampler_tile_size": 1024, "sampler_tile_stride": 512}},
 "46": {"class_type": "SUPIR_decode", "inputs": {"SUPIR_VAE": ["43", 0], "latents": ["45", 0], "use_tiled_vae": True,
        "decoder_tile_size": 512, "decoder_dtype": "auto"}},
 "50": {"class_type": "SaveImage", "inputs": {"images": ["41", 0], "filename_prefix": "00_Degraded_Input_x4"}},
 "47": {"class_type": "SaveImage", "inputs": {"images": ["46", 0], "filename_prefix": "05_SUPIR"}},
 "54": {"class_type": "SaveImage", "inputs": {"images": ["53", 0], "filename_prefix": "GroundTruth"}},
}

# %% [markdown]
# ## 6. 実行（A100 で 1〜2分、初回はモデル読み込みでさらに数分）

# %%
src = load_sample("astronaut")          # 自分の画像を使う場合：Image.open("my.jpg").convert("RGB")
name = upload_image(src, "input.png")
wf = json.loads(json.dumps(SUPIR_WORKFLOW))
wf["1"]["inputs"]["image"] = name
outs = run_workflow(wf)
gt, lq, sr = outs["54"][0], outs["50"][0], outs["47"][0]
print("sizes:", gt.size, lq.size, sr.size)
show([lq, sr.resize(lq.size), gt.resize(lq.size)], ["degraded input (x4 lanczos)", "SUPIR", "ground truth"], size=4.5)
if sr.size == gt.size:
    print(f"PSNR  input {psnr(lq.resize(gt.size), gt):.2f} dB  →  SUPIR {psnr(sr, gt):.2f} dB")

# %% [markdown]
# **観察ポイント**：SUPIR の出力は見た目が大きく改善しますが、**PSNR は入力より下がることがあります**（講師の環境では 22.9 dB → 21.4 dB）。
# 細部（髪・布のしわ・ワッペンの文字）が「もっともらしく生成」されて正解と画素単位では一致しないためです（第4章：知覚–歪みトレードオフ、ハルシネーション）。

# %% [markdown]
# **自分の画像をそのまま高画質化する場合**（劣化させない）：`wf["53"]["inputs"]["enabled"] = "false"`、`wf["2"]["inputs"]["enabled"] = "false"` とし、
# `wf["41"]["inputs"]["scale_by"]` で拡大率を指定します（大きな画像は GPU メモリに注意）。
#
# **プロンプトで復元を誘導**：`wf["44"]["inputs"]["positive_prompt"]` に被写体の説明（例：`"a photo of a stone lantern in a japanese garden"`）を書くと品質が上がることがありますが、
# **テキストに引きずられたハルシネーション** も起きやすくなります（第4章）。

# %% [markdown]
# ## 7. 後片付け（有料枠では必ず）
# 結果を残したい場合は Google Drive に保存してから、**［ランタイム］→［ランタイムを接続解除して削除］** を実行してください。

# %%
os.makedirs("results", exist_ok=True)
sr.save("results/supir_result.png")
print("保存しました: results/supir_result.png（Colab では左のファイル一覧からダウンロード）")
