# %% [markdown]
# # 11. マルチモーダル：「言葉」で画像復元をコントロールする
#
# **セミナー対応章**：第5章（1）（3b）（4a）　｜　**推奨ランタイム**：Colab無料枠 T4（ぎりぎり）／ L4 推奨　｜　**所要時間**：約6分（モデルDL 約10GB を含まず）
#
# 1. **InstructPix2Pix**：「ノイズを消して」「明るくして」など指示文で画像を編集する拡散モデル
# 2. **VLM（視覚言語モデル, Qwen2.5-VL-3B）** に画像の劣化を診断させる
# 3. **AIオーケストレーション**：VLM が日本語の指示と画像を見て「どの復元ツールを、どの順で、どの強さで使うか」を計画 → Python が実行
# 4. **注意**：VLM による「文字の推測」はハルシネーションの温床（防犯カメラの文字復元の危うさ）
#
# > さらに高性能な指示型編集（FLUX.1 Kontext, Qwen-Image-Edit など）は 20GB 以上の GPU メモリが必要です。ローカルの大型GPU か Colab 有料枠（A100 / G4）＋ ComfyUI で試してください。

# %% [setup]

# %%
import cv2, json, re
import spandrel
from diffusers import StableDiffusionInstructPix2PixPipeline

# 劣化させたテスト画像：ブレ + ノイズ + 右側が暗い
x = to_np(load_sample("coffee").resize((512, 344)))
H, W = x.shape[:2]
k = np.zeros((9, 9), np.float32); k[4, :] = 1 / 9
y = cv2.filter2D(x, -1, k)
y = y * np.linspace(1.0, 0.35, W)[None, :, None]                    # 右側ほど暗い
y = np.clip(y + np.random.default_rng(0).normal(0, 0.04, y.shape), 0, 1).astype(np.float32)
degraded = to_pil(y)
show([to_pil(x), degraded], ["original", "degraded: blur + noise + dark right side"], size=4.5)

# %% [markdown]
# ## 1. InstructPix2Pix：指示文で編集する拡散モデル
# 「元画像」と「指示文」の2つを条件として生成します。`image_guidance_scale` を上げると元画像に忠実に、`guidance_scale` を上げると指示に忠実になります。

# %%
ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
ip2p.set_progress_bar_config(disable=True)
instructions = ["remove the noise", "make the right side brighter", "make it sharp and clear"]
edits = [ip2p(ins, image=degraded, num_inference_steps=30, image_guidance_scale=1.5, guidance_scale=7,
              generator=torch.Generator(DEVICE).manual_seed(0)).images[0] for ins in instructions]
show([degraded] + edits, ["input"] + instructions, size=3.5)
del ip2p; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# %% [markdown]
# **観察ポイント**：「明るくして」のような大域的な編集は得意ですが、ノイズ除去・ブレ除去のような **画素レベルの忠実な復元** は苦手で、内容を描き変えてしまうことがあります。
# → 指示の「理解」は言語モデルに任せ、実際の復元は専用モデルに任せる **役割分担（オーケストレーション）** が現実的です。

# %% [markdown]
# ## 2. VLM に劣化を診断させる（Qwen2.5-VL-3B-Instruct, 約7.5GB）

# %%
from transformers import AutoProcessor, AutoModelForImageTextToText

VLM = "Qwen/Qwen2.5-VL-3B-Instruct"
processor = AutoProcessor.from_pretrained(VLM)
vlm = AutoModelForImageTextToText.from_pretrained(VLM, torch_dtype=DTYPE, device_map=DEVICE)

def ask(image, question, max_new_tokens=300):
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

print(ask(degraded, "この画像の画質の問題点（ボケ、ブレ、ノイズ、明るさのムラなど）を日本語で箇条書きで診断してください。"))
gpu_mem()

# %% [markdown]
# ## 3. AIオーケストレーション：日本語の指示 → VLM が計画 → ツールを実行
# ユーザーの指示「このブレを消して、右側を明るくして」と画像を VLM に渡し、使えるツールの一覧から **実行計画（JSON）** を作らせます。

# %%
scunet = spandrel.ModelLoader().load_from_file(
    download("https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth")).to(DEVICE).eval()

def tool_denoise(img, strength=1.0):
    with torch.no_grad():
        d = scunet(to_tensor(img).to(DEVICE))[0].permute(1, 2, 0).cpu().numpy()
    return np.clip(strength * d + (1 - strength) * img, 0, 1)

def tool_deblur(img, strength=1.0):
    blur = cv2.GaussianBlur(img, (0, 0), 2.0)                       # アンシャープマスク（簡易）
    return np.clip(img + strength * 1.5 * (img - blur), 0, 1)

def tool_brighten(img, region="all", gain=1.5):
    w = img.shape[1]
    ramp = {"right": np.linspace(1, gain, w), "left": np.linspace(gain, 1, w)}.get(region, np.full(w, gain))
    return np.clip(img * ramp[None, :, None], 0, 1)

TOOLS = {"denoise": tool_denoise, "deblur": tool_deblur, "brighten": tool_brighten}
TOOL_DOC = """使えるツール:
- denoise(strength: 0〜1)            : ノイズ除去
- deblur(strength: 0〜1)             : ブレ・ボケの軽減
- brighten(region: "all"|"left"|"right", gain: 1.0〜3.0) : 明るさ補正
"""

def plan(image, instruction):
    q = (f"あなたは画像復元の専門家です。ユーザーの指示: 「{instruction}」\n{TOOL_DOC}\n"
         "画像を見て、指示を実現するためのツールの実行順序とパラメータを、次のJSON形式だけで出力してください。"
         "ノイズがある画像では deblur の前に denoise を実行してください。\n"
         '{"steps": [{"tool": "...", "args": {...}}, ...]}')
    text = ask(image, q)
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"steps": []}, text

instruction = "このブレを消して、右側を明るくして"
p, raw = plan(degraded, instruction)
print("VLMの計画:", json.dumps(p, ensure_ascii=False, indent=1))

img = y.copy()
trace = [degraded]
for s in p["steps"]:
    fn = TOOLS.get(s.get("tool"))
    if fn is None:
        print("未知のツールをスキップ:", s); continue
    img = fn(img, **s.get("args", {}))
    trace.append(to_pil(img))
show(trace, ["input"] + [f"{i+1}. {s['tool']}" for i, s in enumerate(p["steps"]) if s.get("tool") in TOOLS], size=3.5)

# %% [markdown]
# ## 4. 注意：VLM による「文字の推測」はハルシネーションになりうる
# 防犯カメラ映像の文字（ナンバープレートなど）を VLM や生成AIで「読める形に復元」すると、**もっともらしい別の文字** が出ることがあります。

# %%
from PIL import ImageDraw, ImageFont
plate = Image.new("RGB", (320, 100), "white")
d = ImageDraw.Draw(plate)
try:
    font = ImageFont.truetype("DejaVuSans-Bold.ttf", 56)
except OSError:
    font = ImageFont.load_default(size=56)
d.text((20, 18), "8 3 - 6 1", fill="black", font=font)
sizes = [(40, 12), (28, 9), (20, 6), (14, 4)]                                       # だんだん解像度を下げる
tinies = [plate.resize(sz, Image.BILINEAR).resize((320, 100), Image.BICUBIC) for sz in sizes]
show([plate] + tinies, ["truth: 83-61"] + [f"{w}x{h} px" for w, h in sizes], size=3)
for sz, im in zip(sizes, tinies):
    print(f"{sz[0]}x{sz[1]} px → VLMの読み取り:", ask(im, "この画像に書かれている数字を読んでください。数字だけを答えてください。", 20))

# %% [markdown]
# **観察ポイント**：ある程度の解像度までは正しく読めますが、情報が足りなくなると **自信ありげに別の数字を答える** ようになります（読めない、とは言わないことが多い）。
# 情報が失われた画像から読み取った（あるいは生成AIで復元した）文字は「証拠」にはなりません。
# 文脈を理解する復元は強力ですが、**出力が観測から決まるのか、推測なのか** を常に区別し、推測部分は明示することが重要です。

# %% [markdown]
# ## まとめ
# - テキストで復元を指示する時代：InstructPix2Pix → FLUX Kontext / Qwen-Image-Edit / SUPIR（テキスト条件付き復元）
# - 実務的には **VLM が診断・計画し、専用の復元モデルを呼び出す「オーケストレーション」** が堅実
# - 文字や顔など「意味」を推測する復元は、ハルシネーションと真正性に特に注意
