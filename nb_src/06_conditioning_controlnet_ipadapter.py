# %% [markdown]
# # 06. 条件付け（Conditioning）：img2img・ControlNet・IP-Adapter で生成AIを「縛る」
#
# **セミナー対応章**：第3章（4）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約7分（モデルDL 約6GB を含まず）
#
# 生成AIで復元するとき、自由に描かせすぎると **元画像の特徴（構造・色・質感・意味）** が失われます。
# 1. **img2img**：strength（どれだけノイズを足して描き直すか）で自由度を調整 → 構造が崩れるジレンマ
# 2. **ControlNet（Tile / Canny）**：構造で縛る
# 3. **IP-Adapter**：参照画像の「意味・質感」で縛る（CLIP画像埋め込みをクロスアテンションに注入）
# 4. **併用**：ControlNet × img2img × IP-Adapter
# 5. **実践例**：5段階パイプラインで低画質な人物写真を高品質化（各段階で画質が上がる様子を確認）

# %% [setup]

# %% [markdown]
# ## 0. 入力画像：低画質な画像（縮小→拡大でぼやけた画像）

# %%
import cv2
from diffusers import (StableDiffusionImg2ImgPipeline, StableDiffusionControlNetImg2ImgPipeline,
                       StableDiffusionControlNetPipeline, ControlNetModel)

hr = load_sample("astronaut")
lq = hr.resize((96, 96), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
PROMPT = "a high quality photo of an astronaut, detailed, sharp focus"
NEG = "blurry, low quality, cartoon, painting, deformed"
gen = lambda s=0: torch.Generator(DEVICE).manual_seed(s)
BASE = "stable-diffusion-v1-5/stable-diffusion-v1-5"
show([hr, lq], ["reference (GT)", "low quality input"], size=3.5)

# %% [markdown]
# ## 1. img2img：strength のジレンマ
# strength が小さい → ほとんど変わらない（ぼやけたまま）／大きい → きれいになるが **別人・別の構図** になる

# %%
i2i = StableDiffusionImg2ImgPipeline.from_pretrained(BASE, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
i2i.set_progress_bar_config(disable=True)
strengths = [0.2, 0.4, 0.6, 0.8]
outs = [i2i(PROMPT, image=lq, strength=s, negative_prompt=NEG, num_inference_steps=30, generator=gen()).images[0]
        for s in strengths]
show([lq] + outs, ["input"] + [f"img2img strength={s}" for s in strengths], size=3.2)
del i2i; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# %% [markdown]
# ## 2. ControlNet Tile：入力画像そのものを「構造の設計図」にする
# ControlNet は元の拡散モデルを凍結したまま、条件画像（エッジ・深度・ぼやけた画像など）から構造情報を注入する追加ネットワークです。
# **Tile** 版は「ぼやけた画像の構造は守りつつ細部を描き足す」ように学習されており、超解像・高品質化に向きます。

# %%
cn_tile = ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile", torch_dtype=DTYPE)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    BASE, controlnet=cn_tile, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
pipe.set_progress_bar_config(disable=True)

tile_outs = [pipe(PROMPT, image=lq, control_image=lq, strength=s, negative_prompt=NEG,
                  num_inference_steps=30, controlnet_conditioning_scale=1.0, generator=gen()).images[0]
             for s in (0.6, 0.8, 1.0)]
show([lq] + tile_outs, ["input"] + [f"ControlNet Tile, strength={s}" for s in (0.6, 0.8, 1.0)], size=3.2)
gpu_mem()

# %% [markdown]
# **観察ポイント**：img2img の strength=0.8 では構図が崩れましたが、ControlNet Tile では strength=1.0（完全にノイズから生成）でも構図・顔の向きが保たれます。

# %% [markdown]
# ## 3. ControlNet Canny：エッジ（輪郭）だけで縛る
# 輪郭は守り、色や質感は自由に生成させたい場合（例：線画・CAD図面から写真風画像、疑似欠陥画像の生成）に使います。

# %%
edges = cv2.Canny(np.array(hr), 100, 200)
edge_img = Image.fromarray(np.stack([edges] * 3, -1))
cn_canny = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=DTYPE)
canny_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE, controlnet=cn_canny, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
canny_pipe.set_progress_bar_config(disable=True)
canny_outs = [canny_pipe(p, image=edge_img, negative_prompt=NEG, num_inference_steps=30, generator=gen()).images[0]
              for p in ["a photo of an astronaut", "a photo of a robot in a factory"]]
show([edge_img] + canny_outs, ["Canny edges", "prompt: astronaut", "prompt: robot in a factory"], size=3.2)
del canny_pipe, cn_canny; torch.cuda.empty_cache() if DEVICE == "cuda" else None

# %% [markdown]
# ## 4. IP-Adapter：参照画像の「意味・質感」で縛る
# テキストでは「この製品のこの質感」を正確に伝えられません。IP-Adapter は参照画像を CLIP 画像エンコーダで埋め込み、
# **テキストとは別のクロスアテンション** として U-Net に注入します（Decoupled Cross-Attention）。
# ここでは、ControlNet Tile（構造）に加えて、**高画質な参照画像**（同じ被写体の良品画像を想定）で質感を縛ります。

# %%
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
ref = load_sample("astronaut").crop((100, 0, 400, 300)).resize((512, 512))   # 参照画像（高画質な部分）
results = {}
for scale in (0.0, 0.5, 0.8):
    pipe.set_ip_adapter_scale(scale)
    results[scale] = pipe(PROMPT, image=lq, control_image=lq, ip_adapter_image=ref, strength=0.9,
                          negative_prompt=NEG, num_inference_steps=30, generator=gen()).images[0]
show([lq, ref] + list(results.values()), ["input", "IP-Adapter reference"] +
     [f"Tile + IP-Adapter scale={s}" for s in results], size=3.2)
for s, im in results.items():
    print(f"IP scale={s}: PSNR {psnr(im, hr):.2f} dB, SSIM {ssim(im, hr):.3f}")
gpu_mem()

# %% [markdown]
# ## 5.【実践例】スライド「ControlNet × img2img × IP-Adapter」の5段階を実際に実行する
# スライドの「① 前処理 → ② 構造で縛る → ③ 色を縛る → ④ 質感で縛る → ⑤ 後処理」を、1枚の画像で順に実行します。
#
# **処理のつながり（直列か？）**
# - ① → ⑤ は直列：① の出力（下地）が、拡散モデルの入力になり、拡散モデルの出力を ⑤ で後処理します
# - ②③④ は **1回の拡散生成の中で同時にかける3つの縛り** です（ControlNet＝構造、img2img の開始画像＝色、IP-Adapter＝質感）
# - 下の図では効果が分かるように、縛りを1つずつ **足していった** 結果を並べます（②③ → ②③＋④）
#
# 設定
# - 入力：高画質な人物写真（Kodak kodim04）を 1/4 に縮小し、ノイズと強い JPEG 圧縮を加えた「古いカメラ画像」相当
# - ① Real-ESRGAN で 4 倍拡大しつつノイズ・圧縮ノイズを除去（忠実な下地）
# - ②③ 下地を ControlNet Tile の **条件画像** にし、同じ下地から img2img（strength 0.6）で描き直す
# - ④ IP-Adapter（scale 0.5）に **同じ人物の高画質な写真** を参照として与える。
#   ここでは別の写真が無いので、元の写真を左右反転したもので代用しています（実務では、同じ対象を別に撮った高画質画像・良品画像を使う）
# - ⑤ 色・明るさ（低周波）を下地 ① に合わせ、細部（高周波）は生成結果を残す（Wavelet 型カラー補正）

# %%
import spandrel
import lpips
from io import BytesIO

hr_p = load_portrait()                                            # 正解（評価用。処理には使わない）
small = hr_p.resize((128, 128), Image.BICUBIC)
noisy = np.clip(to_np(small) + np.random.default_rng(0).normal(0, 0.03, (128, 128, 3)), 0, 1)
buf = BytesIO(); to_pil(noisy).save(buf, "JPEG", quality=30)
lq_in = Image.open(buf).convert("RGB")                             # 128x128 の劣化入力

# ① 前処理：Real-ESRGAN
esr = spandrel.ModelLoader().load_from_file(
    download("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")).to(DEVICE).eval()
with torch.no_grad():
    base = tensor_to_pil(esr(to_tensor(lq_in).to(DEVICE)))
del esr

# ④ で使う参照画像：同じ人物の高画質写真（ここでは元写真の左右反転で代用）
kod = Image.open(download("https://r0k.us/graphics/kodak/kodak/kodim04.png")).convert("RGB")
ref_p = kod.transpose(Image.FLIP_LEFT_RIGHT).crop((0, 100, 512, 612))

P_PROMPT = "a high quality portrait photo of a blonde woman wearing a red hat, detailed skin, sharp focus"
def diffuse(ip_scale):
    pipe.set_ip_adapter_scale(ip_scale)                           # IP-Adapter を読み込んだ後は scale=0 でも参照画像の指定が必要
    return pipe(P_PROMPT, image=base, control_image=base, ip_adapter_image=ref_p, strength=0.6,
                negative_prompt=NEG, num_inference_steps=30, controlnet_conditioning_scale=1.0,
                guidance_scale=6.0, generator=gen()).images[0]

step23 = diffuse(0.0)     # ②③ ControlNet Tile（構造）＋ img2img（下地から開始＝色）
step4 = diffuse(0.5)      # ②③＋④ IP-Adapter（参照の質感）

# ⑤ 後処理：Wavelet 型カラー補正（低周波＝色・明るさは下地、高周波＝細部は生成結果）
def color_fix(out, ref, sigma=3):
    o, r = to_np(out), to_np(ref)
    return to_pil(o - cv2.GaussianBlur(o, (0, 0), sigma) + cv2.GaussianBlur(r, (0, 0), sigma))

step5 = color_fix(step4, base)

lp_ = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
def lp_score(a):
    with torch.no_grad():
        return lp_(to_tensor(a).to(DEVICE) * 2 - 1, to_tensor(hr_p).to(DEVICE) * 2 - 1).item()

stages = {"input (128px, x4 bicubic)": lq_in.resize((512, 512), Image.BICUBIC), "1 Real-ESRGAN": base,
          "1 -> 2+3 Tile + img2img": step23, "1 -> 2+3+4 + IP-Adapter": step4, "-> 5 color fix": step5}
box_p = (110, 40, 430, 360)
show([im.crop(box_p) for im in stages.values()] + [hr_p.crop(box_p)],
     [f"{k}\nLPIPS {lp_score(v):.3f}" for k, v in stages.items()] + ["ground truth"], size=3.2, rows=2)
for k, v in stages.items():
    print(f"{k:28s} PSNR {psnr(v, hr_p):.2f} dB  LPIPS {lp_score(v):.3f}")
show([ref_p], ["IP-Adapter reference (substitute: mirrored photo)"], size=3)

# %% [markdown]
# **観察ポイント**（LPIPS：見た目の差。低いほど正解に近い）
# - ① ノイズ・ブロックは消えるが、肌や髪が「塗り絵」状になる（GAN 系の典型）
# - ②③ 拡散モデルが髪・まつ毛・肌の質感を描き足す。ControlNet Tile が構造を、img2img の開始画像が色を固定するので、顔の形・表情は保たれる
# - ④ 参照画像の質感・色味が加わり、髪の流れや目の色が正解に近づく
# - ⑤ 拡散モデルでずれた色・明るさを下地に戻す → 見た目も数値も最良
# - PSNR（画素の一致）は ②③ で一度下がる：描き足した細部は「もっともらしい」が画素単位では正解と一致しないため。⑤ で一部戻る
# - **必ず入力と並べて、変わってはいけない部分が変わっていないか確認する**（参照の質感は強く移るので、参照画像の選び方と scale が重要）

# %% [markdown]
# ## まとめ：何で縛るか
# | 縛りたいもの | 手段 | 備考 |
# |---|---|---|
# | 構図・形 | ControlNet（Tile / Canny / Depth） | Tile は復元・超解像向き |
# | 色・明るさ | img2img の strength を下げる / 色補正（Wavelet color fix） | |
# | 質感・意味・同一性 | IP-Adapter（参照画像）、テキストプロンプト、LoRA | 良品画像を参照に使える |
# | 観測との整合 | 残差射影（ノートブック08） | ハルシネーション抑制 |
#
# **実践アプローチ**：低品質な合成画像（CG・古いカメラ画像）は「ControlNet Tile で構造を固定 → img2img で高品質化 → IP-Adapter で良品の質感を参照」の併用が効果的です。
