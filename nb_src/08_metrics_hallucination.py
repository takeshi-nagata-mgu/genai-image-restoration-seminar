# %% [markdown]
# # 08. 評価指標（PSNR/SSIM → LPIPS/CLIP-IQA）とハルシネーション対策（残差射影・不確実性マップ）
#
# **セミナー対応章**：第4章（1）（3）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約5分
#
# 1. 同じ劣化画像を「補間 / GAN / 拡散モデル」で超解像し、**PSNR・SSIM・LPIPS・CLIP-IQA** で評価 → 指標によって勝者が変わる（知覚–歪みトレードオフ）
# 2. **残差射影（Residual Projection / Back-projection）**：出力を縮小したら入力と一致する、という制約で生成結果を補正
# 3. **不確実性マップ**：seed を変えて複数回生成し、結果がばらつく場所＝「AIが想像で描いた場所」を可視化

# %% [setup]

# %%
if IN_COLAB:
    pip_install("torchmetrics", "piq")
import cv2
import lpips
from PIL import ImageFilter
import spandrel
from torchmetrics.multimodal import CLIPImageQualityAssessment
from diffusers import StableDiffusionUpscalePipeline

# %% [markdown]
# ## 1. 3種類の超解像結果を作る（128 → 512）

# %%
hr = load_portrait()                          # 顔が大きく写った高画質画像（512x512）
lr = hr.resize((128, 128), Image.BICUBIC)     # 1/4 に縮小したものを入力にする

esrgan = spandrel.ModelLoader().load_from_file(
    download("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")).to(DEVICE).eval()
sd_up = StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=DTYPE).to(DEVICE)
sd_up.set_progress_bar_config(disable=True)

def sd_sr(img, seed=0):
    return sd_up(prompt="a photo of a woman wearing a red hat, high quality, detailed", image=img, num_inference_steps=30,
                 noise_level=20, generator=torch.Generator(DEVICE).manual_seed(seed)).images[0]

with torch.no_grad():
    outs = {
        "Bicubic": lr.resize(hr.size, Image.BICUBIC),
        "Blurred GT": hr.filter(ImageFilter.GaussianBlur(2)),
        "Real-ESRGAN": tensor_to_pil(esrgan(to_tensor(lr).to(DEVICE))),
        "SD x4 Upscaler": sd_sr(lr),
    }

# %% [markdown]
# ## 2. 4つの指標で評価
# - **PSNR / SSIM**：正解との画素・構造の一致度（大きいほど良い）
# - **LPIPS**：深層特徴空間での距離。人間の知覚的な差に近い（**小さいほど良い**）
# - **CLIP-IQA**：正解画像なしで「Good photo / Bad photo」とのCLIP類似度から画質を推定（大きいほど良い）

# %%
lp = lpips.LPIPS(net="alex", verbose=False).to(DEVICE)
clipiqa = CLIPImageQualityAssessment(prompts=("quality",)).to(DEVICE)

def lpips_score(a, b):
    with torch.no_grad():
        return lp(to_tensor(a).to(DEVICE) * 2 - 1, to_tensor(b).to(DEVICE) * 2 - 1).item()

def clipiqa_score(a):
    with torch.no_grad():
        return clipiqa(to_tensor(a).to(DEVICE)).item()

scores = {n: (psnr(im, hr), ssim(im, hr), lpips_score(im, hr), clipiqa_score(im)) for n, im in outs.items()}
print(f"{'method':16s} {'PSNR↑':>7s} {'SSIM↑':>7s} {'LPIPS↓':>7s} {'CLIP-IQA↑':>9s}")
for n, (p_, s_, l_, c_) in scores.items():
    print(f"{n:16s} {p_:7.2f} {s_:7.3f} {l_:7.3f} {c_:9.3f}")
gt_iqa = clipiqa_score(hr)
print(f"{'Ground truth':16s} {'-':>7s} {'-':>7s} {'-':>7s} {gt_iqa:9.3f}")

# 画像の下に各指標を表示（各指標で最も良い手法に (best)）
box = (110, 40, 430, 360)                     # 顔の部分を拡大して表示
best = {0: max, 1: max, 2: min, 3: max}
best_val = {k: f(v[k] for v in scores.values()) for k, f in best.items()}
fig, axes = plt.subplots(1, len(outs) + 1, figsize=(3.2 * (len(outs) + 1), 4.6))
for ax, (n, im) in zip(axes, [("Ground truth", hr)] + list(outs.items())):
    ax.imshow(im.crop(box)); ax.set_title(n, fontsize=13); ax.set_xticks([]); ax.set_yticks([])
    if n == "Ground truth":
        lines = [("PSNR -", False), ("SSIM -", False), ("LPIPS -", False), (f"CLIP-IQA {gt_iqa:.3f}", False)]
    else:
        v = scores[n]
        lines = [(f"{a} {b}" + ("  (best)" if np.isclose(v[k], best_val[k]) else ""), np.isclose(v[k], best_val[k]))
                 for a, b, k in [("PSNR", f"{v[0]:.2f} dB", 0), ("SSIM", f"{v[1]:.3f}", 1), ("LPIPS", f"{v[2]:.3f}", 2), ("CLIP-IQA", f"{v[3]:.3f}", 3)]]
    for i, (t, is_best) in enumerate(lines):    # best は太字・赤
        ax.text(0.5, -0.07 - 0.1 * i, t, transform=ax.transAxes, ha="center", va="top", fontsize=12,
                color="#C00000" if is_best else "black", fontweight="bold" if is_best else "normal")
plt.tight_layout(); plt.subplots_adjust(bottom=0.3); plt.show()

# %% [markdown]
# **観察ポイント（知覚–歪みトレードオフ, Blau & Michaeli 2018）**
# - PSNR/SSIM は「ぼかした画像」を高く評価しがち（画素の平均二乗誤差は“平均的な答え”で最小になる）
# - LPIPS・CLIP-IQA は生成系（GAN・拡散）を高く評価するが、**それが正しい細部かどうかは保証しない**
# - FID（多数の画像の分布の近さ）は数百〜数万枚単位で使う指標で、1枚の画像の正しさは測れない
# → **用途に合わせて複数の指標＋下流タスク（OCR認識率・検査精度）＋専門家評価** を組み合わせる

# %% [markdown]
# ## 3. 残差射影（Residual Projection）でハルシネーションを抑える
# 「超解像結果を縮小すると、観測された低解像度画像に一致するはず」という制約集合へ出力を射影します：
# $$x \leftarrow x - \mathrm{Up}\big(\mathrm{Down}(x) - y_{LR}\big)$$
# 生成AIが描いた **観測と矛盾する成分だけ** を取り除き、観測と整合する細部は残します（ResShift・Resfusion なども同じ考え方を拡散過程に組み込んでいます）。

# %%
def residual_projection(sr, lr, iters=20):
    x = to_np(sr).astype(np.float32)
    y = to_np(lr).astype(np.float32)
    errs = []
    for _ in range(iters):
        d = cv2.resize(x, y.shape[1::-1], interpolation=cv2.INTER_AREA) - y        # 縮小して観測との差
        errs.append(np.abs(d).max())
        x = np.clip(x - cv2.resize(d, x.shape[1::-1], interpolation=cv2.INTER_CUBIC), 0, 1)
    return to_pil(x), errs

# 縮小方法を揃えるため、観測 LR を INTER_AREA で作り直して実験
lr_area = to_pil(cv2.resize(to_np(hr), (128, 128), interpolation=cv2.INTER_AREA))
sr0 = sd_sr(lr_area, seed=0)
sr_proj, errs = residual_projection(sr0, lr_area)
plt.figure(figsize=(5, 3)); plt.plot(errs, "o-"); plt.xlabel("iteration"); plt.ylabel("max |Down(x) - y|"); plt.title("LR consistency error"); plt.show()
for n, im in [("SD x4", sr0), ("SD x4 + projection", sr_proj)]:
    print(f"{n:20s} PSNR {psnr(im, hr):.2f}  SSIM {ssim(im, hr):.3f}  LPIPS {lpips_score(im, hr):.3f}")
show([hr.crop(box), sr0.crop(box), sr_proj.crop(box)], ["GT", "SD x4 (raw)", "SD x4 + residual projection"], size=3.5)

# %% [markdown]
# > **ノイズを含む入力の場合**：入力そのものに射影するとノイズまで戻ってしまいます。
# > ハルシネーションの少ないノイズ除去（例：SwinIR・SCUNet）の結果を参照画像にして射影し、拡散モデルでの修復と交互に繰り返す方法が有効です（講師らの DIA2026 発表）。

# %% [markdown]
# ## 4. 不確実性マップ：AIが「想像で描いた」場所を可視化する
# 生成AIの出力は確率的です。seed を変えて複数回生成し、画素ごとの **ばらつき（標準偏差）** を見ると、
# 観測から決まらない＝ハルシネーションが起きやすい場所が分かります。結果を提示するときは、このような **信頼度マップ** を添えるのが望ましいです。

# %%
samples = [to_np(sd_sr(lr, seed=s)) for s in range(4)]
std_map = np.stack(samples).std(0).mean(-1)
show([to_pil(samples[0]), to_pil(samples[1]), std_map / std_map.max()],
     ["seed 0", "seed 1", "uncertainty (std over 4 seeds)"], size=4)

# %% [markdown]
# ## まとめ
# - 指標は1つで判断しない：**PSNR/SSIM（忠実性）× LPIPS/CLIP-IQA/FID（自然さ）× 下流タスク × 専門家評価**
# - ハルシネーション対策：**残差射影で観測と整合させる**／**条件付けで縛る**（ノートブック06）／**不確実性を可視化して提示**
# - 医療・検査・証拠用途では「原画像の保存・処理内容の記録・生成部分の明示」をセットで運用する
