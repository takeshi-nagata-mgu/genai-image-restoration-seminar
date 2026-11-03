# %% [markdown]
# # 03. 超解像（Super Resolution）：補間 vs Real-ESRGAN vs 拡散モデル
#
# **セミナー対応章**：第3章（1）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約5分（初回はモデルDLで+数分）
#
# 1. Real-ESRGAN が学習に使う **高次劣化モデル**（劣化を2回繰り返す）で、リアルな低画質画像を作る
# 2. **Bicubic補間 / Real-ESRGAN（GAN）/ Stable Diffusion x4 Upscaler（拡散モデル）** を比較
# 3. **処理時間・GPUメモリ** を測り、倍率と計算コストのトレードオフを確認
# 4. 大きな画像を扱うための **タイル処理**
#
# > StableSR・DiffBIR・SUPIR などの大型モデルは、配布の ComfyUI ワークフロー、またはノートブック 15（Colab有料枠向け）で試せます。

# %% [setup]

# %% [markdown]
# ## 1. 高次劣化モデル（Real-ESRGAN の学習データの作り方）
# 現実の画像は「撮影時のボケ → 縮小 → センサノイズ → JPEG → SNSで再圧縮 …」と **劣化が何重にも重なっています**。
# Real-ESRGAN はこれを「劣化処理を2回繰り返す」ことで模擬し、きれいな画像から大量の学習ペアを合成しました。

# %%
import cv2

def degrade_once(x, rng, scale):
    s = rng.uniform(0.3, 1.0)                                     # ボケ
    x = cv2.GaussianBlur(x, (0, 0), s)
    h, w = x.shape[:2]                                            # 縮小（補間方法もランダム）
    interp = rng.choice([cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
    x = cv2.resize(x, (int(w / scale), int(h / scale)), interpolation=interp)
    x = np.clip(x + rng.normal(0, rng.uniform(0.003, 0.012), x.shape), 0, 1)   # ノイズ
    q = int(rng.uniform(60, 95))                                  # JPEG圧縮
    _, buf = cv2.imencode(".jpg", (x * 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, 1)[..., ::-1].astype(np.float32) / 255

def high_order_degradation(img, seed=0):
    rng = np.random.default_rng(seed)
    x = degrade_once(to_np(img), rng, 2)      # 1回目の劣化（1/2）
    x = degrade_once(x, rng, 2)               # 2回目の劣化（さらに1/2 → 合計1/4）
    return to_pil(x)

hr = load_sample("astronaut")                 # 512x512 の正解画像
lr = high_order_degradation(hr)               # 128x128 の劣化画像
print("HR:", hr.size, " LR:", lr.size)
show([hr, lr.resize(hr.size, Image.NEAREST)], ["Ground truth (512)", "Degraded LR (128, nearest x4)"])

# %% [markdown]
# ### 一次劣化と二次劣化を見比べる
# 従来の超解像（SRCNN・ESRGAN など）は「bicubic で縮小しただけ」の **理想的な劣化** で学習していました。
# しかし実際の画像は、ボケ・ノイズ・圧縮が何重にも重なっているため、理想的な劣化で学習したモデルは実写でうまく働きません。
# そこで Real-ESRGAN は、劣化を1回（一次劣化）だけでなく **2回重ねる（二次＝高次劣化）** ことで、現実の複雑な劣化を学習データに含めました。
# 分かりやすいように、ここでは劣化の強さを固定して比べます（どれも 512 → 128 px、表示は最近傍で拡大）。

# %%
def degrade_fixed(x, blur, scale, noise, q):
    x = cv2.GaussianBlur(x, (0, 0), blur)
    h, w = x.shape[:2]
    x = cv2.resize(x, (int(w / scale), int(h / scale)), interpolation=cv2.INTER_AREA)
    x = np.clip(x + np.random.default_rng(0).normal(0, noise, x.shape), 0, 1)
    _, buf = cv2.imencode(".jpg", (x * 255).astype(np.uint8)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, 1)[..., ::-1].astype(np.float32) / 255

x0 = to_np(hr)
ideal = cv2.resize(x0, (128, 128), interpolation=cv2.INTER_CUBIC)                      # 理想的な劣化：縮小だけ
first = degrade_fixed(x0, 1.2, 4, 0.02, 60)                                           # 一次劣化：ボケ→縮小→ノイズ→JPEG を1回
second = degrade_fixed(degrade_fixed(x0, 1.2, 2, 0.02, 60), 0.8, 2, 0.015, 40)        # 二次劣化：1/2 ずつ 2 回重ねる
box = (110, 20, 366, 276)                                                             # 顔のまわりを拡大
crops = [hr.crop(box)] + [to_pil(v).resize((512, 512), Image.NEAREST).crop(box) for v in (ideal, first, second)]
show(crops, ["Ground truth", "Ideal (bicubic down only)", "1st-order degradation", "2nd-order (high-order)"], size=3.6)

# %% [markdown]
# ## 2. 手法1：Bicubic 補間

# %%
results, times = {}, {}
t = time.time()
results["Bicubic"] = lr.resize(hr.size, Image.BICUBIC)
times["Bicubic"] = time.time() - t

# %% [markdown]
# ## 3. 手法2：Real-ESRGAN（GAN 系）
# `spandrel` ライブラリを使うと、Real-ESRGAN・SwinIR・HAT・SCUNet など多数の復元モデルを **同じ書き方** で読み込めます（ComfyUI 内部でも使われています）。

# %%
import spandrel

def load_spandrel(url):
    model = spandrel.ModelLoader().load_from_file(download(url))
    model = model.to(DEVICE).eval()
    print(f"{model.architecture.name}: scale x{model.scale}")
    return model

@torch.no_grad()
def run_sr(model, img, half=True):
    x = to_tensor(img).to(DEVICE)
    if half and DEVICE == "cuda" and model.supports_half:
        model.model.half(); x = x.half()
    y = model(x)
    model.model.float()
    return tensor_to_pil(y)

esrgan = load_spandrel("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")
compact = load_spandrel("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth")

for name, m in [("Real-ESRGAN x4plus", esrgan), ("Real-ESRGAN compact", compact)]:
    run_sr(m, lr)                                   # ウォームアップ（初回はCUDA初期化で遅い）
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.time()
    results[name] = run_sr(m, lr)
    if DEVICE == "cuda": torch.cuda.synchronize()
    times[name] = time.time() - t

# %% [markdown]
# ## 4. 手法3：拡散モデルによる超解像（Stable Diffusion x4 Upscaler）
# 低解像度画像を条件として、拡散モデルが細部を「生成」します。プロンプトで内容のヒントも与えられます。
# 初回は約3GBのダウンロードがあります。T4 では 128→512 で数十秒程度です。

# %%
from diffusers import StableDiffusionUpscalePipeline

sd_up = StableDiffusionUpscalePipeline.from_pretrained(
    "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=DTYPE).to(DEVICE)
sd_up.set_progress_bar_config(disable=True)

if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats()
t = time.time()
results["SD x4 Upscaler"] = sd_up(prompt="a photo of an astronaut, high quality, detailed",
                                  image=lr, num_inference_steps=30, noise_level=20,
                                  generator=torch.Generator(DEVICE).manual_seed(0)).images[0]
times["SD x4 Upscaler"] = time.time() - t
gpu_mem()

# %% [markdown]
# ## 5. 比較（全体・拡大）

# %%
names = list(results)
show([hr] + [results[n] for n in names],
     ["Ground truth"] + [f"{n}\nPSNR {psnr(results[n], hr):.2f} / SSIM {ssim(results[n], hr):.3f}" for n in names], size=3.4)
box = (180, 80, 340, 240)   # 顔まわりを拡大
show([hr.crop(box)] + [results[n].crop(box) for n in names], ["GT"] + names, size=3.4)

print("\n処理時間（128→512）")
for n in names:
    print(f"  {n:22s}: {times[n]*1000:8.1f} ms")

# %% [markdown]
# **観察ポイント**
# - Bicubic は PSNR は悪くないが、ぼやけている（新しいテクスチャは作れない）
# - Real-ESRGAN は輪郭がくっきりし、ノイズも消えるが、細かいテクスチャが「塗り絵」状になることがある
# - 拡散モデルは質感をよく生成するが、**正解と異なる細部（ハルシネーション）** を作りうるので PSNR は必ずしも高くない
# - 処理時間は「補間 ≪ GAN ≪ 拡散モデル」。用途（リアルタイム処理 / オフラインの高品質化）で選ぶ

# %% [markdown]
# ## 6. 実務Tips：大きな画像はタイルに分けて処理する
# 4K 画像をそのまま入力すると GPU メモリが不足します。重なり（overlap）付きのタイルに分割して処理し、重なり部分を平均して継ぎ目を消します。

# %%
@torch.no_grad()
def tiled_sr(model, img, tile=128, overlap=16):
    x = to_tensor(img).to(DEVICE)
    _, _, H, W = x.shape
    s = model.scale
    out = torch.zeros(1, 3, H * s, W * s, device=DEVICE)
    weight = torch.zeros_like(out)
    step = tile - overlap
    for y0 in range(0, max(H - overlap, 1), step):
        for x0 in range(0, max(W - overlap, 1), step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0_, x0_ = max(y1 - tile, 0), max(x1 - tile, 0)
            patch = model(x[:, :, y0_:y1, x0_:x1])
            out[:, :, y0_ * s:y1 * s, x0_ * s:x1 * s] += patch
            weight[:, :, y0_ * s:y1 * s, x0_ * s:x1 * s] += 1
    return tensor_to_pil(out / weight)

big_lr = hr.resize((256, 256), Image.BICUBIC)        # 256 → 1024
t = time.time()
big_sr = tiled_sr(compact, big_lr, tile=96, overlap=16)
print(f"tiled SR {big_lr.size} -> {big_sr.size}: {time.time()-t:.2f}s")
show([big_lr.resize(big_sr.size, Image.BICUBIC).crop((300, 150, 700, 550)), big_sr.crop((300, 150, 700, 550))],
     ["Bicubic x4 (crop)", "Tiled Real-ESRGAN compact x4 (crop)"])

# %% [markdown]
# ## まとめ
# | 手法 | 仕組み | 長所 | 短所 |
# |---|---|---|---|
# | 補間（Bicubic等） | 周辺画素の重み付き平均 | 超高速・ハルシネーションなし | ぼやける |
# | GAN系（Real-ESRGAN） | 劣化→高画質の写像を敵対的学習 | 高速・実用的・エッジ実装可 | テクスチャが不自然になることがある |
# | 拡散系（SD Upscaler, StableSR, DiffBIR, SUPIR） | 劣化画像で条件付けた生成 | 質感が最も自然 | 遅い・大メモリ・ハルシネーション |
#
# **自分の画像で試す**：`lr = Image.open("my_image.jpg").convert("RGB")` に置き換えて 2. 以降を再実行してください（正解画像がない場合 PSNR は計算できないので、`show` だけ使ってください）。
