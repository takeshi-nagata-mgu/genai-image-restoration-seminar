# %% [markdown]
# # 03. 超解像（Super Resolution）：補間 vs Real-ESRGAN vs 拡散モデル
#
# **セミナー対応章**：第3章（1）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：1.〜6. 約5分（初回はモデルDLで+数分）、7. の CNN の学習（2 つ）は T4 で約 3〜4 時間（`L12_ITERS = 10000` なら約 40 分、`RUN_L1L2 = False` で飛ばせる）
#
# 1. Real-ESRGAN が学習に使う **高次劣化モデル**（劣化を2回繰り返す）で、リアルな低画質画像を作る
# 2. **Bicubic補間 / Real-ESRGAN（GAN）/ Stable Diffusion x4 Upscaler（拡散モデル）** を比較
# 3. **処理時間・GPUメモリ** を測り、倍率と計算コストのトレードオフを確認
# 4. 大きな画像を扱うための **タイル処理**
# 5. **同じ CNN を L2 損失と L1 損失で学習** して、CNN の復元がボケる理由（候補の平均）を確かめる
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
# ## 7. CNN（L2 損失）の復元はなぜボケるのか：同じ CNN を損失だけ変えて学習する
# 同じネットワーク（SRResNet 型、16 ブロック、約 150 万係数）を、同じデータ（DIV2K の検証用 100 枚）・同じ乱数で、**損失だけ L2（二乗誤差）と L1（絶対誤差）に変えて** 学習します。
# 劣化は「縦横 1/4 に縮小 ＋ ノイズ σ = 5」。比較として、公開モデルの Real-ESRNet（L1 損失）と Real-ESRGAN（L1 ＋ 知覚 ＋ GAN 損失。同じ RRDB・同じデータで損失だけが違う）も並べます。
# - L2 を最小にする出力は、正解の候補の「平均」（L1 なら「中央値」）。石の目地の位置が少しずつ違う候補を平均すると、模様はぼやけて消える
# - 評価は Kodak 24 枚の平均（PSNR は高いほど、LPIPS は低いほど良い）と、kodim01 の石の壁の拡大
#
# 使うモデル・データ：SRResNet 型の CNN（C. Ledig+, CVPR 2017 の生成器と同じ構成。このノートブックで学習）、
# Real-ESRNet x4plus / Real-ESRGAN x4plus（X. Wang+, ICCVW 2021。RRDB ネットワーク、約 1,670 万パラメータ、github.com/xinntao/Real-ESRGAN、BSD-3）、
# LPIPS（R. Zhang+, CVPR 2018。AlexNet の特徴で見た目の差を測る、BSD-2）、DIV2K（E. Agustsson & R. Timofte, CVPRW 2017。学習用の高解像度画像）
#
# 学習は 1 つ 50,000 回で、RTX 5090 で約 15 分、**T4 では 1 つ 約 1.5〜2 時間** かかります。`L12_ITERS` を 10,000 にすると約 5 分の 1 の時間になります（値は少し下がる）。
# 学習した重みは weights/ に保存し、次からは読み込むだけです（`RUN_L1L2 = False` で飛ばせます）。

# %%
RUN_L1L2 = True
L12_ITERS = 50000
if RUN_L1L2:
    import glob, zipfile, cv2, lpips
    import torch.nn as nn, torch.nn.functional as nnF
    class RB(nn.Module):
        def __init__(s, c): super().__init__(); s.b = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(True), nn.Conv2d(c, c, 3, 1, 1))
        def forward(s, x): return x + s.b(x)
    class SRResNet(nn.Module):
        def __init__(s, c=64, n=16):
            super().__init__()
            s.head = nn.Conv2d(3, c, 3, 1, 1); s.body = nn.Sequential(*[RB(c) for _ in range(n)], nn.Conv2d(c, c, 3, 1, 1))
            s.up = nn.Sequential(nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True),
                                 nn.Conv2d(c, 4 * c, 3, 1, 1), nn.PixelShuffle(2), nn.ReLU(True), nn.Conv2d(c, 3, 3, 1, 1))
        def forward(s, x):
            h = s.head(x); return s.up(h + s.body(h)) + nnF.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)

    def degrade_t(hr, g):                                      # 学習用の劣化：縦横 1/4（bicubic）＋ノイズ σ=5 → 8 bit
        lr = nnF.interpolate(hr, scale_factor=0.25, mode="bicubic", antialias=True, align_corners=False)
        lr = lr + torch.randn(lr.shape, generator=g, device=lr.device) * 5 / 255
        return (lr.clamp(0, 1) * 255).round() / 255

    def train_srresnet(loss_name, iters):
        path = os.path.join(WEIGHTS_DIR, f"srresnet_{loss_name}.pth")
        net = SRResNet().to(DEVICE).to(memory_format=torch.channels_last)
        if os.path.exists(path):
            net.load_state_dict(torch.load(path, map_location=DEVICE)); print(f"{loss_name}: 学習済みの重みを読み込みました（{path}）")
            return net.eval()
        ddir = os.path.join("data", "DIV2K_valid_HR")
        if not os.path.exists(ddir):
            os.makedirs("data", exist_ok=True)
            zipfile.ZipFile(download("https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip")).extractall("data")   # 約 450 MB
        imgs = [torch.from_numpy(np.asarray(Image.open(f).convert("RGB"))).permute(2, 0, 1).to(DEVICE) for f in sorted(glob.glob(os.path.join(ddir, "*.png")))]
        torch.manual_seed(0); g = torch.Generator(device=DEVICE).manual_seed(0); rs = np.random.default_rng(0)
        opt = torch.optim.Adam(net.parameters(), 2e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, 1e-6)
        crit = nn.MSELoss() if loss_name == "l2" else nn.L1Loss()
        bf16 = torch.cuda.is_bf16_supported()                  # T4 は bfloat16 が使えないので float16 ＋ 損失のスケーリング
        scaler = torch.amp.GradScaler(enabled=not bf16)
        B, Pz = 16, 192; t0 = time.time()
        for it in range(1, iters + 1):
            batch = []
            for _ in range(B):
                im = imgs[rs.integers(len(imgs))]; y0 = rs.integers(im.shape[1] - Pz); x0 = rs.integers(im.shape[2] - Pz)
                p = im[:, y0:y0 + Pz, x0:x0 + Pz]
                if rs.random() < 0.5: p = p.flip(2)
                if rs.random() < 0.5: p = p.transpose(1, 2)
                batch.append(p)
            hr_b = torch.stack(batch).float() / 255
            with torch.autocast("cuda", torch.bfloat16 if bf16 else torch.float16):
                sr_b = net(degrade_t(hr_b, g).contiguous(memory_format=torch.channels_last))
            loss = crit(sr_b.float(), hr_b)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if it % 5000 == 0:
                print(f"{loss_name} it {it} loss {loss.item():.5f} ({time.time()-t0:.0f}s)")
        torch.save(net.state_dict(), path)
        del imgs; return net.eval()

    cnn = {k: train_srresnet(k, L12_ITERS) for k in ("l2", "l1")}
    esrnet = load_spandrel("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth")

    def degrade_pil(hr_, seed=0):                              # 評価用の劣化（学習と同じ：1/4 ＋ ノイズ σ=5）
        lr_ = hr_.resize((hr_.width // 4, hr_.height // 4), Image.BICUBIC)
        a = np.clip(np.asarray(lr_, np.float32) / 255 + np.random.default_rng(seed).normal(0, 5 / 255, (lr_.height, lr_.width, 3)), 0, 1)
        return Image.fromarray((a * 255).round().astype(np.uint8))
    @torch.no_grad()
    def run_net(net, lr_):
        y = net(to_tensor(lr_).to(DEVICE))
        return Image.fromarray((y[0].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8))
    lp = lpips.LPIPS(net="alex", verbose=False)
    lpd = lambda a, b: lp(to_tensor(a) * 2 - 1, to_tensor(b) * 2 - 1).item()
    ps = lambda a, b: cv2.PSNR(np.asarray(a), np.asarray(b))
    nets = {"CNN (L2 loss)": cnn["l2"], "CNN (L1 loss)": cnn["l1"], "Real-ESRNet": esrnet, "Real-ESRGAN": esrgan}
    acc = {k: [] for k in ["Bicubic"] + list(nets)}
    for i in range(1, 25):                                     # Kodak 24 枚の平均
        hr_ = Image.open(download(f"https://r0k.us/graphics/kodak/kodak/kodim{i:02d}.png")).convert("RGB"); lr_ = degrade_pil(hr_)
        bic = lr_.resize(hr_.size, Image.BICUBIC); acc["Bicubic"].append((ps(bic, hr_), lpd(bic, hr_)))
        for k, n in nets.items():
            o_ = run_net(n, lr_); acc[k].append((ps(o_, hr_), lpd(o_, hr_)))
    print("Kodak 24 枚の平均")
    for k, v in acc.items():
        v = np.array(v); print(f"  {k:16s} PSNR {v[:, 0].mean():.2f} dB  LPIPS {v[:, 1].mean():.3f}")

    hr_ = Image.open(download("https://r0k.us/graphics/kodak/kodak/kodim01.png")).convert("RGB"); lr_ = degrade_pil(hr_)
    box_w = (250, 150, 506, 406)                               # 石の壁
    ims = [("Ground truth", hr_), ("Bicubic", lr_.resize(hr_.size, Image.BICUBIC))] + [(k, run_net(n, lr_)) for k, n in nets.items()]
    show([im.crop(box_w) for _, im in ims],
         [k if k == "Ground truth" else f"{k}\nPSNR {ps(im, hr_):.2f} / LPIPS {lpd(im, hr_):.3f}" for k, im in ims], size=3.3)

# %% [markdown]
# **結果の読み方**：L2 でも L1 でも、CNN の出力は石の目地や表面のざらつきがぼやける（候補の平均・中央値になる）。PSNR は高いが見た目はボケ（知覚–歪みトレードオフ）。
# Kodak 24 枚の平均では、PSNR（＝L2 の尺度）でも L1 の方がわずかに高い（L1 は誤差の小さい細部でも同じ強さで直すので、細部まで学習が進む）。
# 右端の Real-ESRGAN は質感がくっきりするが、模様は正解と一致しない（「もっともらしい 1 サンプル」を出す GAN の性質）。

# %% [markdown]
# ## まとめ
# | 手法 | 仕組み | 長所 | 短所 |
# |---|---|---|---|
# | 補間（Bicubic等） | 周辺画素の重み付き平均 | 超高速・ハルシネーションなし | ぼやける |
# | GAN系（Real-ESRGAN） | 劣化→高画質の写像を敵対的学習 | 高速・実用的・エッジ実装可 | テクスチャが不自然になることがある |
# | 拡散系（SD Upscaler, StableSR, DiffBIR, SUPIR） | 劣化画像で条件付けた生成 | 質感が最も自然 | 遅い・大メモリ・ハルシネーション |
#
# **自分の画像で試す**：`lr = Image.open("my_image.jpg").convert("RGB")` に置き換えて 2. 以降を再実行してください（正解画像がない場合 PSNR は計算できないので、`show` だけ使ってください）。
