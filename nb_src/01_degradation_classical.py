# %% [markdown]
# # 01. 画像劣化のメカニズムと古典的復元（Wiener Filter・補間拡大）
#
# **セミナー対応章**：第1章（1）〜（3）　｜　**推奨ランタイム**：CPUでも可（Colab無料枠でOK）　｜　**所要時間**：約2分
#
# このノートブックでは、
# 1. 劣化モデル $y = (x * k)\downarrow_s + n$（ボケ・縮小・ノイズ）で劣化画像を作る
# 2. 古典的な復元：補間拡大（Nearest / Bicubic / Lanczos）と Wiener Filter
# 3. 「失われた情報は戻らない」ことを周波数スペクトルで確認する
#
# を体験します。生成AIを使う前に「何が難しいのか」を理解するための出発点です。

# %% [setup]

# %% [markdown]
# ## 1. 劣化モデルで「劣化画像」を作る
# 実世界の劣化は、**ボケ（PSF：点拡がり関数 k との畳み込み）→ 縮小（↓s）→ ノイズ（n）→ JPEG圧縮** の組み合わせとして近似できます。

# %%
import cv2

def gaussian_psf(size=15, sigma=2.0):
    ax = np.arange(size) - size // 2
    g = np.exp(-ax**2 / (2 * sigma**2))
    k = np.outer(g, g)
    return k / k.sum()

def motion_psf(length=15, angle=30):
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    M = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    k = cv2.warpAffine(k, M, (length, length))
    return k / k.sum()

def blur(x, k):
    return cv2.filter2D(x, -1, k, borderType=cv2.BORDER_REFLECT)

def jpeg(x, q=30):
    ok, buf = cv2.imencode(".jpg", (np.clip(x, 0, 1) * 255).astype(np.uint8)[..., ::-1],
                           [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, 1)[..., ::-1].astype(np.float32) / 255

rng = np.random.default_rng(0)
x = to_np(load_sample("astronaut"))           # 正解画像（512x512）
k = gaussian_psf(15, 2.0)

y_blur = blur(x, k)
y_noise = np.clip(x + rng.normal(0, 0.05, x.shape), 0, 1)
y_down = cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)
y_jpeg = jpeg(x, 10)
y_all = jpeg(np.clip(cv2.resize(blur(x, k), (128, 128), interpolation=cv2.INTER_AREA)
                     + rng.normal(0, 0.03, (128, 128, 3)), 0, 1), 40)

show([x, y_blur, y_noise, y_down, y_jpeg, y_all],
     ["Ground truth", "Blur (Gaussian PSF)", "Noise (sigma=0.05)",
      "Downscale x1/4", "JPEG q=10", "Blur+Down+Noise+JPEG"], size=3.2, rows=2)

# %% [markdown]
# ## 2. 補間による拡大 vs 正解
# 補間（Nearest / Bilinear / Bicubic / Lanczos）は **周辺画素の重み付き平均** なので、新しい高周波（細かいテクスチャ）は作れません。

# %%
small = cv2.resize(x, (128, 128), interpolation=cv2.INTER_AREA)
ups = {
    "Nearest": cv2.resize(small, (512, 512), interpolation=cv2.INTER_NEAREST),
    "Bilinear": cv2.resize(small, (512, 512), interpolation=cv2.INTER_LINEAR),
    "Bicubic": cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC),
    "Lanczos": cv2.resize(small, (512, 512), interpolation=cv2.INTER_LANCZOS4),
}
crop = (slice(80, 240), slice(160, 320))   # 顔付近を拡大表示
show([x[crop]] + [np.clip(v[crop], 0, 1) for v in ups.values()],
     ["Ground truth"] + [f"{k}  PSNR={psnr(np.clip(v,0,1), x):.2f}dB" for k, v in ups.items()], size=3.2)

# %% [markdown]
# ## 3. Wiener Filter によるボケ除去
# 周波数領域で
# $$\hat X(u,v) = \frac{K^*(u,v)}{|K(u,v)|^2 + \lambda}\,Y(u,v)$$
# とする古典的な手法です。$\lambda$（ノイズ対信号比）が小さいとノイズが爆発し、大きいとボケが残ります。

# %%
def psf2otf(k, shape):
    pad = np.zeros(shape, np.float32)
    kh, kw = k.shape
    pad[:kh, :kw] = k
    pad = np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.fft.fft2(pad)

def wiener(y, k, lam):
    K = psf2otf(k, y.shape[:2])
    H = np.conj(K) / (np.abs(K) ** 2 + lam)
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(y[..., c]) * H)) for c in range(3)], -1)

y = np.clip(blur(x, k) + rng.normal(0, 0.01, x.shape), 0, 1)   # ボケ + 弱いノイズ
results = [wiener(y, k, lam) for lam in (1e-4, 1e-2, 1e-1)]
show([x, y] + results,
     ["Ground truth", f"Blurred+noise {psnr(y,x):.2f}dB"]
     + [f"Wiener lam={l}  {psnr(np.clip(r,0,1),x):.2f}dB" for l, r in zip((1e-4, 1e-2, 1e-1), results)],
     size=3.2)

# %% [markdown]
# ## 4. 「失われた情報」を周波数で見る
# PSF の周波数応答 |K| がほぼ 0 になる帯域では、どんなフィルタでも情報を取り戻せません。
# 縮小（ダウンサンプリング）でも、ナイキスト周波数より上の成分は消えてしまいます。
# → ここを **学習データから「予測」して埋める** のが生成AIのアプローチです。

# %%
def log_spectrum(img):
    g = img.mean(-1)
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(g))))

K = np.abs(np.fft.fftshift(psf2otf(k, x.shape[:2])))
show([log_spectrum(x), log_spectrum(y), K, log_spectrum(np.clip(ups["Bicubic"], 0, 1))],
     ["Spectrum: GT", "Spectrum: blurred", "|K| of PSF (0=lost)", "Spectrum: bicubic x4"], size=3.5)

# %% [markdown]
# ## 5. Richardson–Lucy 法（反復型デコンボリューション）
# 天文学や顕微鏡で使われてきた反復法。反復回数を増やすとシャープになりますが、ノイズやリンギングも増えます。

# %%
from skimage.restoration import richardson_lucy

rl = {n: np.stack([richardson_lucy(y[..., c], k, num_iter=n) for c in range(3)], -1) for n in (5, 30, 100)}
show([y] + list(rl.values()), ["Input"] + [f"RL iter={n}  {psnr(np.clip(v,0,1),x):.2f}dB" for n, v in rl.items()], size=3.5)

# %% [markdown]
# ## まとめ
# - 劣化は「ボケ・縮小・ノイズ・圧縮」の合成としてモデル化できる
# - 補間や Wiener Filter は **観測された情報の範囲内** でしか復元できない
# - 消えた高周波を取り戻すには「自然画像らしさ」の知識（事前分布）が必要 → 次のノートブック以降で生成AIを使います
#
# **自分の画像で試す**：`x = to_np(Image.open("my_image.jpg").convert("RGB"))` に置き換えて再実行してください（Colabでは左のフォルダアイコンからアップロード）。
