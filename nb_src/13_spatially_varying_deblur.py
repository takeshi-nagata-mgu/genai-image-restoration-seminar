# %% [markdown]
# # 13. 最小二乗法による「逆カーネル」と、奥行きで変わるピンボケの復元
#
# **セミナー対応章**：第5章（3）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU（9. の拡散モデルは GPU メモリ 約 8 GB）
#
# **所要時間（T4 の目安）**：1.〜5. 約5分　｜　6. CNN 約10〜15分　｜　7. 先行研究 4 手法 約60〜90分　｜　8. DA-LSIK-PB/PH 約30〜45分　｜　9. DA-LSIK-Diff 約20〜30分（重み 6.7 GB のダウンロードを含む）
# 　→ 全部で 約2〜3時間。6.〜9. はそれぞれ `RUN_CNN`・`RUN_PRIOR`・`RUN_UNFOLD`・`RUN_DIFF` を False にすると飛ばせます（Colab 無料枠は長時間の連続使用で切断されることがあるので、必要な所だけ実行するのがおすすめ）
#
# 1. **講師の手法 LSIK の再現**（PSF が一定の場合）：ボケ画像の 15×15 画素から、鮮明画像の中心画素を予測する **線形フィルタ（＝PSF の逆カーネル）を最小二乗法で学習** する
#    （LSIK：Least-Squares Inverse Kernel）。論文と同じ条件（ガウスぼけ σ=1.5、学習画像 3 枚）で、Wiener フィルタと比べる
#    （T. Nagata, "Fast Blind Deconvolution with Simple Machine Learning", Proc. 7th Asia Int. Symp. on Mechatronics, Springer, 2020）
# 2. **奥行きで PSF が変わる場合**：深層学習で奥行きを推定 → 場所ごとのボケの大きさ σ を求め、**σ 別に学習した逆カーネルを場所ごとに切り替えて** 復元する
# 3. **奥行き推定の誤差まで含めて学習**（提案手法 **DA-LSIK**）：推定 σ ごとに「どの逆カーネルをどれだけ混ぜるか」も最小二乗法で学習する
# 4. 比較：Wiener フィルタ（固定／場所ごと）、画素ごとに直接予測する CNN、**先行研究 IFAN・KPAC・INIKNet・MultiWienerNet**（同じ学習画像 3 枚で学習し直す）
# 5. **DA-LSIK の出力からさらに追い込む**：DRUNet と交互にくり返す **DA-LSIK-PB / PH**（Deep Unfolding）、拡散モデルで描き直す **DA-LSIK-Diff**
#
# **関連する先行研究**（「場所ごとのボケの大きさを推定 → 場所ごとに復元」と「画素ごとに直接予測する CNN」）
# - JNB（Shi+, CVPR 2015）、EBDB（Karaali+, TIP 2018）、DMENet（Lee+, CVPR 2019）：ボケの大きさのマップを推定してから、場所ごとに逆畳み込み
# - DPDNet（Abuolaim & Brown, ECCV 2020）：画素ごとに直接予測する CNN。DPDD データセットで上記の 2 段階法を 0.5〜0.7 dB 上回った
# - J. Lee et al., *IFAN*, CVPR 2021／H. Son et al., *KPAC*, ICCV 2021：画素ごとのデブラー用フィルタ（逆カーネル）を CNN で予測
# - Y. Quan et al., *Implicit Neural Inverse Kernels*, ICCV 2023：逆カーネルを辞書の線形結合で表し、構造的な制約を加える
# - K. Yanny et al., *MultiWienerNet*, Optica 2022：場所ごとに異なる PSF を、複数の Wiener フィルタの重み付き和で扱う（顕微鏡）
#
# 画像：Kodak 標準テスト画像（r0k.us/graphics/kodak/）
#
# **使うモデル・ネットワーク**（学習済みの重みは、各章で自動的にダウンロードします）
# | 章 | モデル・ネットワーク | 役割 | 出典・ライセンス |
# |---|---|---|---|
# | 2.〜 | **Depth Anything V2 Small**（DINOv2 の ViT-S ＋ DPT、約 2,500 万パラメータ、学習済み） | ボケ画像 1 枚から奥行き（視差）を推定 | L. Yang+, NeurIPS 2024／Apache-2.0 |
# | 1.〜4. | **LSIK・DA-LSIK**（逆カーネル 15×15 または 21×21 × 17 個＋混ぜ方の重み 289 個） | 講師の手法（最小二乗法で学習） | このノートブックで学習 |
# | 6. | 12 層の CNN（約 37 万パラメータ） | 画素ごとに直接予測する比較用 | このノートブックで学習 |
# | 7. | **IFAN・KPAC・INIKNet・MultiWienerNet**（公式コードのネットワーク） | 先行研究との比較 | 各公式リポジトリ（7. に記載）。重みはこのノートブックで学習し直す |
# | 8. | **DRUNet**（U-Net 型のノイズ除去 CNN、約 3,260 万パラメータ、学習済み。演習 05 と同じ） | DA-LSIK-PB／PH の仕上げ（重みを少し調整） | K. Zhang+, *Plug-and-Play Image Restoration with Deep Denoiser Prior*, TPAMI 2021（github.com/cszn/DPIR）／MIT |
# | 9. | **DiffBIR v2.1**（SwinIR ＋ Stable Diffusion 2.1-base ＋ IRControlNet、学習済み） | DA-LSIK-Diff のベース | X. Lin+, ECCV 2024（github.com/XPixelGroup/DiffBIR）／Apache-2.0。Stable Diffusion 2.1 の重みは CreativeML Open RAIL++-M |

# %% [setup]

# %% [markdown]
# ## 1. 講師の論文の再現：PSF が一定の場合
# - 学習：Kodak の 3 枚（01, 13, 23）をボケさせ、15×15 画素 → 中心画素 の組を約 22 万個作って最小二乗法で解く（論文と同じ規模）
# - 論文の工夫：ブロックごとに平均を引き標準偏差で割ってから学習（明るさ・コントラストの違いを吸収）。結果は同じく 15×15 の線形フィルタになる
# - 比較する Wiener フィルタは、**論文と同じく 15×15 の空間フィルタ** にしたもの（＋参考に大きさ無制限の FFT 版）。λ はテスト画像ごとに最良の値を探す（Wiener に有利な条件）
# - 実際のボケが **Wiener の仮定（ガウス）と違う場合**（円盤状のピンボケ）も試す。Wiener はガウスの σ と λ をテスト画像で最良に選ぶ
#
# ボケ画像は論文と同じく 8 bit に量子化するだけで、ノイズは加えません。

# %%
import cv2

KODAK = "https://r0k.us/graphics/kodak/kodak/kodim{}.png"
load = lambda n: to_np(Image.open(download(KODAK.format(n))))
TRAIN, TESTS = ["01", "13", "23"], ["05", "08", "21"]     # 学習は 3 枚だけ（講師の論文と同じ考え方）
train_imgs = [load(t) for t in TRAIN]
test_imgs = {t: load(t) for t in TESTS}
q8 = lambda x: (np.round(np.clip(x, 0, 1) * 255) / 255).astype(np.float32)
c = slice(24, -24)                                          # 画像の端は評価から除く
P = lambda a, g: psnr(np.clip(a, 0, 1)[c, c], g[c, c])

def apply_kernel(b, k):
    return cv2.filter2D(b, -1, k.astype(np.float32), borderType=cv2.BORDER_REFLECT)

def gauss_psf(s):
    r = int(np.ceil(4 * s)); g = cv2.getGaussianKernel(2 * r + 1, s); return (g @ g.T).astype(np.float32)

def disk_psf(rad):
    k = np.zeros((2 * rad + 1,) * 2, np.float32); cv2.circle(k, (rad, rad), rad, 1, -1); return k / k.sum()

def sample_patches(pairs, n_each, ks, seed=0):
    """ボケ画像の ks×ks 画素（説明変数）と、鮮明画像の中心画素（目的変数）の組をランダムに取り出す"""
    rng = np.random.default_rng(seed); r = ks // 2; idx = np.arange(-r, r + 1)
    Xs, ys = [], []
    for b, x in pairs:
        H, W = b.shape[:2]
        yy, xx, cc = rng.integers(r, H - r, n_each), rng.integers(r, W - r, n_each), rng.integers(0, 3, n_each)
        Xs.append(b[yy[:, None, None] + idx[None, :, None], xx[:, None, None] + idx[None, None, :], cc[:, None, None]].reshape(n_each, -1))
        ys.append(x[yy, xx, cc])
    return np.concatenate(Xs).astype(np.float64), np.concatenate(ys).astype(np.float64)

def ls_kernel(pairs, ks, n_each=40000, lam=1e-3, standardize=False, seed=0):
    """最小二乗法で逆カーネルを求める（係数を ks×ks に並べたものがフィルタ）"""
    X, y = sample_patches(pairs, n_each, ks, seed)
    if standardize:                                         # 論文：ブロックごとに標準化し、目的変数は p − p'
        m, s = X.mean(1, keepdims=True), X.std(1, keepdims=True) + 1e-3
        Z = (X - m) / s; ci = ks * ks // 2
        t = (y - m[:, 0]) / s[:, 0] - Z[:, ci]
        w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ t)
        k = w - w.sum() / ks ** 2; k[ci] += 1                # 予測 = x_c + w·(x − 平均) なので、同じ働きの線形フィルタに直せる
    else:
        k = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
    return k.reshape(ks, ks).astype(np.float32)

def otf(k, shape):                                          # PSF の周波数応答
    p = np.zeros(shape, np.float32); p[:k.shape[0], :k.shape[1]] = k
    return np.fft.fft2(np.roll(p, (-(k.shape[0] // 2), -(k.shape[1] // 2)), (0, 1)))

def wiener_full(b, k, lam, pad=48):                         # 大きさ無制限の Wiener（FFT）
    bp = cv2.copyMakeBorder(b, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    Kf = otf(k, bp.shape[:2]); Hf = np.conj(Kf) / (np.abs(Kf) ** 2 + lam)
    return np.stack([np.real(np.fft.ifft2(np.fft.fft2(bp[..., i]) * Hf)) for i in range(3)], -1)[pad:-pad, pad:-pad]

def wiener_kernel(k, lam, n, size=256):                     # Wiener を n×n の空間フィルタにしたもの（論文の比較対象）
    Kf = otf(k, (size, size)); h = np.fft.fftshift(np.real(np.fft.ifft2(np.conj(Kf) / (np.abs(Kf) ** 2 + lam))))
    m = size // 2; h = h[m - n // 2:m + n // 2 + 1, m - n // 2:m + n // 2 + 1]
    return h / h.sum()                                      # 打ち切ると明るさ（直流成分）がずれるので、合計 1 に直す（Wiener に有利な補正）

N15, LAMS, GSIG = 15, 10.0 ** np.arange(-6, -0.9, 0.25), np.arange(1.0, 2.51, 0.25)
paper, paper_imgs = {}, {}                                  # paper_imgs：画像 05 の復元結果（あとで表示する）
for case, true_psf in {"Gaussian sigma=1.5": gauss_psf(1.5), "disk r=3 (defocus)": disk_psf(3)}.items():
    deg = lambda x: q8(apply_kernel(x, true_psf))
    pairs = [(deg(x), x) for x in train_imgs]
    k_ls = ls_kernel(pairs, N15, n_each=74000)
    k_std = ls_kernel(pairs, N15, n_each=74000, standardize=True)
    rows = {}
    for t, g in test_imgs.items():
        b = deg(g)
        # Wiener：ガウス PSF を仮定し、σ と λ をテスト画像で最良に選ぶ
        sg, lm = max(((s, l) for s in GSIG for l in LAMS), key=lambda p: P(wiener_full(b, gauss_psf(p[0]), p[1]), g))
        best = lambda outs: max(outs, key=lambda o: P(o, g))   # λ を変えた結果のうち、最も PSNR の高いもの
        imgs = {"input": b,
                "LSIK 15x15": apply_kernel(b, k_ls),
                "LSIK 15x15, block-standardized (paper)": apply_kernel(b, k_std),
                "Wiener 15x15 (Gaussian assumed)": best(apply_kernel(b, wiener_kernel(gauss_psf(sg), l, N15)) for l in LAMS),
                "Wiener full FFT (Gaussian assumed)": wiener_full(b, gauss_psf(sg), lm),
                "Wiener full FFT (exact PSF, reference)": best(wiener_full(b, true_psf, l) for l in LAMS)}
        rows[t] = {n: P(o, g) for n, o in imgs.items()}
        if t == "05":
            paper_imgs[case] = imgs
    paper[case] = rows
    print(f"\n[{case}]  PSNR (dB)        " + "  ".join(TESTS) + "   mean")
    for name in rows[TESTS[0]]:
        v = [rows[t][name] for t in TESTS]
        print(f"  {name:40s} " + " ".join(f"{a:5.2f}" for a in v) + f"  {np.mean(v):5.2f}")
    if case.startswith("Gaussian"):
        kernels_paper = (k_ls, k_std, wiener_kernel(gauss_psf(1.5), 10 ** -3.5, N15))

# %% [markdown]
# **結果の読み方**
# - **同じ 15×15 の大きさ** で比べると、最小二乗の逆カーネルは Wiener を上回る（論文の比較条件）。Wiener の理論解は本来無限に広がるので、15×15 で打ち切ると損をする。
#   最小二乗法は「15×15 という制約の中での最適」を直接求める
# - 実際のボケが **ガウスでない（円盤状のピンボケ）** と、ガウスを仮定した Wiener は大きく崩れるが、最小二乗法はデータから学ぶので影響を受けない。
#   論文で 4〜6 dB の差がついたのは、このような「仮定した PSF と実際のボケのずれ」が効いていると考えられる
# - ガウスぼけでは、PSF を **正確に** 与え大きさ無制限にした理想的な Wiener（最下行）とほぼ同等。逆カーネルは PSF の式を知らずに、データだけでこの理論上の上限に近づく。
#   円盤状のボケでは、PSF の形まで正確に分かっていれば理想的な Wiener がさらに上だが、逆カーネルも大きくすると近づく（21×21 で約 28.1 dB、31×31 で約 29.1 dB）
# - ブロックごとの標準化（論文の工夫）は、明るさやコントラストに依らず誤差を均等に扱うため、PSNR（2 乗誤差）では素の最小二乗とわずかに差が出る

# %% [markdown]
# **復元した画像**（テスト画像 05 の一部を拡大。数値は画像全体の PSNR）
# - 上段：ガウスぼけ（σ=1.5）。ボケが小さいので、差は細かい模様や輪郭に出る
# - 下段：円盤状のピンボケ（半径 3 画素）。ガウスを仮定した Wiener と、データから学んだ逆カーネルの差が大きくなる

# %%
g05 = test_imgs["05"]
box1 = (slice(30, 290), slice(380, 640))                   # 5. と同じ場所を拡大
cols = ["input", "LSIK 15x15, block-standardized (paper)", "Wiener 15x15 (Gaussian assumed)",
        "Wiener full FFT (Gaussian assumed)", "Wiener full FFT (exact PSF, reference)"]
for case, imgs in paper_imgs.items():
    print(case)
    show([np.clip(imgs[n][box1], 0, 1) for n in cols] + [g05[box1]],
         [f"{n}\n{P(imgs[n], g05):.2f} dB" for n in cols] + ["ground truth"], size=3.0)

# %% [markdown]
# **学習した逆カーネル**（15×15 の係数を画像として表示）。最小二乗の逆カーネルは、Wiener を 15×15 で打ち切ったものとよく似た形になる

# %%
show([kernels_paper[0], kernels_paper[1], kernels_paper[2]],
     ["LSIK inverse kernel 15x15", "LSIK, block-standardized (paper)", "Wiener kernel cut to 15x15"], size=3.2)

# %% [markdown]
# ## 2. 奥行きで変わるピンボケを作る
# 薄肉レンズでは、ボケの直径は **視差（奥行きの逆数）とピント位置の視差の差** に比例します。
# 奥行き推定 AI の出力は視差に相当するので、ボケの大きさ σ（ガウスぼけの標準偏差）を σ = A·|D − D_f| とします。
# ここでは、最も手前（D_f = 1）にピントを合わせ、最も遠い所で σ = 4 画素とします。ノイズ（σ=0.005）も加えます。

# %%
from transformers import pipeline

A, DF, NOISE = 4.0, 1.0, 0.005
depth_pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if DEVICE == "cuda" else -1)

def disparity(img, top=99):
    """奥行き推定 AI の出力（視差：大きいほど手前）を 0〜1 に正規化"""
    d = np.asarray(depth_pipe(to_pil(img))["predicted_depth"], np.float32)
    d = cv2.resize(d, img.shape[1::-1], interpolation=cv2.INTER_CUBIC)
    lo, hi = np.percentile(d, 1), np.percentile(d, top)
    return np.clip((d - lo) / (hi - lo), 0, 1)

def gblur(x, s):
    return x if s < 1e-3 else cv2.GaussianBlur(x, (0, 0), s, borderType=cv2.BORDER_REFLECT)

def sv_blur(x, smap, step=0.1):
    """場所ごとに σ が違うボケ（σ を細かい段階に分け、隣り合う段階を線形補間）"""
    out = np.zeros_like(x)
    for s in np.arange(0, smap.max() + step, step):
        w = np.clip(1 - np.abs(smap - s) / step, 0, 1)[..., None]
        if w.max() > 0:
            out += w * gblur(x, s)
    return out

def add_noise(x, seed=0):
    return np.round(np.clip(x + np.random.default_rng(seed).normal(0, NOISE, x.shape), 0, 1) * 255) / 255

scenes = {}
for t, g in test_imgs.items():
    D_true = disparity(g)                                   # 「正解の奥行き」（鮮明画像から推定したものを正解とみなす）
    smap = A * np.abs(D_true - DF)
    scenes[t] = dict(gt=g, D=D_true, smap=smap, blur=add_noise(sv_blur(g, smap)))
s05 = scenes["05"]
show([s05["gt"], s05["D"], s05["smap"], s05["blur"]], ["ground truth", "depth (disparity)", "blur size sigma (0-4 px)", "defocused input"], size=4)

# %% [markdown]
# ## 3. σ 別の逆カーネルを用意し、奥行きから場所ごとに切り替える
# 1. ボケの大きさ σ = 0, 0.25, …, 4 の 17 段階それぞれで、21×21 の逆カーネルを最小二乗法で学習する（一様なボケ・学習画像 3 枚）
# 2. **ボケ画像から** 奥行きを推定し、少しぼかして滑らかにする ＝「近くの場所の逆カーネルは似ている」という制約
# 3. 奥行き → σ（カメラのピント位置・絞りは既知とする）
# 4. σ に応じて、隣り合う 2 つの逆カーネルの結果を線形補間（逆カーネル自体を場所ごとに滑らかに補間するのと同じ）

# %%
KS = 21
LEVELS = np.arange(0, 4.01, 0.25)
t0 = time.time()
kernels = [ls_kernel([(add_noise(gblur(x, s), seed=100 + k), x) for k, x in enumerate(train_imgs)], KS, seed=int(s * 100)) for s in LEVELS]
print(f"逆カーネル {len(kernels)} 個（各 {KS}x{KS}）を学習: {time.time()-t0:.0f}s")

def best_K(sig):                                 # Wiener の定数 K も、学習画像で最良の値を選ぶ（公平な比較のため）
    if sig < 1e-3: return 1.0
    bs = [add_noise(gblur(x, sig), seed=100 + k) for k, x in enumerate(train_imgs)]
    return max([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2], key=lambda K: np.mean([P(wiener_full(b, gauss_psf(sig), K), x) for b, x in zip(bs, train_imgs)]))
Ks = [best_K(s) for s in LEVELS]
wiener_at = lambda b, s, K: b if s < 1e-3 else wiener_full(b, gauss_psf(s), K)

def est_sigma(blur, df=DF):
    """ボケ画像から奥行きを推定 → 平滑化 → σ"""
    return A * np.abs(cv2.GaussianBlur(disparity(blur), (0, 0), 4) - df)

def bin_weights(s):
    """σ に対する各段階の補間の重み（H×W×17、隣り合う 2 段階だけが 0 でない）"""
    st = LEVELS[1] - LEVELS[0]; s = np.clip(s, LEVELS[0], LEVELS[-1])
    return np.stack([np.clip(1 - np.abs(s - l) / st, 0, 1) for l in LEVELS], -1).astype(np.float32)   # float32：Colab のメモリ節約

def blend(outs, s):
    return np.einsum("hwl,lhwc->hwc", bin_weights(s), np.stack(outs))

for t, sc in scenes.items():
    sc["s_est"] = est_sigma(sc["blur"])
    sc["ls_outs"] = [apply_kernel(sc["blur"], k) for k in kernels]
    print(f"{t}: 推定した σ の誤差（平均） {np.abs(sc['s_est'] - sc['smap'])[c, c].mean():.2f} 画素")

# %% [markdown]
# ## 4. 奥行き推定の誤差まで含めて「混ぜ方」を学習する
# 奥行きの推定には誤差があり、**σ を大きく見積もりすぎると強調しすぎて大きく壊れ、小さく見積もると少しボケが残るだけ** という非対称性があります。
# そこで、学習画像 3 枚にもピント位置を変えた奥行きボケをかけ、**同じ手順で σ を推定** したうえで、
# 「推定 σ がこの値のとき、17 個の逆カーネルの出力をどう混ぜれば正解に最も近いか」（17 × 17 = 289 個の重み）を最小二乗法で求めます。
# 推定の誤差の傾向を、データから学ぶことになります。
#
# **学習データの作り方**（下の図）：学習画像の奥行きはそのままで、ピント位置 D_f を 1.0〜0.5 の 6 通りに変えると、同じ画像からボケ方の違う入力ができる
# （σ の図は明るいほどボケが大きい）。3 枚 × 6 通り = 18 組を使います。

# %%
x13 = train_imgs[TRAIN.index("13")]; D13 = disparity(x13)
ims, tts = [x13, D13], ["ground truth (training image 13)", "depth D (bright = near)"]
for df in (1.0, 0.8, 0.6):
    sm = np.clip(A * np.abs(D13 - df), 0, 4)
    ims += [sm / 4, sv_blur(x13, sm)]; tts += [f"blur size sigma (D_f = {df})", f"input (D_f = {df})"]
show([ims[i] for i in (0, 2, 4, 6, 1, 3, 5, 7)], [tts[i] for i in (0, 2, 4, 6, 1, 3, 5, 7)], size=3.6, rows=2)

# %%
def mix_features(outs, s):
    """特徴 = （推定 σ の区間の重み）×（各逆カーネルの出力）。これに重みを掛けて足すと復元画像になる"""
    O = np.stack(outs, -1).astype(np.float32)               # H×W×3×17（float32：Colab のメモリ節約）
    return (bin_weights(s)[:, :, None, :, None] * O[:, :, :, None, :]).reshape(*O.shape[:3], -1)

t0 = time.time()
rng = np.random.default_rng(1)
n_w = len(LEVELS) ** 2
XtX, Xty, n_px = np.zeros((n_w, n_w)), np.zeros(n_w), 0   # 正規方程式の XᵀX と Xᵀy を少しずつ足していく
for k, x in enumerate(train_imgs):                          # （全部の X を溜めると Colab 無料枠のメモリ 12.7GB を超えるため）
    D = disparity(x)
    for df in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):              # ピント位置を変えて、いろいろな奥行きボケを作る
        sm = np.clip(A * np.abs(D - df), 0, 4)
        b = add_noise(sv_blur(x, sm), seed=200 + k)
        F = mix_features([apply_kernel(b, kk) for kk in kernels], est_sigma(b, df))
        m = rng.random(F.shape[:3]) < 0.08; m[:24] = m[-24:] = m[:, :24] = m[:, -24:] = False
        Fm, ym = F[m].astype(np.float64), x[m].astype(np.float64)
        XtX += Fm.T @ Fm; Xty += Fm.T @ ym; n_px += len(ym)
        del F, Fm
W_mix = np.linalg.solve(XtX + 1e-4 * np.eye(n_w), Xty).astype(np.float32)
print(f"混ぜ方の重み {W_mix.size} 個を学習（{n_px//1000}k 画素, {time.time()-t0:.0f}s）")

res = {}
for t, sc in scenes.items():
    b, sm, se = sc["blur"], sc["smap"], sc["s_est"]
    i_fix = int(np.argmin(np.abs(LEVELS - sm.mean())))      # 「PSF は一定」とみなす場合は平均的な σ を使う
    wi_outs = [wiener_at(b, s, K) for s, K in zip(LEVELS, Ks)]
    sc["out"] = {
        "input (defocused)": b,
        "Wiener, fixed PSF": wi_outs[i_fix],
        "LSIK, fixed PSF": sc["ls_outs"][i_fix],
        "Wiener, local (est. depth)": blend(wi_outs, se),
        "LSIK, local (est. depth)": blend(sc["ls_outs"], se),
        "LSIK, local + learned mixing": mix_features(sc["ls_outs"], se) @ W_mix,
        "LSIK, local (true depth)": blend(sc["ls_outs"], sm),
    }
    res[t] = {n: P(o, sc["gt"]) for n, o in sc["out"].items()}
print("PSNR (dB)                              " + "  ".join(TESTS) + "   mean")
for n in res[TESTS[0]]:
    v = [res[t][n] for t in TESTS]
    print(f"{n:38s} " + " ".join(f"{a:5.2f}" for a in v) + f"  {np.mean(v):5.2f}")

# %% [markdown]
# ## 5. 比較（拡大）

# %%
assert "scenes" in globals() and "out" in scenes["05"], "2〜4 のセルが実行されていません。上から順に実行してください（ランタイムが再起動した場合も最初から）"
s05 = scenes["05"]
o, g = s05["out"], s05["gt"]
names = ["input (defocused)", "Wiener, fixed PSF", "Wiener, local (est. depth)", "LSIK, local (est. depth)", "LSIK, local + learned mixing"]
box = (slice(30, 290), slice(380, 640))                   # 奥の選手（ボケが大きい所）を拡大
show([np.clip(o[n][box], 0, 1) for n in names] + [g[box]], [f"{n}\n{P(o[n], g):.2f} dB" for n in names] + ["ground truth"], size=3.2)

# %% [markdown]
# ## 6. 画素ごとに直接予測する CNN と比べる
# 混ぜ方の重みと同じ「奥行き由来のボケ」（学習画像 3 枚 × ピント位置 6 通り）で、小さな CNN（12 層、約 37 万パラメータ）に「ボケ画像 → 鮮明画像」を直接学習させます。
# CNN には奥行きを与えません（ボケの大きさも画像から自分で判断する）。T4 で約 10〜15 分かかるので、不要なら `RUN_CNN = False` にしてください。
# - 学習の条件（小片 96×96・バッチ 16・6,000 回・学習率は 70% の所で 1/5・二乗誤差）は、7. の先行研究と同じ。ここで作る `train_net` を 7. でも使います
# - 学習を長くしすぎると、学習画像 3 枚に過学習して、かえって下がります
# - スライドの表は乱数 3 通り（`SEEDS = [0, 1, 2]`）の平均 ± 標準偏差。ここでは時間の都合で乱数 0 の 1 回だけ学習します（±0.1 dB 程度ばらつく）

# %%
RUN_CNN = True
ITERS, SEED = 6000, 0
import torch.nn as nn
import torch.nn.functional as nnF                                 # F は 4. で変数名に使ったので別名にする
train_pairs = []                                                  # 混ぜ方の重みと同じ「奥行き由来のボケ」（3 枚 × ピント位置 6 通り）
for x in train_imgs:
    D = disparity(x)
    for df in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        train_pairs.append((sv_blur(x, np.clip(A * np.abs(D - df), 0, 4)).astype(np.float32), x))

def make_sampler(seed):
    rng = np.random.default_rng(seed)
    def sample(Pz=96):
        b, x = train_pairs[rng.integers(len(train_pairs))]
        y0, x0 = rng.integers(0, x.shape[0] - Pz), rng.integers(0, x.shape[1] - Pz)
        bb = np.clip(b[y0:y0 + Pz, x0:x0 + Pz] + rng.normal(0, NOISE, (Pz, Pz, 3)), 0, 1)   # ノイズは小片ごとに加える
        return bb.astype(np.float32), x[y0:y0 + Pz, x0:x0 + Pz]
    return sample

def evaluate_net(net):
    net.eval(); out = {}
    with torch.no_grad():
        for t, sc in scenes.items():
            out[t] = net(to_tensor(sc["blur"]).to(DEVICE))[-1][0].permute(1, 2, 0).cpu().numpy()
    net.train(); return out

def train_net(name, net, lr, multiscale=False, seed=SEED):
    """すべての手法で同じ条件（同じ小片の取り出し方・バッチ 16・96 画素・ITERS 回）で学習し、評価 3 枚の結果を scenes に入れる"""
    torch.manual_seed(seed); net = net.to(DEVICE)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(ITERS * 0.7)], 0.2)
    sample = make_sampler(seed); t0 = time.time()
    for it in range(1, ITERS + 1):
        bb, gg = zip(*[sample() for _ in range(16)])
        xb = torch.from_numpy(np.stack(bb)).permute(0, 3, 1, 2).to(DEVICE)
        xg = torch.from_numpy(np.stack(gg)).permute(0, 3, 1, 2).to(DEVICE)
        outs = net(xb)                                            # どの手法も「出力のリスト」を返す（最後が最終結果）
        if multiscale:                                            # INIKNet：公式と同じく 1/4・1/2・等倍の 3 つの出力すべてに損失
            ys = [nnF.interpolate(xg, scale_factor=0.25, mode="bilinear"), nnF.interpolate(xg, scale_factor=0.5, mode="bilinear"), xg]
            loss = sum(nnF.mse_loss(o, y) for o, y in zip(outs, ys))
        else:
            loss = nnF.mse_loss(outs[-1], xg)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % (ITERS // 3) == 0:
            o = evaluate_net(net)
            v = [P(o[t], scenes[t]["gt"]) for t in TESTS]
            print(f"{name:16s} iter {it:5d}  loss {loss.item():.5f}  PSNR " + " ".join(f"{a:5.2f}" for a in v) + f"  mean {np.mean(v):5.2f}  ({time.time()-t0:.0f}s)")
    for t, sc in scenes.items():
        sc["out"][name] = o[t]; res[t][name] = P(o[t], sc["gt"])
    n_params[name] = npar
    return time.time() - t0

class SmallCNN(nn.Module):
    def __init__(self, ch=64, L=12):
        super().__init__()
        layers = [nn.Conv2d(3, ch, 3, padding=1), nn.ReLU(True)]
        for _ in range(L - 2):
            layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.f = nn.Sequential(*layers, nn.Conv2d(ch, 3, 3, padding=1))
    def forward(self, x):
        return [x + self.f(x)]                                    # 入力との差（残差）を学習

def print_table(names):
    print("PSNR (dB)                              " + "  ".join(TESTS) + "   mean   params")
    for n in names:
        if n in res[TESTS[0]]:
            v = [res[t][n] for t in TESTS]
            print(f"{n:38s} " + " ".join(f"{a:5.2f}" for a in v) + f"  {np.mean(v):5.2f}   {n_params.get(n, '')}")

n_ls = len(kernels) * KS * KS + W_mix.size
n_params = {"LSIK, local + learned mixing": f"{n_ls/1e3:.1f}k"}
if RUN_CNN:
    t_ = train_net("CNN (per-pixel)", SmallCNN(), 1e-3)
    n_params["CNN (per-pixel)"] = f"{n_params['CNN (per-pixel)']/1e3:.0f}k"
    print(f"学習 {t_:.0f}s")
    print_table(["input (defocused)", "LSIK, local (est. depth)", "LSIK, local + learned mixing", "LSIK, local (true depth)", "CNN (per-pixel)"])
    o = s05["out"]
    cmp = ["CNN (per-pixel)", "LSIK, local + learned mixing", "LSIK, local (true depth)"]
    show([np.clip(o[n][box], 0, 1) for n in cmp] + [g[box]], [f"{n}\n{P(o[n], g):.2f} dB" for n in cmp] + ["ground truth"], size=3.2)

# %% [markdown]
# ## 7. 先行研究を同じ条件で学習し直す（IFAN・KPAC・INIKNet・MultiWienerNet）
# 「画素ごとの逆カーネル（デブラー用フィルタ）を CNN で予測する」最新の手法を、**公式コードのネットワーク** を使って、6. と同じ学習画像 3 枚・同じ奥行きボケ・同じ条件で学習し直します。
# - **IFAN**（Lee+, CVPR 2021, github.com/codeslake/IFAN, AGPL-3.0）：画素ごとに分離型のフィルタを予測して特徴に掛ける。デュアルピクセル（左右 2 枚）の画像がないので、視差・再ボケの補助損失は外し、復元の損失だけで学習
# - **KPAC**（Son+, ICCV 2021, github.com/HyeongseokSon1/KPAC, AGPL-3.0）：同じ重みの畳み込みを間隔（dilation）を変えて並列に掛け、注意で混ぜる。公式は TensorFlow 1 なので、2 段の版（Defocus_Deblur_Net6_ms）を下で PyTorch に移植
# - **INIKNet**（Quan+, ICCV 2023, github.com/xinyao240/INIKNet, MIT）：逆カーネルを辞書の線形結合で表す。損失は公式の多重スケールの二乗誤差（知覚的な損失・周波数の損失は省略）
# - **MultiWienerNet**（Yanny+, Optica 2022, github.com/Waller-Lab/MultiWienerNet, BSD-3）：9 個の PSF（ガウス σ = 0〜4 で初期化、学習する）で Wiener 逆畳み込みをして U-Net で混ぜる。顕微鏡用（白黒）をカラー用に改造
#
# 公式コードはこのセルで GitHub から取得します（ライセンスは各リポジトリを参照）。T4 で合計 約 60〜90 分（INIKNet だけで約 60 分）。不要なら `RUN_PRIOR = False`、一部だけなら `PRIOR` から外してください。

# %%
RUN_PRIOR = True
PRIOR = ["MultiWienerNet", "KPAC", "IFAN", "INIKNet"]
EXT = os.path.abspath("ext")
REPOS = {"IFAN": ("https://github.com/codeslake/IFAN.git", "66bf136"), "INIKNet": ("https://github.com/xinyao240/INIKNet.git", "10c2a32"),
         "MultiWienerNet": ("https://github.com/Waller-Lab/MultiWienerNet.git", "f49a38e")}   # 動作を確認した版
if RUN_PRIOR:
    import types, json, importlib
    os.makedirs(EXT, exist_ok=True)
    for name, (url, rev) in REPOS.items():
        d = os.path.join(EXT, name)
        if name in PRIOR and not os.path.exists(d):
            subprocess.run(["git", "clone", "-q", url, d], check=True)
            subprocess.run(["git", "-C", d, "checkout", "-q", rev], check=True)

    def clear_models():                                           # IFAN と MultiWienerNet はどちらも「models」という名前のパッケージなので入れ替える
        for m in [k for k in sys.modules if k == "models" or k.startswith("models.")]:
            del sys.modules[m]

    # ---- IFAN（公式のネットワーク。視差を推定する枝（DME）の出力も特徴として使うのは公式と同じ。補助損失だけ使わない）
    if "IFAN" in PRIOR:
        clear_models(); sys.path.insert(0, os.path.join(EXT, "IFAN"))
        from models.archs.IFAN import Network as IFANNet
        from models.IAC import IAC
        sys.path.pop(0); clear_models()
        class IFAN(IFANNet):
            def forward(self, C):                                     # 公式の forward から、学習用の補助出力を除いたもの
                f1 = self.conv1_3(self.conv1_2(self.conv1_1(C)))
                f2 = self.conv2_3(self.conv2_2(self.conv2_1(f1)))
                f3 = self.conv3_3(self.conv3_2(self.conv3_1(f2)))
                f_C = self.conv4_3(self.conv4_2(self.conv4_1(f3)))
                f = self.kconv1_3(self.kconv1_2(self.kconv1_1(C)))
                f = self.kconv2_3(self.kconv2_2(self.kconv2_1(f)))
                f = self.kconv3_3(self.kconv3_2(self.kconv3_1(f)))
                f = self.kconv4_3(self.kconv4_2(self.kconv4_1(f)))
                DM = self.DME(f)
                f = self.conv4_4(torch.cat([f, self.conv_DME(DM)], 1))
                f = IAC(f_C, self.F(f), self.N, self.ch4, self.Fs)    # 画素ごとのフィルタを特徴に掛ける（IAC 層）
                f = self.conv_res(f)
                f = self.upconv3_2(self.upconv3_1(self.upconv3_u(f) + f3))
                f = self.upconv2_2(self.upconv2_1(self.upconv2_u(f) + f2))
                f = self.upconv1_2(self.upconv1_1(self.upconv1_u(f) + f1))
                return [self.out_res(f) + C]
        def make_ifan():
            cfg = types.SimpleNamespace(device=DEVICE, ks=3, Fs=3, res_num=2, ch=32, wiF=1.5, N=17)   # 公式の configs/config_IFAN.py と同じ
            net = IFAN(cfg); net.init_F(); return net

    # ---- KPAC（公式 TensorFlow 1 版の 2 段のネットワークを PyTorch に移植）
    lrelu = lambda x: nnF.leaky_relu(x, 0.2)
    class KPACBlock(nn.Module):
        def __init__(self, cin, ch=48, ks=5):
            super().__init__()
            self.k = nn.Parameter(torch.randn(ch, cin, ks, ks) * 0.04)    # 5 つの間隔（dilation）で共有するカーネル
            self.sca = nn.ModuleList([nn.Conv2d(cin, 32, 5, padding=4, dilation=2), nn.Conv2d(32, 32, 5, padding=4, dilation=2),
                                      nn.Conv2d(32, 16, 5, padding=4, dilation=2), nn.Conv2d(16, 16, 5, padding=4, dilation=2)])
            self.sca_out = nn.Conv2d(16, 5, 5, padding=2)
            self.sha1, self.sha2 = nn.Conv2d(cin, ch // 4, 1), nn.Conv2d(ch // 4, ch, 1)
            self.cf = nn.Conv2d(5 * ch, 2 * ch, 3, padding=1); self.ch = ch
        def forward(self, n):
            nn_ = torch.cat([lrelu(nnF.conv2d(n, self.k, padding=2 * r, dilation=r)) for r in range(1, 6)], 1)
            a = n
            for c_ in self.sca: a = lrelu(c_(a))
            a = torch.sigmoid(self.sca_out(a)).repeat_interleave(self.ch, 1)   # スケールの注意（場所ごと）
            s = torch.sigmoid(self.sha2(lrelu(self.sha1(n.mean((2, 3), keepdim=True))))).repeat(1, 5, 1, 1)   # 形の注意
            return lrelu(self.cf(nn_ * a * s))
    class KPAC(nn.Module):
        def __init__(self, ch=48, bs=2):
            super().__init__()
            C = lambda i, o, k, s=1: nn.Conv2d(i, o, k, s, padding=k // 2)
            self.c0, self.c0_2 = C(3, ch, 5), C(ch, ch, 3)
            self.c2, self.c2_2 = C(ch, ch, 3, 2), C(ch, ch, 3)
            self.c3, self.c3_2 = C(ch, 2 * ch, 3, 2), C(2 * ch, 2 * ch, 3)
            self.blocks = nn.ModuleList([KPACBlock(2 * ch, ch) for _ in range(bs)])
            self.cm_1, self.cm_2 = C(2 * ch * (bs + 1), 2 * ch, 3), C(2 * ch, 2 * ch, 3)
            self.d1 = nn.ConvTranspose2d(2 * ch, ch, 4, 2, 1); self.d1_2 = C(2 * ch, ch, 3)
            self.d2 = nn.ConvTranspose2d(ch, ch, 4, 2, 1); self.d2_2 = C(2 * ch, 3, 5)
        def forward(self, x):
            f1 = lrelu(self.c0_2(lrelu(self.c0(x))))
            f2 = lrelu(self.c2_2(lrelu(self.c2(f1))))
            n = lrelu(self.c3_2(lrelu(self.c3(f2)))); stack = [n]
            for b_ in self.blocks:
                n = b_(n); stack.append(n)
            n = lrelu(self.cm_2(lrelu(self.cm_1(torch.cat(stack, 1)))))
            n = lrelu(self.d1_2(torch.cat([lrelu(self.d1(n)), f2], 1)))
            n = self.d2_2(torch.cat([lrelu(self.d2(n)), f1], 1))
            return [n + x]

    # ---- INIKNet（公式のネットワークと設定ファイル。thop は計算量を数えるためだけなので不要）
    if "INIKNet" in PRIOR:
        sys.modules.setdefault("thop", types.SimpleNamespace(profile=None))
        sys.path.insert(0, os.path.join(EXT, "INIKNet"))
        import matplotlib
        backend0 = matplotlib.get_backend()
        INIKNet = importlib.import_module("model").INIKNet
        matplotlib.use(backend0)                                  # 公式の model.py は図を表示しない設定（Agg）に変えるので、元に戻す
        INIK_CFG = json.load(open(os.path.join(EXT, "INIKNet", "model_config.json")))
        sys.path.pop(0)

    # ---- MultiWienerNet（公式の U-Net。Wiener 層は公式と同じ式を、カラー画像の各色に掛ける）
    if "MultiWienerNet" in PRIOR:
        sys.path.insert(0, os.path.join(EXT, "MultiWienerNet", "pytorch"))
        from models.unet import Unet as MWUnet
        sys.path.pop(0); clear_models()
    class MultiWiener(nn.Module):
        def __init__(self, n=9, ks=33):
            super().__init__()
            sig = np.linspace(0, 4, n); r = ks // 2; yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            ps = [np.exp(-(xx ** 2 + yy ** 2) / (2 * max(s, 0.3) ** 2)) for s in sig]
            self.psfs = nn.Parameter(torch.tensor(np.stack([p / p.sum() for p in ps]), dtype=torch.float32))   # n×ks×ks（学習する）
            self.Ks = nn.Parameter(torch.full((n, 1, 1), 1e-5))                                             # Wiener の定数（公式は 100·K）
            self.unet = MWUnet(n_channel_in=3 * n, n_channel_out=3, residual=True, down="conv", up="nearest", activation="relu")
            self.n, self.r = n, r
        def forward(self, y):
            B, C, H, W = y.shape; p = 24
            yp = nnF.pad(y, (p, p, p, p), mode="reflect"); Hp, Wp = H + 2 * p, W + 2 * p
            k = torch.zeros(self.n, Hp, Wp, device=y.device)
            k[:, :2 * self.r + 1, :2 * self.r + 1] = self.psfs
            k = torch.roll(k, (-self.r, -self.r), (1, 2))                          # PSF の中心を原点へ
            Hf = torch.fft.fft2(k)[None, :, None]
            X = torch.conj(Hf) * torch.fft.fft2(yp)[:, None] / (Hf.abs() ** 2 + 100 * self.Ks.abs()[None, :, None])
            x = torch.fft.ifft2(X).real[..., p:p + H, p:p + W].reshape(B, self.n * C, H, W)
            return [self.unet(x)]

    MAKERS = {"MultiWienerNet": (MultiWiener, 1e-4, False), "KPAC": (KPAC, 1e-4, False),
              "IFAN": (lambda: make_ifan(), 1e-4, False), "INIKNet": (lambda: INIKNet(INIK_CFG), 2e-4, True)}
    for name in PRIOR:
        mk, lr, ms = MAKERS[name]
        torch.manual_seed(SEED)
        t_ = train_net(name, mk(), lr, ms)
        n_params[name] = f"{n_params[name]/1e6:.2f}M"
        print(f"  → {name} 学習 {t_:.0f}s"); torch.cuda.empty_cache()
    print_table(["input (defocused)", "LSIK, local + learned mixing", "LSIK, local (true depth)", "CNN (per-pixel)"] + PRIOR)
    o = s05["out"]
    cmp = [n for n in ["LSIK, local + learned mixing"] + PRIOR if n in o]
    show([np.clip(o[n][box], 0, 1) for n in cmp] + [g[box]], [f"{n}\n{P(o[n], g):.2f} dB" for n in cmp] + ["ground truth"], size=3.2)

# %% [markdown]
# **結果の読み方**：最新の IFAN・INIKNet と DA-LSIK（4. の「LSIK, local + learned mixing」）はほぼ同等で、KPAC・12 層の CNN はやや低い。
# 係数の数は DA-LSIK が約 7.8 千、先行研究は約 158 万〜1,048 万（250〜1,300 倍）。
# ただし先行研究は大規模データ向けの設計で、ここでは学習 3 枚・6,000 回・二乗誤差だけなので、大規模データでは先行研究が上回る可能性が高い。

# %% [markdown]
# ## 8. DA-LSIK の出力から、さらに追い込む（DA-LSIK-PB / DA-LSIK-PH）
# DA-LSIK の出力を出発点に、ノイズ除去の CNN と交互に **4 回** くり返して仕上げます。
# CNN は **DRUNet**（K. Zhang+, TPAMI 2021。U-Net 型、約 3,260 万パラメータ。ノイズの強さを 4 枚目のチャンネルで受け取る。演習 05 の DPIR と同じ学習済みモデル drunet_color、github.com/cszn/DPIR、MIT）。
# くり返し全体を、DA-LSIK と同じ学習画像 3 枚で「DA-LSIK の出力 → 4 回のくり返し → 正解」となるように学習します（**Deep Unfolding**：各回の強さと DRUNet の重みを調整）。
# - **DA-LSIK-PB**（Pull-Back：引き戻し）：毎回、DA-LSIK の出力 G(y) に引き戻す　x = (G(y) + ρ_k z) / (1 + ρ_k) → DRUNet。物理の式は使わない
# - **DA-LSIK-PH**（Physics：物理）：奥行きで変わるボケの式 H で、観測 y に合うように解く（FFT）　x = argmin ‖y − Hx‖² + ρ_k ‖x − z‖² → DRUNet
# - （比較）DA-LSIK を使わず、**ボケ画像から** 同じ「物理＋CNN」を始めたもの
#
# 損失は「L1 ＋ 隣の画素との差の L1」（輪郭がのっぺりしないように）。T4 で合計 約 30〜45 分。不要なら `RUN_UNFOLD = False`。

# %%
RUN_UNFOLD = True
if RUN_UNFOLD:
    import copy, spandrel
    t0 = time.time()
    unfold_train = []                                             # 学習画像 3 枚 × ピント位置 6 通り：ボケ画像・推定 σ・DA-LSIK の出力・正解
    for n, x in zip(TRAIN, train_imgs):
        D = disparity(x)
        for j, df in enumerate((1.0, 0.9, 0.8, 0.7, 0.6, 0.5)):
            b = add_noise(sv_blur(x, np.clip(A * np.abs(D - df), 0, 4)), seed=1000 + int(n) * 10 + j)
            se = est_sigma(b, df)
            prop = mix_features([apply_kernel(b, kk) for kk in kernels], se) @ W_mix
            unfold_train.append(dict(x=x, b=b.astype(np.float32), se=se.astype(np.float32), prop=prop.astype(np.float32)))
    print(f"学習データ {len(unfold_train)} 組を作成（{time.time()-t0:.0f}s）")

    SN, PAD, NL = NOISE, 24, len(LEVELS)
    lv = torch.tensor(LEVELS, dtype=torch.float32, device=DEVICE); st = float(LEVELS[1] - LEVELS[0])
    Keff = torch.einsum("lj,jhw->lhw", torch.from_numpy(W_mix.reshape(NL, len(kernels))).float().to(DEVICE),
                        torch.from_numpy(np.stack(kernels)).float().to(DEVICE))   # σ の段ごとの実効カーネル（DA-LSIK は線形なのでまとめられる）
    KH = Keff.shape[-1] // 2
    binw = lambda s: (1 - (s.clamp(lv[0], lv[-1]) - lv.view(1, -1, 1, 1)).abs() / st).clamp(min=0)   # (B,17,H,W)：σ の段ごとの補間の重み
    def G_dalsik(y, se):
        """DA-LSIK を torch で（y：(B,3,H,W)、se：推定 σ (B,1,H,W)）"""
        B = y.shape[0]
        o = nnF.conv2d(nnF.pad(y, (KH,) * 4, mode="reflect").reshape(B * 3, 1, *[m + 2 * KH for m in y.shape[-2:]]), Keff[:, None])
        return (binw(se)[:, None] * o.view(B, 3, NL, *y.shape[-2:])).sum(2)
    def otfs(H, W):                                               # σ の段ごとのガウスぼけの周波数応答 (17,H,W)
        fy = torch.fft.fftfreq(H, device=DEVICE)[:, None]; fx = torch.fft.fftfreq(W, device=DEVICE)[None, :]
        return torch.exp(-2 * (np.pi * lv[:, None, None]) ** 2 * (fx ** 2 + fy ** 2))
    def den(net, z, s):                                           # DRUNet：入力にノイズの強さ s の地図を足して 4 チャンネルで渡す
        H, W = z.shape[-2:]; ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        t = nnF.pad(z, (0, pw, 0, ph), mode="replicate")
        return net(torch.cat([t, s.view(-1, 1, 1, 1).expand(t.shape[0], 1, *t.shape[-2:])], 1))[..., :H, :W]
    def unfold(kind, net, prm, y, se, z):
        """kind = "PB"：DA-LSIK の出力に引き戻す／"PH"：ボケの式で観測に合わせる。z：出発点。DRUNet と交互に 4 回"""
        if kind == "PB":
            g0 = G_dalsik(y, se)
        else:
            yp = nnF.pad(y, (PAD,) * 4, mode="reflect"); K = otfs(*yp.shape[-2:]); FY = torch.fft.fft2(yp)
            w = binw(se * prm["fac"])                             # 推定 σ の倍率も学習する（見積もりすぎの傾向を直す）
        for k in range(4):
            s = prm["s"][k]
            if kind == "PB":
                x = (g0 + prm["r"][k] * z) / (1 + prm["r"][k])
            else:
                rho = prm["lam"][k] * SN ** 2 / s ** 2
                FZ = torch.fft.fft2(nnF.pad(z, (PAD,) * 4, mode="reflect"))
                X = torch.fft.ifft2((K[None, :, None] * FY[:, None] + rho * FZ[:, None]) / (K[None, :, None] ** 2 + rho)).real
                x = (w[:, :, None] * X[..., PAD:-PAD, PAD:-PAD]).sum(1)   # σ の段ごとに解いて、推定 σ の地図で混ぜる
            z = den(net, x.clamp(0, 1), s).clamp(0, 1)
        return z
    def init_raw():
        raw = dict(ls=torch.log(torch.tensor(np.geomspace(3 / 255, SN, 4), dtype=torch.float32, device=DEVICE)),
                   lr_=torch.zeros(4, device=DEVICE), llam=torch.full((4,), np.log(0.23), device=DEVICE), lfac=torch.tensor(np.log(0.8), device=DEVICE))
        for v in raw.values(): v.requires_grad_(True)
        return raw
    prm_of = lambda raw: dict(s=raw["ls"].exp(), r=raw["lr_"].exp(), lam=raw["llam"].exp(), fac=raw["lfac"].exp())
    grad_l1 = lambda a, b: (nnF.l1_loss(a[..., 1:, :] - a[..., :-1, :], b[..., 1:, :] - b[..., :-1, :]) +
                            nnF.l1_loss(a[..., :, 1:] - a[..., :, :-1], b[..., :, 1:] - b[..., :, :-1]))
    drunet0 = spandrel.ModelLoader().load_from_file(download("https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth")).model
    t2 = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(2, 0, 1)[None].to(DEVICE)

    def train_unfold(name, kind, start, iters=1500, Pz=128, B=6, seed=0):
        """start = "prop"：DA-LSIK の出力から／"b"：ボケ画像から始める"""
        torch.manual_seed(seed); rng = np.random.default_rng(seed)
        net = copy.deepcopy(drunet0).to(DEVICE).train(); raw = init_raw()
        opt = torch.optim.Adam([dict(params=list(raw.values()), lr=2e-2), dict(params=list(net.parameters()), lr=1e-5)])
        stk = lambda L: torch.from_numpy(np.stack(L)).permute(0, 3, 1, 2).float().to(DEVICE)
        t0 = time.time()
        for it in range(iters):
            ys, ss, zs, gs = [], [], [], []
            for _ in range(B):
                d = unfold_train[rng.integers(len(unfold_train))]; H, W = d["x"].shape[:2]
                y0, x0 = rng.integers(24, H - Pz - 24), rng.integers(24, W - Pz - 24); sl = (slice(y0, y0 + Pz), slice(x0, x0 + Pz))
                ys.append(d["b"][sl]); ss.append(d["se"][sl][..., None]); zs.append(d[start][sl]); gs.append(d["x"][sl])
            o = unfold(kind, net, prm_of(raw), stk(ys), stk(ss), stk(zs)); gt_ = stk(gs)
            loss = nnF.l1_loss(o, gt_) + grad_l1(o, gt_)
            opt.zero_grad(); loss.backward(); opt.step()
            if it % 500 == 0 or it == iters - 1:
                print(f"  {name:12s} it {it:4d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
        net.eval(); prm = prm_of(raw)
        with torch.no_grad():
            for t, sc in scenes.items():
                z0 = sc["out"]["LSIK, local + learned mixing"] if start == "prop" else sc["blur"]
                o = unfold(kind, net, prm, t2(sc["blur"]), t2(sc["s_est"][..., None]), t2(z0))[0].permute(1, 2, 0).cpu().numpy()
                sc["out"][name] = o; res[t][name] = P(o, sc["gt"])
        n_params[name] = f"{sum(p.numel() for p in net.parameters())/1e6:.1f}M (DRUNet)"
        del net; torch.cuda.empty_cache()

    for name, kind, start in (("DA-LSIK-PB", "PB", "prop"), ("DA-LSIK-PH", "PH", "prop"), ("blur -> physics+CNN", "PH", "b")):
        train_unfold(name, kind, start)
    print(f"合計 {time.time()-t0:.0f}s")
    print_table(["input (defocused)", "LSIK, local + learned mixing", "DA-LSIK-PB", "DA-LSIK-PH", "blur -> physics+CNN"])
    o = s05["out"]
    cmp = ["LSIK, local + learned mixing", "DA-LSIK-PB", "DA-LSIK-PH", "blur -> physics+CNN"]
    show([np.clip(o[n][box], 0, 1) for n in cmp] + [g[box]], [f"{n}\n{P(o[n], g):.2f} dB" for n in cmp] + ["ground truth"], size=3.2)
    # 疑似輪郭が出やすい所：左上の白と赤のヘルメットの先端付近をさらに拡大（数値は 3 枚の平均）
    tip = (slice(25, 165), slice(375, 515))
    cmp = ["input (defocused)", "LSIK, local + learned mixing", "DA-LSIK-PB", "DA-LSIK-PH"]
    show([np.clip(o[n][tip], 0, 1) for n in cmp] + [g[tip]],
         [f"{n}\nmean of 3: {np.mean([res[t][n] for t in TESTS]):.2f} dB" for n in cmp] + ["ground truth"], size=3.2)

# %% [markdown]
# **結果の読み方**：数値だけ見ると「ボケ画像から物理＋CNN」が最も高いが、輪郭がくっきりした結果ほど、左上の白と赤のヘルメットの先端付近に
# 輪郭に沿った疑似輪郭（リンギング）が出やすい（面積が小さいので PSNR にはほとんど表れない）。
# **DA-LSIK-PB** だけは、輪郭がくっきりしているのに疑似輪郭がほとんどなく、見た目では最も好ましい。物理の式で逆に解く処理をくり返さないので、
# 推定した σ の誤差で強調しすぎることがないためと考えられる。

# %% [markdown]
# ## 9. 拡散モデルで描き直す（DA-LSIK-Diff：ベースは DiffBIR）
# DA-LSIK の出力を出発点に、**DiffBIR v2.1** をベースにした拡散モデルで描き直します。DiffBIR の学習済みモデルはそのまま使い（追加の学習はしない）、
# 「出発点」と「各段で観測に合わせる修正」だけを DA-LSIK 用に変えています。
#
# **ベースの DiffBIR v2.1**（X. Lin+, *DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior*, ECCV 2024。github.com/XPixelGroup/DiffBIR、Apache-2.0）は、次の 3 つのネットワークでできています：
# - **段階 1：SwinIR**（Transformer の復元ネットワーク。Real-ESRGAN 型の劣化で学習した版 realesrgan_s4_swinir_100k）：入力から劣化をおおまかに除いて、条件画像を作る
# - **段階 2：Stable Diffusion 2.1-base**（潜在拡散モデル。DiffBIR が zero terminal SNR に調整した版 sd2.1-base-zsnr-laionaes5。U-Net ＋ VAE ＋ OpenCLIP の文章エンコーダ）：画像を描く本体
# - **IRControlNet**（DiffBIR_v2.1.pt）：段階 1 の条件画像に合うように、Stable Diffusion の描き方を誘導する
# - サンプラーは EDM の DPM++ 3M SDE（50 段）、文章への従わせ方 CFG = 6
#
# DA-LSIK 用に変えた所：
# 1. **出発点**：DA-LSIK の出力を 2 倍に拡大して VAE で潜在表現に変換し、時刻 t = 800 までノイズを加える（砂嵐からではなく、ここから描き直す。50 段のうち残りの 40 段を使う）。
#    条件画像も DA-LSIK の出力で、IRControlNet の効きは 0.1 と弱くする
# 2. **観測に合わせる修正**：逆拡散の各段で、拡散モデルが推定した鮮明画像 x̂₀ を VAE で画像に戻し、**ぼかすと入力 y に近づくように** 修正してから潜在表現に戻して次の段へ
#    （argmin ‖y − Hx‖²/σn² + ρ‖x − x̂₀‖² を共役勾配法で解く。DiffPIR・ReSample と同じ考え方）
# 3. 修正は 40 段のうち最初の 35 段まで。最後の 5 段は拡散モデルに任せて仕上げさせる（最後まで修正すると細い疑似輪郭が残るため）
#
# ボケの式 H には、推定 σ の誤差を含めて学習した混ぜ方を使います（4. の逆カーネルの混ぜ方と同じ考え方で、こちらは「推定 σ がこの値のとき、実際のボケはどの σ のガウスぼけの組合せか」を非負の最小二乗法で求める）。
# シーンの説明文（プロンプト）も与えます。計算の都合で、画像を 384×384 の小片 5 枚に分けて処理してつなぎます（`DIFF_FULL = False` なら拡大範囲を含む 1 枚だけ）。
#
# 重み（約 6.7 GB）のダウンロードを含めて T4 で 約 20〜30 分、GPU メモリ 約 8 GB。不要なら `RUN_DIFF = False`。
# **描かれた細部（ヘルメットの模様や文字）は拡散モデルが作ったもの** で、正しいとは限りません。

# %%
RUN_DIFF = True
DIFF_FULL = True
DIFF_PROMPT = "motocross race photo, riders wearing white and red helmets with the number 21, blue and white racing jerseys, dirt bikes"
DIFF_SEED, T_START, STOP_AFTER, SN_D, LAM_D, N_CG = 233, 800, 35, 0.01, 2.0, 10
if RUN_DIFF:
    from scipy.optimize import nnls
    for k_ in ["drunet0", "unfold_train"]:                       # GPU・CPU のメモリを空ける
        globals().pop(k_, None)
    torch.cuda.empty_cache()
    # ---- 1. ボケの式の混ぜ方 v（推定 σ の段 j × ガウスぼけの σ_i、非負）：4. と同じ学習データで、XᵀX を足していく
    t0 = time.time(); rng = np.random.default_rng(0)
    AtA, Atb = np.zeros((n_w, n_w)), np.zeros(n_w)
    for k, x in enumerate(train_imgs):
        D = disparity(x)
        Gs = np.stack([gblur(x, s) for s in LEVELS], -1).astype(np.float32)       # H×W×3×17：σ_i のガウスぼけ
        for df in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
            b = add_noise(sv_blur(x, np.clip(A * np.abs(D - df), 0, 4)), seed=200 + k)
            Fv = (bin_weights(est_sigma(b, df))[:, :, None, :, None] * Gs[:, :, :, None, :]).reshape(*x.shape, -1)
            m = rng.random(Fv.shape[:3]) < 0.08; m[:24] = m[-24:] = m[:, :24] = m[:, -24:] = False
            Fm = Fv[m].astype(np.float64); AtA += Fm.T @ Fm; Atb += Fm.T @ b[m]
            del Fv, Fm
    lam_v = 1e-3 * np.trace(AtA) / n_w
    Lc = np.linalg.cholesky(AtA + lam_v * np.eye(n_w))            # 正規方程式の形のまま、非負の最小二乗（nnls）を解く
    V_mix = nnls(Lc.T, np.linalg.solve(Lc, Atb), maxiter=10000)[0].reshape(len(LEVELS), len(LEVELS))
    print(f"ボケの式の混ぜ方 v を学習（{time.time()-t0:.0f}s）")

    # ---- 2. DiffBIR の取得と準備（途中の時刻から描き直せるように 2 か所を書き換える）
    DIFFBIR = os.path.abspath("DiffBIR")
    if not os.path.exists(DIFFBIR):
        subprocess.run(["git", "clone", "-q", "https://github.com/XPixelGroup/DiffBIR.git", DIFFBIR], check=True)
        subprocess.run(["git", "-C", DIFFBIR, "checkout", "-q", "5c2d6c1"], check=True)
    if IN_COLAB:
        pip_install("torchsde", "ftfy", "regex", "facexlib")
    def patch(path, old, new):
        s = open(path, encoding="utf-8").read()
        if new not in s:
            assert old in s, f"{path} の書き換え箇所が見つかりません"
            open(path, "w", encoding="utf-8").write(s.replace(old, new, 1))
    patch(os.path.join(DIFFBIR, "diffbir", "pipeline.py"), '        if start_point_type == "cond":',
          '        if __import__("os").environ.get("DIFFBIR_T_START"):\n            x_T = cond["c_img"]   # 途中から描き直す：入力の潜在表現を渡す\n        elif start_point_type == "cond":')
    edm = os.path.join(DIFFBIR, "diffbir", "sampler", "edm_sampler.py")
    patch(edm, "x_size: torch.Tuple[int],", "x_size: tuple,")
    patch(edm, "        x = x_T * torch.sqrt(1.0 + self.sigmas[0] ** 2.0)\n",
          "        sigmas = self.sigmas\n"
          "        t_start = __import__('os').environ.get('DIFFBIR_T_START')\n"
          "        if t_start:   # 途中から描き直す：時刻 t_start 以下の段だけ使い、入力の潜在表現にその段のノイズを加える\n"
          "            sigmas = self.sigmas[int((self.timesteps <= int(t_start)).nonzero()[0]):]\n"
          "            x = x_T.float() + sigmas[0] * torch.randn(x_T.shape, device=x_T.device, dtype=torch.float32)\n"
          "        else:\n"
          "            x = x_T * torch.sqrt(1.0 + self.sigmas[0] ** 2.0)\n")
    patch(edm, "            sigmas=self.sigmas,\n            extra_args=None,", "            sigmas=sigmas,\n            extra_args=None,")
    os.environ["DIFFBIR_T_START"] = str(T_START)
    sys.path.insert(0, DIFFBIR)
    cwd0, argv0 = os.getcwd(), sys.argv
    os.chdir(DIFFBIR)                                             # 重みは DiffBIR/weights に保存される
    sys.argv = ["inference.py", "--task", "sr", "--upscale", "1", "--version", "v2.1", "--captioner", "none", "--precision", "fp16",
                "--device", DEVICE, "--input", ".", "--output", ".", "--steps", "50", "--sampler", "edm_dpm++_3m_sde", "--cfg_scale", "6.0"]
    try:
        import inference
        from diffbir.inference import BSRInferenceLoop
        from diffbir.sampler.edm_sampler import EDMSampler
        dargs = inference.parse_args()
        dloop = BSRInferenceLoop(dargs)
    finally:
        os.chdir(cwd0); sys.argv = argv0
    dpipe, cldm = dloop.pipeline, dloop.cldm

    # ---- 3. ボケの式 H（推定 σ → 学習した v で 17 個のガウスぼけを混ぜる）と、観測に合わせる修正（共役勾配法）
    def gk(s):
        r = max(1, int(round(4 * s))); t = torch.arange(-r, r + 1, dtype=torch.float32, device=DEVICE)
        e = torch.exp(-t ** 2 / (2 * s * s)); return e / e.sum()
    def make_H(se):
        Wv = torch.from_numpy((bin_weights(se) @ V_mix).transpose(2, 0, 1).copy()).float().to(DEVICE)   # 画素ごとの σ_i の重み
        used = [i for i in range(len(LEVELS)) if Wv[i].abs().max() > 1e-6]
        def H(img):
            out = 0
            for i in used:
                s = float(LEVELS[i])
                if s == 0:
                    bl = img
                else:
                    k = gk(s); r = len(k) // 2
                    bl = nnF.conv2d(nnF.pad(img, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).expand(3, 1, 1, -1), groups=3)
                    bl = nnF.conv2d(nnF.pad(bl, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).expand(3, 1, -1, 1), groups=3)
                out = out + Wv[i] * bl
            return out
        return H
    def Ht(H, r):                                                 # Hᵀ r（自動微分で正確に）
        x = torch.zeros_like(r, requires_grad=True)
        with torch.enable_grad():
            g_, = torch.autograd.grad((H(x) * r).sum(), x)
        return g_
    def data_step(H, y, x0, rho):                                 # (HᵀH + ρ I) x = Hᵀ y + ρ x̂₀ を共役勾配法で
        Aop = lambda v: Ht(H, H(v)) + rho * v
        x = x0.clone(); r = Ht(H, y) + rho * x0 - Aop(x); p = r.clone(); rr = (r * r).sum()
        for _ in range(N_CG):
            if rr < 1e-10:
                break
            Ap = Aop(p); al = rr / (p * Ap).sum()
            x = x + al * p; r = r - al * Ap
            rn = (r * r).sum(); p = r + rn / rr * p; rr = rn
        return x
    down = lambda im: nnF.avg_pool2d(im, 2)
    up = lambda im: nnF.interpolate(im, scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
    DS = {}                                                       # いま処理中の小片の H と y、何段目か

    _orig_den = EDMSampler.convert_to_denoiser
    def _den_with_constraint(self, model, cond, uncond, cfg_scale):
        den_ = _orig_den(self, model, cond, uncond, cfg_scale)
        def den_dc(x, sigma):
            x0 = den_(x, sigma)
            DS["n"] += 1
            if DS["n"] >= STOP_AFTER:                             # 後半は拡散モデルに任せる
                return x0
            h, w = DS["y"].shape[2:]
            with torch.autocast("cuda", dtype=torch.float16):
                img = (model.vae_decode(x0).float().clamp(-1, 1) + 1) / 2
            lo = down(img)[:, :, :h, :w]
            rho = LAM_D * SN_D ** 2 / float(sigma.max()) ** 2
            fixed = data_step(DS["H"], DS["y"], lo, rho).clamp(0, 1)
            if DS.get("dump") is not None:                        # 説明用：途中の段の「拡散モデルの推定」と「入力に戻るよう直したもの」を残す
                tonp = lambda v: v[0].permute(1, 2, 0).cpu().numpy()
                if DS["n"] == 0:
                    with torch.autocast("cuda", dtype=torch.float16):
                        DS["dump"]["start"] = tonp(down((model.vae_decode(x / (1 + float(sigma.max()) ** 2) ** 0.5).float().clamp(-1, 1) + 1) / 2)[:, :, :h, :w])
                DS["dump"][f"est{DS['n']}"] = tonp(lo); DS["dump"][f"fix{DS['n']}"] = tonp(fixed)
            d = nnF.interpolate(fixed - lo, scale_factor=2, mode="bicubic", align_corners=False)
            img2 = img.clone(); img2[:, :, :d.shape[2], :d.shape[3]] += d
            with torch.autocast("cuda", dtype=torch.float16):
                return model.vae_encode(img2.clamp(0, 1) * 2 - 1, sample=False).float()
        return den_dc
    EDMSampler.convert_to_denoiser = _den_with_constraint
    _orig_sample = EDMSampler.sample
    def _sample_from(self, *a_, **kw):                            # 出発点を DA-LSIK の出力の潜在表現に差し替える
        kw["x_T"] = DS["xT"]
        return _orig_sample(self, *a_, **kw)
    EDMSampler.sample = _sample_from

    def diff_tile(y, se, prop, dump=None):
        DS.update(H=make_H(se), y=t2(y), n=-1, dump=dump)
        with torch.no_grad():
            DS["xT"] = cldm.vae_encode(up(t2(prop)) * 2 - 1, sample=False).float()
        lq = (up(t2(prop))[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)[None]   # 条件画像も DA-LSIK の出力
        torch.manual_seed(DIFF_SEED)
        out = dpipe.run(lq, dargs.steps, 0.1, dargs.cleaner_tiled, dargs.cleaner_tile_size, dargs.cleaner_tile_stride,
                        dargs.vae_encoder_tiled, dargs.vae_encoder_tile_size, dargs.vae_decoder_tiled, dargs.vae_decoder_tile_size,
                        dargs.cldm_tiled, dargs.cldm_tile_size, dargs.cldm_tile_stride, DIFF_PROMPT + ", " + inference.DEFAULT_POS_PROMPT,
                        inference.DEFAULT_NEG_PROMPT, dargs.cfg_scale, dargs.start_point_type, dargs.sampler, dargs.noise_aug,
                        dargs.rescale_cfg, dargs.s_churn, dargs.s_tmin, dargs.s_tmax, dargs.s_noise, dargs.eta, dargs.order)[0]
        o = down(torch.from_numpy(out.astype(np.float32) / 255).permute(2, 0, 1)[None])[0].permute(1, 2, 0).numpy()
        return o[:y.shape[0], :y.shape[1]]

    # ---- 4. 画像 05 を 384×384 の小片で処理してつなぐ（拡大範囲を含む (0,320) の小片は、そのまま使う）
    y5, se5, p5 = (np.asarray(a_, np.float32) for a_ in (s05["blur"], s05["s_est"], s05["out"]["LSIK, local + learned mixing"]))
    Hh, Ww = y5.shape[:2]; TS = 384
    cut = lambda a_, r0, c0: a_[r0:r0 + TS, c0:c0 + TS]
    def ramp(n, lo_edge, hi_edge, f):                             # つなぎ目をなめらかにする重み
        w = np.ones(n, np.float32); r = np.linspace(1 / f, 1, f, dtype=np.float32)
        if not lo_edge: w[:f] = np.minimum(w[:f], r)
        if not hi_edge: w[-f:] = np.minimum(w[-f:], r[::-1])
        return w
    t0 = time.time()
    steps_dump = {}
    main = diff_tile(cut(y5, 0, 320), cut(se5, 0, 320), cut(p5, 0, 320), dump=steps_dump)
    with torch.no_grad():
        main_blur = make_H(cut(se5, 0, 320))(t2(main))[0].permute(1, 2, 0).cpu().numpy()    # 生成した画像をボケの式でぼかす
    c16 = (slice(16, -16), slice(16, -16))
    rms_ = lambda a_: np.sqrt(np.mean((a_ - cut(y5, 0, 320))[c16] ** 2))
    print(f"小片 (0,320)：{time.time()-t0:.0f}s　GPU 最大 {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
    print(f"ぼかすと入力にどれだけ近いか（RMS）：生成した画像 {rms_(main_blur):.3f}（正解 {rms_(make_H(cut(se5, 0, 320))(t2(cut(s05['gt'], 0, 320)))[0].permute(1, 2, 0).cpu().numpy()):.3f}）")
    if DIFF_FULL:
        acc, ws = np.zeros_like(y5), np.zeros((Hh, Ww), np.float32)
        for r0, c0 in ((0, 0), (Hh - TS, 0), (0, Ww - TS), (Hh - TS, Ww - TS)):
            o_ = diff_tile(cut(y5, r0, c0), cut(se5, r0, c0), cut(p5, r0, c0))
            w = np.outer(ramp(TS, r0 == 0, r0 + TS == Hh, 48), ramp(TS, c0 == 0, c0 + TS == Ww, 48))
            acc[r0:r0 + TS, c0:c0 + TS] += w[..., None] * o_; ws[r0:r0 + TS, c0:c0 + TS] += w
        others = acc / ws[..., None]
        m = np.zeros((Hh, Ww), np.float32); m[0:TS, 320:320 + TS] = np.outer(ramp(TS, True, False, 32), ramp(TS, False, False, 32))
        full = others.copy(); full[0:TS, 320:320 + TS] = main
        full = m[..., None] * full + (1 - m[..., None]) * others
        s05["out"]["DA-LSIK-Diff"] = full; res["05"]["DA-LSIK-Diff"] = P(full, s05["gt"])
        print(f"5 枚の合計 {time.time()-t0:.0f}s　画像 05 全体の PSNR {res['05']['DA-LSIK-Diff']:.2f} dB（DA-LSIK {res['05']['LSIK, local + learned mixing']:.2f} dB）")
    else:
        full = p5.copy(); full[0:TS, 320:320 + TS] = main       # 拡大範囲だけ描き直したもの（PSNR は計算しない）
        s05["out"]["DA-LSIK-Diff"] = full
    EDMSampler.convert_to_denoiser, EDMSampler.sample = _orig_den, _orig_sample
    del dloop, dpipe, cldm; torch.cuda.empty_cache()
    show([np.clip(s05["out"]["LSIK, local + learned mixing"][box], 0, 1), np.clip(full[box], 0, 1), g[box]],
         ["DA-LSIK (start point)", "DA-LSIK-Diff (base: DiffBIR)", "ground truth"], size=4.2)
    # 描く途中の様子（スライドの「DA-LSIK-Diff」の図）：ヘルメットの付近（小片 (0,320) の中の座標）
    bt = (slice(25, 165), slice(55, 195))
    k25 = 24                                                     # 25 段目（0 始まりで 24）
    show([cut(y5, 0, 320)[bt], cut(se5, 0, 320)[bt] / 4, cut(p5, 0, 320)[bt], steps_dump["start"][bt],
          steps_dump[f"est{k25}"][bt], steps_dump[f"fix{k25}"][bt], main[bt], main_blur[bt], cut(s05["gt"], 0, 320)[bt]],
         ["input y", "estimated sigma\n(bright = more blur)", "DA-LSIK output", "start point\n(noise added, t=800)",
          "diffusion estimate\n(step 25)", "fixed so that blurring\nit gives y (step 25)", "generated (step 40)", "generated, blurred\nby the blur model", "ground truth"],
         size=2.6)
    show([steps_dump[f"est{k}"][bt] for k in (0, 9, 14, 19, 29, 34)] + [main[bt]],
         [f"step {k + 1}\n(constrained)" for k in (0, 9, 14, 19, 29, 34)] + ["step 40\n(no constraint)"], size=2.6)

# %% [markdown]
# ## 10. 全手法の比較（スライド「結果画像：全手法の比較」と同じ図）
# 画像 05 の奥の選手（ボケ σ = 2〜4 画素）を拡大。数値は画像 05 全体の PSNR（3 枚の平均ではない）。
# 赤枠＝提案手法 DA-LSIK と DA-LSIK-PB（見た目が最も好ましい）、青枠＝DA-LSIK-PH。実行しなかった手法は省きます。

# %%
o = s05["out"]
PANELS = [("input (defocused)", "input (defocused)"), ("LSIK, fixed PSF", "LSIK\n(fixed PSF)"),
          ("Wiener, local (est. depth)", "Wiener, local\n(est. depth)"), ("LSIK, local (est. depth)", "LSIK, local\n(est. depth)"),
          ("LSIK, local + learned mixing", "DA-LSIK\n(proposed)"), ("LSIK, local (true depth)", "(ref.) DA-LSIK\nwith true depth"),
          ("CNN (per-pixel)", "CNN (12 layers)"), ("MultiWienerNet", "MultiWienerNet"), ("KPAC", "KPAC"), ("IFAN", "IFAN"), ("INIKNet", "INIKNet"),
          ("DA-LSIK-PB", "DA-LSIK-PB\n(pull-back + CNN)"), ("DA-LSIK-PH", "DA-LSIK-PH\n(physics + CNN)"),
          ("blur -> physics+CNN", "(ref.) blur ->\nphysics + CNN"), ("DA-LSIK-Diff", "DA-LSIK-Diff\n(base: DiffBIR)")]
PANELS = [(k, t) for k, t in PANELS if k in o]
RED, BLUE = {"LSIK, local + learned mixing", "DA-LSIK-PB"}, {"DA-LSIK-PH"}
cols = 8; rows = (len(PANELS) + 1 + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(2.5 * cols, 3.45 * rows), squeeze=False)
for ax in axes.flat:
    ax.set_xticks([]); ax.set_yticks([])
    for sp_ in ax.spines.values(): sp_.set_visible(False)
for ax, (k, t) in zip(axes.flat, PANELS + [("gt", "ground truth")]):
    im = g if k == "gt" else o[k]
    ax.imshow(np.clip(im[box], 0, 1))
    sub = "" if k == "gt" or (k == "DA-LSIK-Diff" and "DA-LSIK-Diff" not in res["05"]) else f"\n{P(im, g):.2f} dB"
    col = "#B00000" if k in RED or k.startswith("DA-LSIK") else ("#1F5FA8" if k in BLUE else "black")
    ax.set_title(("" if "\n" in t else "\n") + t + sub, fontsize=11, color="#1F5FA8" if k in BLUE else col)
    if k in RED or k in BLUE:
        for sp_ in ax.spines.values():
            sp_.set_visible(True); sp_.set_edgecolor("#B00000" if k in RED else "#1F5FA8"); sp_.set_linewidth(4 if k in RED else 3)
plt.tight_layout(); plt.show()
print("3 枚の平均（DA-LSIK-Diff は画像 05 だけ処理したので除く）")
print_table([k for k, _ in PANELS if k != "DA-LSIK-Diff"])

# %% [markdown]
# ## まとめ
# - **PSF が一定なら（論文の再現）**：同じ 15×15 の大きさでは、LSIK（最小二乗の逆カーネル）が Wiener を上回る。実際のボケがガウスでない（円盤状）と差はさらに大きい。
#   PSF を正確に与えた理想的な Wiener（大きさ無制限）とはほぼ同等で、LSIK は PSF の式を知らずにデータだけでそこへ到達する
# - **奥行きで PSF が変わると**：固定 PSF の手法は、手前を壊し、奥はボケが残るので、かえって悪化する
# - **奥行き推定 × 場所ごとの逆カーネル（DA-LSIK）**：奥行き推定の誤差（特に σ の見積もりすぎ）で性能が落ちるが、
#   **誤差の傾向まで含めて混ぜ方を最小二乗法で学習** すると、正しい奥行きを使った場合に近づく（約 8 千係数）。
#   同じ学習画像 3 枚で学習し直した先行研究（IFAN・INIKNet）とほぼ同等で、KPAC・12 層の CNN・MultiWienerNet を上回る（係数は 250〜1,300 分の 1）
# - ただし先行研究は大規模データ（DPDD など）向けの設計で、大規模データでは先行研究が上回る可能性が高い。データが少ない・計算を軽くしたい・中身を説明したい場面で、逆カーネルの方法が有利
# - **DA-LSIK の出力からさらに追い込む**：DRUNet と交互に 4 回（Deep Unfolding）。DA-LSIK に引き戻す DA-LSIK-PB は、疑似輪郭がほとんどなく見た目が最も好ましい。
#   拡散モデルで描き直す DA-LSIK-Diff は最もくっきりするが、描かれた細部は作り物で、PSNR は下がる
# - ボケが大きい所（σ = 3〜4 画素）は、拡散モデル以外のどの手法でも細部は戻らない（高い周波数の情報が、ノイズに埋もれて失われている）
# - 実際のレンズのボケは円盤状で、収差により画面の端で形も変わる → PSF の式を仮定せず、キャリブレーション撮影の画像の組から逆カーネルを学習できるのが強み
