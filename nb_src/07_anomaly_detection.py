# %% [markdown]
# # 07. 外観検査・異常検知への画像復元の応用
#
# **セミナー対応章**：第3章（5）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約6分
#
# 1. 金属表面を模した **人工データセット**（良品・欠陥品）を生成（実データ不要・ライセンス問題なし）
# 2. **オートエンコーダ（AE）による再構成** で「良品に戻らない部分＝欠陥」を検出（良品データだけで学習）
# 3. **生成AIによる修復（Inpainting）で良品に戻す** → 差分で欠陥を検出（学習不要）
# 4. **疑似欠陥データの自動生成**：拡散モデル Inpainting で「キズ・汚れ」を合成し、マスク付き学習データを作るパイプライン
#
# > 自社データで試す場合は、`normal_imgs`（良品）と `test_imgs`（検査画像）を差し替えてください。公開データセットでは MVTec AD（非商用ライセンス）が有名です。

# %% [setup]

# %% [markdown]
# ## 1. 人工データセット：ヘアライン加工の金属表面

# %%
import cv2

S = 256

def metal_surface(seed):
    r = np.random.default_rng(seed)
    n = r.normal(0, 1, (S, S)).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigmaX=r.uniform(8, 14), sigmaY=0.6)          # 横方向のヘアライン
    n = (n - n.mean()) / (n.std() + 1e-6) * 0.06
    yy, xx = np.mgrid[0:S, 0:S] / S
    light = 0.55 + 0.15 * np.sin(np.pi * (xx * r.uniform(0.5, 1.5) + r.uniform(0, 1)))   # 照明ムラ
    g = np.clip(light + n, 0, 1)
    tint = np.array([1.0, 0.98, 0.95]) * r.uniform(0.95, 1.05)
    return np.clip(g[..., None] * tint, 0, 1).astype(np.float32)

def add_defect(img, seed, kind):
    r = np.random.default_rng(seed)
    out, m = img.copy(), np.zeros((S, S), np.uint8)
    if kind == "scratch":
        p1, p2 = r.integers(30, S - 30, 2), r.integers(30, S - 30, 2)
        cv2.line(m, tuple(int(v) for v in p1), tuple(int(v) for v in p2), 255, int(r.integers(2, 4)))
        out[m > 0] = np.clip(out[m > 0] + 0.3, 0, 1)
    elif kind == "stain":
        c = tuple(int(v) for v in r.integers(40, S - 40, 2))
        cv2.ellipse(m, c, (int(r.integers(8, 20)), int(r.integers(6, 14))), int(r.integers(0, 180)), 0, 360, 255, -1)
        a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 3)[..., None]
        out = out * (1 - 0.6 * a) + 0.6 * a * np.array([0.35, 0.25, 0.15])
    elif kind == "dent":
        c = tuple(int(v) for v in r.integers(40, S - 40, 2))
        cv2.circle(m, c, int(r.integers(6, 12)), 255, -1)
        a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 4)[..., None]
        out = np.clip(out - 0.3 * a, 0, 1)
    return out.astype(np.float32), m

normal_imgs = [metal_surface(i) for i in range(200)]                 # 学習用：良品のみ
kinds = ["scratch", "stain", "dent"]
test_imgs, test_masks, test_labels = [], [], []
for i in range(30):                                                   # テスト：良品15 + 欠陥15
    base = metal_surface(1000 + i)
    if i % 2 == 0:
        test_imgs.append(base); test_masks.append(np.zeros((S, S), np.uint8)); test_labels.append(0)
    else:
        d, m = add_defect(base, i, kinds[i % 3]); test_imgs.append(d); test_masks.append(m); test_labels.append(1)

show([normal_imgs[0], normal_imgs[1]] + [test_imgs[i] for i in (1, 3, 5)],
     ["normal", "normal", "defect: " + kinds[1 % 3], "defect: " + kinds[3 % 3], "defect: " + kinds[5 % 3]], size=3)

# %% [markdown]
# ## 2. オートエンコーダ（AE）で「良品への再構成」を学習
# 良品だけで学習した AE は、欠陥を **うまく再構成できない（良品っぽく直してしまう）**。入力との差が大きい場所が欠陥です。

# %%
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, c=32):
        super().__init__()
        def down(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.2))
        def up(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.ReLU())
        self.enc = nn.Sequential(down(3, c), down(c, 2 * c), down(2 * c, 4 * c), down(4 * c, 4 * c), nn.Conv2d(4 * c, 16, 1))
        self.dec = nn.Sequential(nn.Conv2d(16, 4 * c, 1), up(4 * c, 4 * c), up(4 * c, 2 * c), up(2 * c, c),
                                 nn.ConvTranspose2d(c, 3, 4, 2, 1), nn.Sigmoid())
    def forward(self, x):
        return self.dec(self.enc(x))     # 256 → 16x16x16 のボトルネック → 256

train = torch.stack([to_tensor(im)[0] for im in normal_imgs]).to(DEVICE)
ae = AE().to(DEVICE)
opt = torch.optim.Adam(ae.parameters(), 2e-3)
t = time.time()
for step in range(1500):
    batch = train[torch.randint(0, len(train), (16,))]
    loss = nn.functional.mse_loss(ae(batch), batch)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 500 == 0:
        print(f"step {step}: loss {loss.item():.5f}")
print(f"学習時間 {time.time()-t:.0f}s")
ae.eval()

# %%
def anomaly_map(x, rec):
    d = np.abs(x - rec).mean(-1)
    return cv2.GaussianBlur(d, (0, 0), 3)

# 良品の検証用画像（学習には使わない）で「良品でも出る誤差」の平均・ばらつきを求め、正規化する
val_imgs = [metal_surface(2000 + i) for i in range(20)]

def fit_normalizer(maps):
    st = np.stack(maps)
    return st.mean(0), st.std(0) + 1e-3

def normalize(m, stats):
    mu, sd = stats
    return np.maximum((m - mu) / sd, 0)           # 良品より何σ大きいか

@torch.no_grad()
def ae_reconstruct(im):
    return ae(to_tensor(im).to(DEVICE))[0].permute(1, 2, 0).cpu().numpy()

def auroc(scores, labels):
    """ROC曲線下面積（1.0 = 完全に分離、0.5 = ランダム）"""
    s, l = np.asarray(scores, float), np.asarray(labels, int)
    pos, neg = s[l == 1], s[l == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))

def pixel_auroc(maps, masks, n=4000, seed=0):
    """画素単位の AUROC（欠陥の位置をどれだけ正しく示せたか）。計算量削減のため画素をサンプリング"""
    r = np.random.default_rng(seed)
    s_, l_ = np.concatenate([m.ravel() for m in maps]), np.concatenate([(k > 0).ravel() for k in masks])
    pos = r.choice(np.where(l_)[0], n); neg = r.choice(np.where(~l_)[0], n)
    return auroc(np.r_[s_[pos], s_[neg]], np.r_[np.ones(n), np.zeros(n)])

ae_stats = fit_normalizer([anomaly_map(v, ae_reconstruct(v)) for v in val_imgs])
ae_recs = [ae_reconstruct(im) for im in test_imgs]
ae_maps = [normalize(anomaly_map(x, r), ae_stats) for x, r in zip(test_imgs, ae_recs)]
print("AE 画像単位 AUROC:", round(auroc([m.max() for m in ae_maps], test_labels), 3),
      " / 画素単位 AUROC:", round(pixel_auroc(ae_maps, test_masks), 3))
idx = [1, 3, 5]
show([test_imgs[i] for i in idx] + [ae_recs[i] for i in idx] + [ae_maps[i] for i in idx],
     [f"input ({kinds[i%3]})" for i in idx] + ["AE reconstruction"] * 3 + ["anomaly map"] * 3, size=3, rows=3)

# %% [markdown]
# **観察ポイント**：AE の再構成は全体がぼやけるため、ヘアラインのテクスチャ自体も差分に出てしまい、異常度マップの背景がざらつきます（誤検出の原因）。
# 良品の検証画像で求めた「良品でも出る誤差」で正規化することで、ある程度抑えられます。

# %% [markdown]
# ## 3. 生成AIの修復（Inpainting）で良品に戻して比較する（学習不要）
# 画像を格子状のマスクで2回に分けて隠し、学習済み Inpainting モデル（LaMa）に「周囲から推測して埋めさせる」と、
# 欠陥は周囲から推測できないので **良品の表面として再生成** されます。その差分が欠陥です（再構成型異常検知の生成AI版）。

# %%
import spandrel

lama = spandrel.ModelLoader().load_from_file(
    download("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt")).to(DEVICE).eval()

@torch.no_grad()
def inpaint_reconstruct(im, cell=48):
    x = to_tensor(im).to(DEVICE)
    yy, xx = np.mgrid[0:S, 0:S] // cell
    out = torch.zeros_like(x)
    for a in (0, 1):                                        # 格子状のマスクを4回に分けて復元
        for b in (0, 1):
            m = torch.from_numpy(((yy % 2 == a) & (xx % 2 == b)).astype(np.float32))[None, None].to(DEVICE)
            out += lama(x, m) * m
    return out[0].permute(1, 2, 0).cpu().numpy()

t = time.time()
gen_stats = fit_normalizer([anomaly_map(v, inpaint_reconstruct(v)) for v in val_imgs])
gen_recs = [inpaint_reconstruct(im) for im in test_imgs]
print(f"{len(val_imgs) + len(test_imgs)}枚の処理時間 {time.time()-t:.1f}s")
gen_maps = [normalize(anomaly_map(x, r), gen_stats) for x, r in zip(test_imgs, gen_recs)]
print("Inpainting再構成 画像単位 AUROC:", round(auroc([m.max() for m in gen_maps], test_labels), 3),
      " / 画素単位 AUROC:", round(pixel_auroc(gen_maps, test_masks), 3))
show([test_imgs[i] for i in idx] + [gen_recs[i] for i in idx] + [gen_maps[i] for i in idx],
     [f"input ({kinds[i%3]})" for i in idx] + ["restored to 'normal'"] * 3 + ["anomaly map"] * 3,
     size=3, rows=3)

# %% [markdown]
# **観察ポイント**：Inpainting 再構成は良品のテクスチャをシャープに再現できるため、背景の誤差が小さく、**画像単位の判定（良品か不良品か）で AE より優れます**。
# 一方、**長いキズ** は隣のセルにもキズが写っているため「周囲から推測」されて一部が再生成されてしまい、画素単位の精度は AE に劣ることがあります。
# このように手法ごとに得意・不得意があるので、**自社データで画像単位・画素単位の両方を評価** して選ぶことが大切です（マスクのセルサイズも調整してみてください）。

# %% [markdown]
# > **最新研究の方向**：拡散モデルで「少しノイズを足してから良品として逆拡散する」ことで、欠陥だけを消した再構成を得る手法
# > （DiffusionAD, DDAD, TransFusion など）が提案されています。原理は本ノートブックの 2・3 と同じ「良品への復元＋差分」です。

# %% [markdown]
# ## 4. 「良品データしかない」を打破する：疑似欠陥の自動生成
# 欠陥を **マスクで指定した場所に生成** すれば、画素単位の正解ラベル付き学習データが自動で手に入ります。
# ここでは「キズ」「サビ」「油汚れ」を良品画像に描き込みます。
#
# **コツ**：Stable Diffusion Inpainting に「キズを描いて」と頼むだけだと、周囲の金属の模様を自然に延長してしまい **欠陥が描かれない** ことが多いです。
# そこで、マスク内に欠陥の **おおまかな下絵**（明るい線・茶色の斑点・黒い染み）を描き、それを途中までノイズ化して（strength 0.6）拡散モデルに **リアルに描き直させます**。
#
# 流れ：(1) 良品画像 → (2) 欠陥の位置・形のマスク（＝正解ラベル） → (3) マスク内に下絵 → (4) 拡散モデルで描き直し

# %%
from diffusers import AutoPipelineForInpainting

inpaint = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=DTYPE).to(DEVICE)
inpaint.set_progress_bar_config(disable=True)

def defect_mask(kind, seed):
    """欠陥の位置・形状（＝画素単位のラベル）をランダムに作る"""
    r = np.random.default_rng(seed)
    m = np.zeros((512, 512), np.uint8)
    if kind == "scratch":
        p1 = (int(r.integers(120, 200)), int(r.integers(140, 220)))
        p2 = (int(r.integers(320, 400)), int(r.integers(280, 380)))
        cv2.line(m, p1, p2, 255, 16)
    else:
        c = tuple(int(v) for v in r.integers(200, 310, 2))
        cv2.ellipse(m, c, (int(r.integers(55, 80)), int(r.integers(35, 55))), int(r.integers(0, 180)), 0, 360, 255, -1)
    return m

def sketch(base, m, kind, seed):
    """マスク内に欠陥の下絵を描く（明るい線・茶色の斑点・黒い染み）"""
    r = np.random.default_rng(seed)
    b = to_np(base)
    a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 2)[..., None]
    tex = r.normal(0, 0.08, b.shape[:2])[..., None]
    if kind == "scratch":
        col = np.clip(b + 0.35, 0, 1)
    elif kind == "rust":
        col = np.clip(np.array([0.55, 0.28, 0.10]) + tex, 0, 1)
    else:
        col = np.clip(np.array([0.08, 0.07, 0.06]) + tex * 0.5, 0, 1)
    return to_pil(a * col + (1 - a) * b)

prompts = {"scratch": "a long deep bright scratch groove on brushed steel plate, close-up photo",
           "rust": "orange brown rust corrosion spot on brushed steel plate, close-up photo",
           "dirt": "black oil stain on brushed steel plate, close-up photo"}
pseudo = []
for j, (name, p) in enumerate(prompts.items()):
    base = to_pil(metal_surface(5000 + j)).resize((512, 512))
    m = defect_mask(name, j)
    init = sketch(base, m, name, j)
    out = inpaint(prompt=p, negative_prompt="text, watermark, cartoon", image=init,
                  mask_image=Image.fromarray(cv2.dilate(m, np.ones((15, 15), np.uint8))),
                  num_inference_steps=30, strength=0.6, guidance_scale=7.5,
                  generator=torch.Generator(DEVICE).manual_seed(j)).images[0]
    a = cv2.GaussianBlur(cv2.dilate(m, np.ones((15, 15), np.uint8)).astype(np.float32) / 255, (0, 0), 3)[..., None]
    out = to_pil(a * to_np(out) + (1 - a) * to_np(base))         # 境界をなじませて合成（マスク外は良品のまま）
    pseudo.append((name, base, init, m, out))
# 列の順番はスライドの流れ ①良品 → ②マスク → ③下絵 → ④拡散で描き直し に合わせる
show([x for _, b, i, m, o in pseudo for x in (b, m, i, o)],
     [t for n, *_ in pseudo for t in ("(1) normal", "(2) mask = label", "(3) rough sketch", f"(4) redrawn: {n}")],
     size=3, rows=3)

# %% [markdown]
# ### 疑似欠陥データ生成パイプライン（実務での流れ）
# 1. 良品画像を用意（数十〜数百枚）
# 2. 欠陥の **位置・形状マスク** をランダム生成（実欠陥の統計に合わせる）
# 3. 下絵＋Inpainting で欠陥を生成（プロンプト、IP-Adapter で実欠陥の見た目を参照、LoRA で自社欠陥を学習）
# 4. 境界ブレンド → **マスク＝画素単位のラベル** として保存
# 5. セグメンテーション/検出モデルを学習 → **実欠陥で必ず評価**（疑似データだけで評価しない）

# %% [markdown]
# ## まとめ
# - 「良品へ復元して差分を取る」＝ 画像復元技術はそのまま異常検知になる
# - AE はぼやけて背景の誤差が大きい → 生成AI（Inpainting / 拡散モデル）でシャープに良品化すると画像単位の判定が向上（ただし長い欠陥の取りこぼしなど弱点もある）
# - 欠陥データ不足は、生成AIによる **ラベル付き疑似欠陥の合成** で補える（ただし評価は実データで）
