# %% [markdown]
# # 02. 画像生成AIのコア技術：GAN・拡散モデル・CLIP を手で動かす
#
# **セミナー対応章**：第2章　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約5分
#
# 1. **2次元のおもちゃデータ**で GAN と拡散モデルを学習し、GAN の **モード崩壊（Mode Collapse）** と拡散モデルの安定性を比較
# 2. 画像に「砂嵐」を足していく **拡散（前向き過程）** を可視化
# 3. 学習済み拡散モデル（DDPM, CelebA-HQ 顔）で **砂嵐から顔が現れる逆過程** を観察し、劣化画像に途中までノイズを加えてから逆拡散する **復元（SDEdit）** を試す
# 4. **CLIP** で画像とテキストの近さを測る（テキストで生成・復元を制御する仕組みの基礎）

# %% [setup]

# %% [markdown]
# ## 1. おもちゃデータ：8つの山（ガウス分布）
# 本物のデータ分布は「円周上の8つの山」。生成器はこの8つの山を **すべて** 再現できるでしょうか？

# %%
import torch.nn as nn

torch.manual_seed(0)

def real_data(n):
    centers = torch.tensor([[np.cos(a), np.sin(a)] for a in np.linspace(0, 2 * np.pi, 9)[:-1]], dtype=torch.float32) * 2.0
    idx = torch.randint(0, 8, (n,))
    return centers[idx] + 0.05 * torch.randn(n, 2)

def mlp(i, o, h=128):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))

# %% [markdown]
# ### 1-a. GAN（敵対的生成ネットワーク）
# 生成器 G は「偽物」を作り、識別器 D は本物/偽物を見分ける。この **騙し合い** で G が上達します。
# ただし G は「D を騙せる少数の山」だけを作るように逃げることがあり、これが **モード崩壊** です。

# %%
def train_gan(steps=3000, snapshots=(500, 1500, 3000)):
    G, D = mlp(2, 2).to(DEVICE), mlp(2, 1).to(DEVICE)
    oG = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.999))
    oD = torch.optim.Adam(D.parameters(), 1e-3, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    snaps = {}
    for s in range(1, steps + 1):
        x = real_data(256).to(DEVICE)
        z = torch.randn(256, 2, device=DEVICE)
        fake = G(z)
        lossD = bce(D(x), torch.ones(256, 1, device=DEVICE)) + bce(D(fake.detach()), torch.zeros(256, 1, device=DEVICE))
        oD.zero_grad(); lossD.backward(); oD.step()
        lossG = bce(D(fake), torch.ones(256, 1, device=DEVICE))   # 「本物」と判定させたい
        oG.zero_grad(); lossG.backward(); oG.step()
        if s in snapshots:
            with torch.no_grad():
                snaps[s] = G(torch.randn(2000, 2, device=DEVICE)).cpu()
    return snaps

gan_snaps = train_gan()

# %% [markdown]
# ### 1-b. 拡散モデル（DDPM）
# データに少しずつノイズを足す過程を考え、ネットワークには **「加えられたノイズを当てる」** だけを学習させます。
# 損失は単純な二乗誤差なので学習が安定し、すべての山をカバーしやすくなります。

# %%
T = 100
betas = torch.linspace(1e-4, 0.05, T)
alphas = 1 - betas
abar = torch.cumprod(alphas, 0)

class EpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = mlp(3, 2, 256)
    def forward(self, x, t):
        return self.net(torch.cat([x, (t.float() / T)[:, None]], 1))

def train_diffusion(steps=3000):
    net = EpsNet().to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    ab = abar.to(DEVICE)
    for s in range(steps):
        x0 = real_data(512).to(DEVICE)
        t = torch.randint(0, T, (512,), device=DEVICE)
        eps = torch.randn_like(x0)
        xt = ab[t].sqrt()[:, None] * x0 + (1 - ab[t]).sqrt()[:, None] * eps   # 前向き過程（ノイズを足す）
        loss = ((net(xt, t) - eps) ** 2).mean()                                # ノイズを当てるだけ
        opt.zero_grad(); loss.backward(); opt.step()
    return net

@torch.no_grad()
def sample_diffusion(net, n=2000):
    x = torch.randn(n, 2, device=DEVICE)                       # 砂嵐からスタート
    traj = {T: x.cpu()}
    for t in reversed(range(T)):
        tt = torch.full((n,), t, device=DEVICE)
        eps = net(x, tt)
        x = (x - betas[t] / (1 - abar[t]).sqrt() * eps) / alphas[t].sqrt()   # 少しずつノイズを取り除く
        if t > 0:
            x = x + betas[t].sqrt() * torch.randn_like(x)
        if t in (75, 50, 25, 0):
            traj[t] = x.cpu()
    return traj

diff_net = train_diffusion()
traj = sample_diffusion(diff_net)

# %%
real = real_data(2000)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
panels = [("Real data", real)] + [(f"GAN step {s}", v) for s, v in gan_snaps.items()] + \
         [(f"Diffusion t={t}", traj[t]) for t in (T, 50, 25, 0)]
for ax, (title, pts) in zip(axes.flat, panels):
    ax.scatter(real[:, 0], real[:, 1], s=2, c="lightgray")
    ax.scatter(pts[:, 0], pts[:, 1], s=2, c="crimson")
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal"); ax.set_title(title)
plt.tight_layout(); plt.show()

# %% [markdown]
# **観察ポイント**：GAN は学習が進んでも一部の山しか作らない（あるいは山の間を揺れ動く）ことが多い一方、拡散モデルは砂嵐（t=100）から徐々に 8 つの山すべてに収束します。
# （乱数により GAN がうまく行く回もあります。`torch.manual_seed` を変えて何度か試してください）

# %% [markdown]
# ## 2. 画像に対する前向き過程（砂嵐になっていく様子）
# $$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\varepsilon$$

# %%
from diffusers import DDPMScheduler

sched = DDPMScheduler(num_train_timesteps=1000)
x0 = to_tensor(load_sample("astronaut", 256)) * 2 - 1
noise = torch.randn_like(x0)
ts = [0, 50, 200, 400, 700, 999]
noisy = [sched.add_noise(x0, noise, torch.tensor([t])) for t in ts]
show([tensor_to_pil((n + 1) / 2) for n in noisy], [f"t={t}" for t in ts], size=2.8)

# %% [markdown]
# ## 3. 学習済み拡散モデルで「砂嵐から顔を生成」する逆過程
# Google が公開した DDPM（CelebA-HQ 顔画像 256×256 で学習）を使います（約450MBダウンロード）。
# 途中経過として、各ステップでモデルが予測した「完成予想図（x0予測）」も表示します。

# %%
from diffusers import DDPMPipeline, DDIMScheduler

pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256").to(DEVICE)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)   # 1000 → 50 ステップに高速化
pipe.scheduler.set_timesteps(50)

g = torch.Generator(DEVICE).manual_seed(1)
x = torch.randn(1, 3, 256, 256, generator=g, device=DEVICE)
xs, x0s = [], []
for i, t in enumerate(pipe.scheduler.timesteps):
    with torch.no_grad():
        eps = pipe.unet(x, t).sample
    out = pipe.scheduler.step(eps, t, x)
    x = out.prev_sample
    if i in (0, 10, 20, 30, 40, 49):
        xs.append(tensor_to_pil((x + 1) / 2))
        x0s.append(tensor_to_pil((out.pred_original_sample + 1) / 2))
steps = [0, 10, 20, 30, 40, 49]
show(xs, [f"x_t  step {s}" for s in steps], size=2.8)
show(x0s, [f"predicted x0  step {s}" for s in steps], size=2.8)
gpu_mem()

# %% [markdown]
# **ポイント**：初期の x0 予測は「平均的な顔」でぼんやりしていますが、ステップが進むと細部（髪・肌の質感）が決まっていきます。
# 画像復元では、この逆過程を **劣化画像で条件付け（ガイド）** することで「劣化画像と矛盾しない、もっともらしい画像」を生成します。

# %% [markdown]
# ### 3-b. 復元に使う：劣化画像に途中までノイズを加えてから逆拡散する（SDEdit）
# 砂嵐（t = 999）まで壊さず、劣化画像に **途中（t ≈ 300）までノイズを加えてから** 逆拡散します。
# 元の顔の形や色は残り、ノイズに埋もれた細部だけが描き直されます（SDEdit, Meng+ ICLR 2022）。
# 画像は CelebA-HQ の検証用の 1 枚（Hugging Face の mattymchen/celeba-hq）。縦横 1/4 に縮小して戻し、ノイズ（σ = 0.04）を加えて劣化させます。

# %%
import json, cv2
face_path = os.path.join(WEIGHTS_DIR, "celebahq_val_5.jpg")
if not os.path.exists(face_path):
    rows = json.load(urllib.request.urlopen("https://datasets-server.huggingface.co/rows?dataset=mattymchen/celeba-hq&config=default&split=validation&offset=0&length=10"))["rows"]
    urllib.request.urlretrieve(rows[5]["row"]["image"]["src"], face_path)
gt = to_np(Image.open(face_path).resize((256, 256), Image.LANCZOS))
deg = cv2.resize(cv2.resize(gt, (64, 64), interpolation=cv2.INTER_AREA), (256, 256), interpolation=cv2.INTER_CUBIC)
deg = np.clip(deg + np.random.default_rng(0).normal(0, 0.04, deg.shape), 0, 1).astype(np.float32)

sch = DDIMScheduler.from_config(pipe.scheduler.config); sch.set_timesteps(100)
abar = sch.alphas_cumprod.to(DEVICE)
T0 = 300                                                        # ここまでノイズを加える（大きいほど自由に描き直す）
g = torch.Generator(DEVICE).manual_seed(2)
x = abar[T0].sqrt() * (to_tensor(deg).to(DEVICE) * 2 - 1) + (1 - abar[T0]).sqrt() * torch.randn((1, 3, 256, 256), generator=g, device=DEVICE)
x_t0 = tensor_to_pil((x + 1) / 2)
with torch.no_grad():
    for t in sch.timesteps:
        if t <= T0:                                             # t0 より後の段だけ逆拡散する
            x = sch.step(pipe.unet(x, t).sample, t, x).prev_sample
res = tensor_to_pil((x + 1) / 2)
show([deg, x_t0, res, gt], ["degraded (details lost)", f"noise added up to t = {T0}", "after reverse diffusion\n(same face, details redrawn)", "(ref.) ground truth"], size=3.4)

# %% [markdown]
# ## 4. CLIP：画像とテキストを同じ空間に埋め込む
# Stable Diffusion や SUPIR がテキストで生成・復元を制御できるのは、CLIP のような **画像–言語の共通埋め込み** があるからです。

# %%
from transformers import CLIPModel, CLIPProcessor

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

img = load_sample("astronaut")
blurred = img.resize((64, 64), Image.BICUBIC).resize((512, 512), Image.BICUBIC)
def clip_probs(texts):
    inputs = proc(text=texts, images=[img, blurred], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        return clip(**inputs).logits_per_image.softmax(-1).cpu().numpy()

show([img, blurred], ["original", "blurred"], size=3)
for texts in (["a photo of an astronaut", "a photo of a cat", "a photo of a car"],          # 内容（何が写っているか）
              ["a sharp, high quality photo", "a blurry, low quality photo"]):              # 画質
    for name, p in zip(["original", "blurred"], clip_probs(texts)):
        print(f"{name:9s}", {t: f"{v:.2f}" for t, v in zip(texts, p)})
    print()

# %% [markdown]
# **観察ポイント**：ぼかした画像は "blurry, low quality" との類似度が上がります。
# この性質を使うと、①テキストで復元の方向を指示する（SUPIR）、②画質をテキストとの類似度で評価する（CLIP-IQA、ノートブック08）ことができます。
