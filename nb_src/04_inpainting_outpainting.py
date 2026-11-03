# %% [markdown]
# # 04. Inpainting（欠損・キズ・不要物の除去）と Outpainting（画角拡張）
#
# **セミナー対応章**：第3章（2）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約5分
#
# 1. **マスク**（どこを消して、どこを創るか）を作る
# 2. 古典手法（OpenCV Telea）／ **LaMa**（高速な CNN 系）／ **Stable Diffusion Inpainting**（拡散モデル）を比較
# 3. **Outpainting**：キャンバスを広げて「写っていない背景」を生成
# 4. **境界線の馴染ませ方**：そのまま貼る / フェザー（ぼかしマスク）/ ポアソン合成 / 色合わせ

# %% [setup]

# %% [markdown]
# ## 1. 古い写真の「キズ」を模したマスクを作る

# %%
import cv2

img = load_sample("astronaut")                       # 512x512
rng = np.random.default_rng(3)
mask = np.zeros((512, 512), np.uint8)
for _ in range(12):                                  # ランダムな線状のキズ
    p1 = tuple(int(v) for v in rng.integers(0, 512, 2))
    p2 = tuple(int(v) for v in rng.integers(0, 512, 2))
    cv2.line(mask, p1, p2, 255, int(rng.integers(3, 9)))
damaged = np.array(img).copy()
damaged[mask > 0] = 255                              # 白いキズ
show([img, Image.fromarray(damaged), mask], ["original", "damaged (scratches)", "mask"], size=3.5)

# %% [markdown]
# ## 2. 3つの方法で修復
# ### 2-a. 古典手法：OpenCV（Telea法：周囲の画素から滑らかに補間）

# %%
telea = cv2.inpaint(damaged, mask, 5, cv2.INPAINT_TELEA)

# %% [markdown]
# ### 2-b. LaMa（Large Mask Inpainting, 2022）
# フーリエ畳み込みで画像全体の文脈を使う CNN。拡散モデルより圧倒的に速く、**キズ消し・不要物除去の実務ではまず試す価値あり**。

# %%
import spandrel

lama = spandrel.ModelLoader().load_from_file(
    download("https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt")).to(DEVICE).eval()

@torch.no_grad()
def run_lama(image, mask_np):
    x = to_tensor(image).to(DEVICE)
    m = torch.from_numpy((mask_np > 0).astype(np.float32))[None, None].to(DEVICE)
    return tensor_to_pil(lama(x, m))

t = time.time()
lama_out = run_lama(Image.fromarray(damaged), mask)
print(f"LaMa: {time.time()-t:.2f}s")

# %% [markdown]
# ### 2-c. Stable Diffusion Inpainting（拡散モデル）
# マスク部分をノイズから生成し、マスク外の画素と矛盾しないように逆拡散します。

# %%
from diffusers import AutoPipelineForInpainting

inpaint = AutoPipelineForInpainting.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=DTYPE).to(DEVICE)
inpaint.set_progress_bar_config(disable=True)

def sd_inpaint(image, mask_np, prompt, steps=30, seed=0, **kw):
    return inpaint(prompt=prompt, image=image, mask_image=Image.fromarray(mask_np),
                   num_inference_steps=steps, generator=torch.Generator(DEVICE).manual_seed(seed),
                   height=image.size[1], width=image.size[0], **kw).images[0]

t = time.time()
sd_out = sd_inpaint(Image.fromarray(damaged), cv2.dilate(mask, np.ones((5, 5), np.uint8)),
                    "a photo of an astronaut, high quality")
print(f"SD inpainting: {time.time()-t:.2f}s")

show([Image.fromarray(damaged), Image.fromarray(telea), lama_out, sd_out],
     ["damaged", f"OpenCV Telea {psnr(Image.fromarray(telea), img):.1f}dB",
      f"LaMa {psnr(lama_out, img):.1f}dB", f"SD Inpaint {psnr(sd_out, img):.1f}dB"], size=3.5)

# %% [markdown]
# ## 3. 不要物の除去（大きなマスク）
# 宇宙飛行士の左後ろにある **星条旗** を消してみます。大きな領域では古典手法は破綻し、生成モデルの差が出ます。

# %%
obj_mask = np.zeros((512, 512), np.uint8)
cv2.rectangle(obj_mask, (0, 0), (150, 230), 255, -1)       # 左上の旗のあたり
res_lama = run_lama(img, obj_mask)
res_sd = [sd_inpaint(img, obj_mask, "plain gray studio wall, empty background", seed=s,
                     negative_prompt="flag, stars, stripes, text") for s in (0, 1)]
res_telea = Image.fromarray(cv2.inpaint(np.array(img), obj_mask, 5, cv2.INPAINT_TELEA))
show([img, res_telea, res_lama] + res_sd, ["original", "Telea", "LaMa", "SD seed=0", "SD seed=1"], size=3.2)

# %% [markdown]
# **観察ポイント**：拡散モデルは seed を変えると **毎回違う「もっともらしい」答え** を出します（正解は1つではない）。
# これは創作には長所ですが、記録・証拠用途では「生成された部分」であることの明示が必要です（第4章：真正性）。

# %% [markdown]
# ## 4. Outpainting：画角を広げる
# キャンバスを広げ、元画像がない部分をマスクとして Inpainting するだけで Outpainting になります。

# %%
def outpaint_canvas(image, pad_left=0, pad_right=0, pad_top=0, pad_bottom=0):
    w, h = image.size
    W, H = w + pad_left + pad_right, h + pad_top + pad_bottom
    W8, H8 = (W + 7) // 8 * 8, (H + 7) // 8 * 8
    canvas = Image.new("RGB", (W8, H8), (127, 127, 127))
    canvas.paste(image, (pad_left, pad_top))
    m = np.full((H8, W8), 255, np.uint8)
    m[pad_top:pad_top + h, pad_left:pad_left + w] = 0
    return canvas, m

src = load_sample("coffee").resize((384, 256))            # コーヒーカップの写真
canvas, omask = outpaint_canvas(src, pad_left=128, pad_right=128, pad_top=0, pad_bottom=64)
out_sd = sd_inpaint(canvas, omask, "a cup of coffee on a wooden table, cafe, photo", steps=40)
out_lama = run_lama(canvas, omask)
show([src, canvas, out_lama, out_sd], ["source", "canvas + mask(gray)", "LaMa outpaint", "SD outpaint"], size=4)

# %% [markdown]
# ## 5. 境界線の馴染ませ方
# 画像の一部だけが劣化しているとき、その部分だけを生成AIで修復して元画像に貼り戻すことがあります。
# このとき **境界に段差（明るさ・色の違い）** が見えることがあります。
# 原因：SD Inpainting は画像を VAE で圧縮→復元するので色が少しずれる、生成部分の照明・色味が周囲と微妙に違う、など。
#
# 例：飛行機のプロペラの羽根の部分に **強いノイズが乗った** 画像（Kodak 標準テスト画像 kodim20 に人工的にノイズを加えたもの）。
# 1. 羽根の部分だけをマスクにし、SD Inpainting を **strength 0.7**（ノイズ画像を途中までノイズ化して描き直す）で実行
#    → 羽根の形・色は残したまま、ノイズのない「きれいな羽根」が生成される
# 2. 生成した羽根を元画像に貼り戻す。背景が **空（なめらかなグラデーション）** なので、境界の段差が目立ちやすい

# %%
plane = Image.open(download("https://r0k.us/graphics/kodak/kodak/kodim20.png")).convert("RGB")   # 768x512（正解）
bmask = np.zeros((512, 768), np.uint8)
cv2.line(bmask, (196, 68), (160, 185), 255, 40)           # 上側のプロペラの羽根
src = np.array(plane).astype(np.float32)
noisy_np = src.copy()                                     # 羽根の部分だけに強いノイズ（σ=40/255）
noisy_np[bmask > 0] = np.clip(src + np.random.default_rng(0).normal(0, 40, src.shape), 0, 255)[bmask > 0]
noisy_plane = Image.fromarray(noisy_np.astype(np.uint8))
gen_img = sd_inpaint(noisy_plane, bmask, "a clean airplane propeller blade against a clear hazy sky, photo, sharp",
                     seed=0, negative_prompt="noise, grainy, blurry", strength=0.7)
src = noisy_np                                            # 貼り戻し先＝ノイズ画像（マスク外はきれい）
gen = np.array(gen_img).astype(np.float32)
print(f"マスク外の平均絶対誤差（VAE由来の変化）: {np.abs(gen - src)[bmask == 0].mean():.2f} / 255")
ring = (cv2.dilate(bmask, np.ones((31, 31), np.uint8)) > 0) & (bmask == 0)
print("周辺の空の平均色 (RGB):", src[ring].mean(0).round(1), " 生成部分の平均色:", gen[bmask > 0].mean(0).round(1))

# (a) そのまま貼る：マスク内だけ生成結果、外は元画像
hard = np.where(bmask[..., None] > 0, gen, src)
# (b) フェザー：マスクをぼかしてアルファブレンド。修復では、ぼかしてもマスク内（劣化部分）が元画像と混ざらないよう、先にマスクを広げる
alpha = cv2.GaussianBlur(cv2.dilate(bmask, np.ones((31, 31), np.uint8)).astype(np.float32) / 255, (0, 0), 6)[..., None]
feather = alpha * gen + (1 - alpha) * src
# (c) ポアソン合成：生成部分の「模様（勾配）」だけを使い、境界の明るさ・色は周囲に合わせる
pmask = cv2.dilate(bmask, np.ones((15, 15), np.uint8))
ys, xs = np.where(pmask > 0)
x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
poisson = cv2.seamlessClone(gen.astype(np.uint8)[y0:y1, x0:x1], src.astype(np.uint8), pmask[y0:y1, x0:x1].copy(),
                            ((x0 + x1) // 2, (y0 + y1) // 2), cv2.NORMAL_CLONE).astype(np.float32)
poisson = np.where(pmask[..., None] > 0, poisson, src)          # seamlessClone は外接矩形内のマスク外も変える（渡したマスクも書き換える）ので、マスク内だけ使う
# (d) 色合わせ：生成部分の平均・分散を周辺リングに合わせる
mu_s, sd_s = src[ring].mean(0), src[ring].std(0)
mu_g, sd_g = gen[ring].mean(0), gen[ring].std(0)
matched = np.where(bmask[..., None] > 0, (gen - mu_g) / (sd_g + 1e-6) * sd_s + mu_s, src)

# 入力（赤線＝マスクの境界）と、4つの貼り方の拡大
outline = noisy_np.astype(np.uint8).copy()
cnts, _ = cv2.findContours(bmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(outline, cnts, -1, (255, 0, 0), 2)
crop = (slice(20, 260), slice(80, 300))
results = {"(a) hard paste": hard, "(b) feathered alpha": feather, "(c) Poisson (seamlessClone)": poisson,
           "(d) color matched": matched}
show([outline[crop]] + [np.clip(v, 0, 255).astype(np.uint8)[crop] for v in results.values()],
     ["noisy input (red = mask)"] + list(results), size=3.5)

# 空の明るさの差はわずか（数/255）なので、コントラストを強調して境界を見やすくする
# 明るさ 230〜255 を 0〜255 に引き伸ばす（羽根は黒くつぶれる）
def enhance(v):
    g = np.clip(v, 0, 255).mean(-1)
    return np.clip((g - 230) * 255 / 25, 0, 255).astype(np.uint8)
show([enhance(v)[crop] for v in results.values()], [f"{n}\ncontrast enhanced" for n in results], size=3.5)

# %% [markdown]
# **観察ポイント**
# - 羽根のノイズは消え、形・色は保たれた（strength を上げすぎると羽根の形が変わる。羽根の中央のマークは消えている点に注意）
# - (a) そのまま貼ると、生成部分の空の明るさ・色が周囲とわずかに違い、**羽根の周りにマスクの形の枠** が見える（下段の強調画像で明瞭）
# - (b) フェザーは枠をぼかすだけで、明るさの違い自体は残る（ぼんやりした影になる）。修復ではマスクを広げてからぼかし、劣化した画素を混ぜない
# - (c) ポアソン合成は境界の明るさを周囲に合わせるので、ほぼ見えなくなる（なめらかな背景で特に有効）。OpenCV の seamlessClone はマスク外も変えるので、マスク内だけを使う
# - (d) 色合わせは平均の色を揃える。模様のある背景ではポアソン合成より自然なこともある
# → 実務では「マスク外は元画像を残す」＋「(b)〜(d) のいずれか」を必ず入れる

# %% [markdown]
# ## まとめ
# - **マスク設計** が品質の半分を決める（キズより少し太く、物体は影まで含める）
# - 小さな欠損 → 古典法・LaMa で十分（高速・ハルシネーション小）／大きな欠損・画角拡張 → 拡散モデル
# - 生成結果は **マスク外を元画像で上書き** し、フェザー・ポアソン合成・色合わせで境界をなじませる
# - 配布の ComfyUI ワークフロー `inpaint_sd15.json・outpaint_sd15.json` でも同じことができます
