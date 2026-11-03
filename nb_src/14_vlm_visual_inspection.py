# %% [markdown]
# # 14. VLM による外観検査：「どこが」「どう違う」を言葉で報告する
#
# **セミナー対応章**：第5章（3）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU（VLM 約9GB）　｜　**所要時間**：約7分（初回はモデルDLで+5分）
#
# 人間の検査員は、良品と見比べて「左上に小さな穴、まわりにひび」のように **違和感を言葉で説明** できます。これを AI で再現します。
#
# 1. **どこが（異常マップ）**：良品画像 20 枚だけから、各場所の特徴が良品とどれだけ違うかを測る（PatchCore 方式、学習不要）
# 2. **どう違う（VLM）**：異常の場所を拡大し、同じ場所の良品と並べて、視覚言語モデル（Qwen3-VL-4B）に **言葉で説明** させる
# 3. **登録済み欠陥との照合**：事前に登録した欠陥（「穴・へこみ」「ひび割れ」…）のどれに当たるかを VLM に選ばせる
# 4. **位置・大きさはプログラムで測る**（VLM に数値を言わせると、もっともらしい架空の数値を答えやすい）
# 5. **異常マップでは見つけにくい違和感**（くもり・妙な光沢・個数の不足）を、VLM に登録したチェック項目で確かめる
#
# **関連する先行研究**
# - J. Jeong et al., *WinCLIP*, CVPR 2023：「傷のある〇〇の写真」など、言葉で登録した状態との類似度で、学習なしに異常を検出
# - Z. Gu et al., *AnomalyGPT*, AAAI 2024：大規模視覚言語モデルで異常の有無と位置を対話形式で答える
# - X. Jiang et al., *MMAD*, ICLR 2025：VLM の外観検査ベンチマーク。最良の GPT-4o でも正答率 約75% で、産業用途には不十分と報告
# - J. Xu et al., *Anomaly-OneVision*, CVPR 2025：「二度見る」仕組みで細かな異常を見つけ、理由を説明する VLM
#
# データ：VisA（Amazon, Zou et al., ECCV 2022、**CC BY 4.0**）。Hugging Face のミラー（BrachioLab/visa）から一部だけ取得します。

# %% [setup]

# %%
if IN_COLAB:
    pip_install("transformers>=4.57")       # Qwen3-VL には新しい transformers が必要
import io, json, urllib.request, cv2, torchvision
from PIL import ImageDraw

def visa_rows(split, offset, n):
    url = f"https://datasets-server.huggingface.co/rows?dataset=BrachioLab/visa&config=default&split={split}&offset={offset}&length={n}"
    rows = json.load(urllib.request.urlopen(url))["rows"]
    return [(Image.open(io.BytesIO(urllib.request.urlopen(r["row"]["image"]["src"]).read())).convert("RGB"), r["row"]["label"]) for r in rows]

CATS = ["chewinggum", "candle", "pcb1"]
data = {}
for cat in CATS:
    goods = [im for im, _ in visa_rows(f"{cat}.train", 0, 20)]                        # 良品 20 枚（これだけを使う）
    tests = visa_rows(f"{cat}.test", 0, 6) + visa_rows(f"{cat}.test", 100, 6)          # 検査画像：不良 6 枚＋良品 6 枚
    data[cat] = (goods, tests)
    print(cat, "良品", len(goods), "枚 / 検査画像", len(tests), "枚（うち不良", sum(l for _, l in tests), "枚）")
show([data[c][0][0] for c in CATS] + [data[c][1][0][0] for c in CATS],
     [f"{c}: good" for c in CATS] + [f"{c}: defective" for c in CATS], size=3.2, rows=2)

# %% [markdown]
# ## 1. どこが：良品だけから異常マップを作る（PatchCore 方式）
# ImageNet で学習済みの CNN（Wide ResNet-50）の中間特徴を、画像の各場所（48×48 か所）で取り出し、
# **良品 20 枚の全ての場所の特徴の中で、最も近いものとの距離** を異常度とします。良品に一度も現れない見た目ほど、距離が大きくなります。
# 判定の閾値は、良品どうしで測った異常度の最大値とします（不良品の画像は一切使わない）。

# %%
S = 384
bb = torchvision.models.wide_resnet50_2(weights="IMAGENET1K_V1").to(DEVICE).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)

def feat(im):
    x = (to_tensor(im.resize((S, S))).to(DEVICE) - MEAN) / STD
    with torch.no_grad():
        h = bb.maxpool(bb.relu(bb.bn1(bb.conv1(x))))
        l2 = bb.layer2(bb.layer1(h)); l3 = bb.layer3(l2)
        f2 = torch.nn.functional.avg_pool2d(l2, 3, 1, 1)
        f3 = torch.nn.functional.interpolate(torch.nn.functional.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear")
    return torch.cat([f2, f3], 1)[0].flatten(1).T                 # (48*48 か所) × 1536 次元

def anomaly_map(f, bank):
    d = torch.cdist(f, bank).min(1).values.view(S // 8, S // 8).cpu().numpy()
    return cv2.GaussianBlur(cv2.resize(d, (S, S)), (0, 0), 4)

models = {}
for cat in CATS:
    fs = [feat(g) for g in data[cat][0]]
    thr = max(anomaly_map(f, torch.cat([g for j, g in enumerate(fs) if j != i])).max() for i, f in enumerate(fs))
    models[cat] = (torch.cat(fs), thr, fs)
    scores = [(anomaly_map(feat(im), models[cat][0]).max(), lab) for im, lab in data[cat][1]]
    acc = np.mean([(s > thr) == bool(l) for s, l in scores])
    print(f"{cat:11s} 閾値 {thr:.2f}  正解率 {acc*100:.0f}%  不良 {[round(float(s),2) for s, l in scores if l]}  良品 {[round(float(s),2) for s, l in scores if not l]}")

# %% [markdown]
# ## 2. どう違う：異常の場所を拡大して、VLM に言葉で説明させる
# - 異常マップの最大の場所を含む領域を切り出し、**最も似ている良品画像の同じ場所** と並べて VLM に渡す
# - 質問 1：「良品と比べて何が違うか」を 1 文で説明（形・色・光沢）
# - 質問 2：登録済みの欠陥リストのどれに当たるかを番号で選ぶ
# - 位置（左上など）と大きさは、異常マップからプログラムで計算する（`MM_PER_PX` にカメラの校正値を入れれば mm で表示）

# %%
from transformers import AutoProcessor, AutoModelForImageTextToText
VLM = "Qwen/Qwen3-VL-4B-Instruct"
processor = AutoProcessor.from_pretrained(VLM)
vlm = AutoModelForImageTextToText.from_pretrained(VLM, dtype=DTYPE, device_map=DEVICE)

def ask(images, question, max_new_tokens=150):
    content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": question}]
    inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = vlm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

REGISTERED = {                                                    # 事前に登録しておく欠陥（現場の言葉で）
    "chewinggum": ["穴・へこみ", "ひび割れ", "欠け", "異物・汚れ・変色", "光沢・表面の質感の異常"],
    "candle": ["欠け・えぐれ", "ひび割れ", "異物・汚れ", "変色", "芯の異常"],
    "pcb1": ["ピンの曲がり", "ピンの欠落", "部品の欠落・ずれ", "キズ・焦げ", "異物の付着"],
}
MM_PER_PX = None                                                  # 例：0.05（1 画素 = 0.05 mm）

def position_word(cx, cy, W, H):
    v = "上" if cy < H / 3 else ("下" if cy > 2 * H / 3 else "")
    h = "左" if cx < W / 3 else ("右" if cx > 2 * W / 3 else "")
    return (h + v) if (h and v) else (h or v or "中央") + ("寄り" if (h or v) else "")

def inspect(cat, im):
    bank, thr, fs = models[cat]
    f = feat(im); m = anomaly_map(f, bank)
    if m.max() <= thr:
        return {"判定": "良品"}, None, None, m
    W, H = im.size
    y, x = np.unravel_index(m.argmax(), m.shape)
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > max(thr, 0.6 * m.max())).astype(np.uint8))
    bx, by, bw, bh = st[lab[y, x], :4]
    sx, sy = W / S, H / S
    cx, cy, size = (bx + bw / 2) * sx, (by + bh / 2) * sy, max(bw * sx, bh * sy)
    side = max(size * 2.2, 0.22 * min(W, H))
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    ref = data[cat][0][int(np.argmin([torch.dist(f.mean(0), g.mean(0)).item() for g in fs]))]   # 最も似ている良品
    crop_t, crop_r = im.crop(box).resize((336, 336)), ref.crop(box).resize((336, 336))
    desc = ask([crop_r, crop_t], "1枚目は良品の画像の一部、2枚目は検査する製品の同じ位置の拡大画像です。"
               "2枚目にあって1枚目にない異常を、形・色・光沢の違いに注目して、日本語の1文で具体的に説明してください。")
    items = "、".join(f"{i+1}:{c}" for i, c in enumerate(REGISTERED[cat]))
    kind = ask([crop_r, crop_t], f"1枚目は良品、2枚目は検査画像の同じ位置の拡大です。2枚目に見られる異常は、次の登録済み欠陥のどれに当てはまりますか？ {items}。"
               "最も当てはまる番号を1つだけ数字で答えてください。", 5)
    k = int(re.search(r"\d", kind).group()) if re.search(r"\d", kind) else 0
    size_txt = f"約 {size * MM_PER_PX:.1f} mm" if MM_PER_PX else f"約 {size:.0f} 画素（画像の幅の {100 * size / W:.0f}%）"
    report = {"判定": "不良品", "位置": position_word(cx, cy, W, H), "大きさ": size_txt,
              "登録欠陥": REGISTERED[cat][k - 1] if 1 <= k <= len(REGISTERED[cat]) else "該当なし", "説明": desc}
    return report, box, crop_t, m

import re
figs, titles = [], []
for cat in ["chewinggum", "candle"]:
    for im, lab in data[cat][1][:3] + data[cat][1][6:7]:              # 不良 3 枚＋良品 1 枚
        t = time.time()
        rep, box, crop, m = inspect(cat, im)
        print(f"[{cat} / 正解: {'不良' if lab else '良品'}] {json.dumps(rep, ensure_ascii=False)}  ({time.time()-t:.1f}s)")
        vis = im.copy()
        if box:
            ImageDraw.Draw(vis).rectangle(box, outline=(255, 0, 0), width=10)
        figs += [vis, crop if crop else Image.new("RGB", (336, 336), "white")]
        titles += [f"{cat} ({'defective' if lab else 'good'})", "zoom" if crop else "no anomaly"]
show(figs, titles, size=2.6, rows=2)
gpu_mem()

# %% [markdown]
# ## 3. 比較：画像全体を見せて「欠陥を説明して」と聞くだけだと？
# 場所を教えずに、画像 1 枚と例文つきの質問だけを VLM に渡します。

# %%
for cat in ["chewinggum", "candle"]:
    for im, lab in data[cat][1][:1] + data[cat][1][6:7]:
        a = ask([im.resize((448, int(448 * im.height / im.width)))],
                "あなたは外観検査員です。この製品に欠陥があれば、位置と大きさと見た目を1文で説明してください。"
                "例：「左上に長さ約5mmの線状のキズ、正常品と比べて光沢が強い」。欠陥が無ければ「欠陥なし」と答えてください。")
        print(f"[{cat} / 正解: {'不良' if lab else '良品'}] {a}")

# %% [markdown]
# **観察ポイント**：画像全体だけを渡すと、大きな欠陥（ガムの汚れ）は説明できても、**小さな欠陥（ろうそくの欠け、画像の幅の 5% ほど）は「欠陥なし」と見落とします**。
# また、小さなモデルほど、例文の数値（約5mm など）をそのまま使った架空の説明を返しがちです（講師の試行では Qwen2.5-VL-3B で頻発）。
# **「どこを見るか」は異常検知、「何がどう違うか」は VLM、「数値」はプログラム** と役割を分けると、説明の信頼性が上がります。

# %% [markdown]
# ## 4. 異常マップでは見つけにくい「違和感」：くもり・妙な光沢・個数の不足
# ひび・欠け・汚れのように **その場所だけ見れば分かる欠陥** は、異常マップで十分に見つかります。
# 一方、現場では次のような「違和感」も問題になります。良品の画像を加工して 3 種類を作り、異常マップと VLM で比べます。
# - **くもり**：カプセルの表面の一部が白く濁る（形は正常で、色と透明感だけが変わる）
# - **妙な光沢**：ろうそくの 1 個だけに、照明と合わない「てかり」が出る
# - **個数の不足**：4 個並ぶはずのマカロニが 3 個（どの場所も局所的には正常で、全体を見て初めて分かる）
#
# VLM には、登録しておいた「違和感」のチェック項目を 1 つずつ「はい／いいえ」で聞き、個数は数えさせて登録値と比べます。
# （「違いを説明して」と自由に聞くと、良品どうしでも何かしらの違いを答えてしまうため）

# %%
def soft_ellipse(W, H, cx, cy, rx, ry, ang=0, blur=25):
    m = np.zeros((H, W), np.float32)
    cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), ang, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), blur)

def add_haze(im, cx, cy, rx, ry, a=0.8):
    """くもり：ぼかした画像に白っぽい膜を重ねる"""
    x = np.asarray(im).astype(np.float32); H, W = x.shape[:2]
    m = soft_ellipse(W, H, cx, cy, rx, ry, blur=40)[..., None] * a
    veil = cv2.GaussianBlur(x, (0, 0), 6) * 0.35 + 0.65 * 205
    return Image.fromarray(np.clip(x * (1 - m) + veil * m, 0, 255).astype(np.uint8))

def add_gloss(im, cx, cy, L, w, ang, k=90):
    """妙な光沢：細長いてかり＋その周りのぼんやりした反射"""
    x = np.asarray(im).astype(np.float32); H, W = x.shape[:2]
    m = np.clip(soft_ellipse(W, H, cx, cy, L, w, ang, blur=6) + 0.35 * soft_ellipse(W, H, cx, cy, L * 1.8, w * 4, ang, blur=30), 0, 1)
    return Image.fromarray(np.clip(x + k * m[..., None], 0, 255).astype(np.uint8))

def remove_piece(im, mg=90, blk=160, ov=40, seed=0):
    """右下のマカロニを 1 個消し、同じ画像の背景（マット）の小片で埋める"""
    x = np.asarray(im).astype(np.float32); H, W = x.shape[:2]
    piece = cv2.morphologyEx(((x[..., 0] > x[..., 1] + 20) & (x[..., 0] > 90)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(piece)
    sel = [i for i in range(1, n) if st[i, 4] > 3000 and cen[i, 0] > W / 2 and cen[i, 1] > H / 2]
    ys, xs = np.where(np.isin(lab, sel))
    y0, y1, x0, x1 = max(ys.min() - mg, 0), min(ys.max() + mg, H), max(xs.min() - mg, 0), min(xs.max() + mg, W)
    busy = cv2.dilate(piece, np.ones((2 * mg, 2 * mg), np.uint8))                 # マカロニと影の近くは使わない
    ring = np.zeros((H, W), bool); ring[y0:y1, x0:x1] = True; ring[y0 + 20:y1 - 20, x0 + 20:x1 - 20] = False
    ring_mean = x[ring & (busy == 0)].mean(0)                                      # 埋める場所の周りの平均色
    cands = [(yy, xx, x[yy:yy + blk, xx:xx + blk].mean((0, 1))) for yy in range(0, H - blk, 20) for xx in range(0, W - blk, 20)
             if busy[yy:yy + blk, xx:xx + blk].max() == 0]
    cands = sorted(cands, key=lambda c: np.abs(c[2] - ring_mean).sum())[:max(8, len(cands) // 5)]
    rng = np.random.default_rng(seed)
    win = cv2.GaussianBlur(np.pad(np.ones((blk - 2 * ov, blk - 2 * ov), np.float32), ov), (0, 0), ov / 2)[..., None] + 1e-3
    fill = np.zeros((y1 - y0, x1 - x0, 3), np.float32); wsum = np.zeros((y1 - y0, x1 - x0, 1), np.float32)
    for yy in range(0, y1 - y0, blk - ov):
        for xx in range(0, x1 - x0, blk - ov):
            cy, cx, cm = cands[rng.integers(len(cands))]
            p = x[cy:cy + blk, cx:cx + blk] - cm + ring_mean
            h, w = min(blk, y1 - y0 - yy), min(blk, x1 - x0 - xx)
            fill[yy:yy + h, xx:xx + w] += p[:h, :w] * win[:h, :w]; wsum[yy:yy + h, xx:xx + w] += win[:h, :w]
    e = mg // 3; fm = np.zeros((y1 - y0, x1 - x0), np.float32); fm[e:-e, e:-e] = 1
    fm = cv2.GaussianBlur(fm, (0, 0), e / 2)[..., None]
    out = x.copy(); out[y0:y1, x0:x1] = x[y0:y1, x0:x1] * (1 - fm) + fill / wsum * fm
    return Image.fromarray(out.clip(0, 255).astype(np.uint8))

for cat in ["capsules", "macaroni1"]:                                           # 新しい品種：良品 20 枚と、合成のもとにする良品
    goods = [im for im, _ in visa_rows(f"{cat}.train", 0, 20)]
    data[cat] = (goods, visa_rows(f"{cat}.test", 100, 6))
for cat in ["capsules", "macaroni1"]:
    fs = [feat(g) for g in data[cat][0]]
    thr = max(anomaly_map(f, torch.cat([g for j, g in enumerate(fs) if j != i])).max() for i, f in enumerate(fs))
    models[cat] = (torch.cat(fs), thr, fs)

goods_of = lambda cat: [im for im, lab in data[cat][1] if not lab]            # 異常マップの学習に使っていない良品
cap, can, mac = goods_of("capsules")[0], goods_of("candle")[0], goods_of("macaroni1")[0]
cases = [("capsules", "くもり", add_haze(cap, cap.width * 0.55, cap.height * 0.45, cap.width * 0.16, cap.height * 0.2)),
         ("candle", "妙な光沢", add_gloss(can, can.width * 0.27, can.height * 0.24, can.width * 0.06, can.width * 0.02, -35)),
         ("macaroni1", "個数の不足", remove_piece(mac))]

CHECKS = ["表面の一部が白く曇っている（白濁している）", "一部だけ、周りと違う不自然な光沢・てかりがある"]
COUNT = {"candle": ("キャンドル", 4), "macaroni1": ("パスタ", 4)}               # 登録しておく個数（カプセルは並びが不定なので数えない）
r1024 = lambda z: z.resize((1024, int(1024 * z.height / z.width)))

def vlm_check(cat, im):
    ref = data[cat][0][0]
    found = []
    for c in CHECKS:
        a = ask([r1024(ref), r1024(im)], f"1枚目は良品の見本、2枚目は検査する製品です。2枚目は、見本と比べて「{c}」状態ですか？"
                "「はい」か「いいえ」で答え、「はい」の場合は場所と様子を短く続けてください。", 80)
        if a.startswith("はい"):
            found.append(a)
    if cat in COUNT:
        noun, n_ok = COUNT[cat]
        a = ask([r1024(im)], f"この画像に{noun}は何個写っていますか？数字だけで答えてください。", 5)
        n = int(re.search(r"\d+", a).group()) if re.search(r"\d+", a) else -1
        if n != n_ok:
            found.append(f"{noun}が {n} 個（登録値は {n_ok} 個）")
    return found

figs, titles = [], []
for (cat, kind, im), name in zip(cases, ["haze", "odd gloss", "missing piece"]):
    bank, thr, fs = models[cat]
    m = anomaly_map(feat(im), bank)
    print(f"[{kind}] 異常マップ: 最大 {m.max():.2f}（閾値 {thr:.2f}）→ {'検出' if m.max() > thr else '見逃し'}")
    print(f"   VLM（合成した不良）: {vlm_check(cat, im) or '違和感なし'}")
    print(f"   VLM（良品で確認）  : {vlm_check(cat, goods_of(cat)[1]) or '違和感なし'}")
    figs += [im, Image.fromarray((plt.cm.jet(np.clip(m / 4, 0, 1))[..., :3] * 255).astype(np.uint8))]
    titles += [f"{name} (synthesized)", f"anomaly map {m.max():.2f} (thr {thr:.2f})"]
show(figs, titles, size=3.0, rows=3)

# %% [markdown]
# **観察ポイント**：くもりと個数の不足は、どの場所も「良品のどこかに似た見た目」なので、異常マップの値が閾値を超えません（くもりは閾値ぎりぎり）。
# 妙な光沢は異常マップでも見つかりますが、分かるのは「どこかが違う」までで、**「てかり」だと言葉にできるのは VLM** です。
# VLM は、見本と並べて **登録したチェック項目を 1 つずつ聞く** と、良品では「違和感なし」と答え、合成した不良では「中央付近が白く曇っている」「左上のてかりが不自然」「3 個」と答えました。
# ただし、くもりや光沢の程度がもっと弱いと VLM も見逃します（講師の試行では、光沢を弱めると「いいえ」になった）。判定の最終確認は人が行う前提で使います。

# %% [markdown]
# ## まとめ
# - 良品だけで作る異常マップ（PatchCore 方式）で、**欠陥の有無と場所** を高精度に見つける
# - VLM は、**拡大した異常部分を良品と見比べて言葉で説明** し、登録済みの欠陥名に対応づける → 検査記録・作業者への指示・不良の傾向分析に使える
# - くもり・個数の不足のように **局所的には正常に見える違和感** は異常マップで見逃しやすい。見本と並べ、登録したチェック項目を 1 つずつ VLM に聞くと見つけられる
# - VLM に数値（大きさ）や判定そのものを任せない。MMAD ベンチマークでも最良の VLM の正答率は約75%（2025年）
# - 発展：説明文をもとに、生成AIで「良品に戻した画像」を作って見比べる（第4章の修復による異常検知）、自社の欠陥画像で VLM を LoRA 追加学習する
