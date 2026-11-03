# %% [markdown]
# # 09. モデルの軽量化・高速化：FP16・知識蒸留・INT8量子化・ONNX
#
# **セミナー対応章**：第4章（2）　｜　**推奨ランタイム**：Colab無料枠 T4 GPU　｜　**所要時間**：約6分
#
# 工場のライン PC や Jetson などのエッジデバイスで動かすには、モデルを小さく・速くする必要があります。
# 1. 大きいモデル（Real-ESRGAN, 1670万パラメータ）と軽量モデル（Compact, 120万）の速度比較
# 2. **FP16（半精度）** で高速化
# 3. **知識蒸留**：大きいモデル（教師）の出力を小さいモデル（生徒）に真似させる
# 4. **ONNX 変換 + INT8 量子化**：PyTorch 不要のランタイムで CPU 推論（エッジ実装の第一歩）

# %% [setup]

# %%
if IN_COLAB:
    pip_install("onnx", "onnxruntime", "onnxscript")
import cv2
import spandrel
import torch.nn as nn

def n_params(m):
    return sum(p.numel() for p in m.parameters())

def bench(fn, n=10):
    fn()                                                    # ウォームアップ
    if DEVICE == "cuda": torch.cuda.synchronize()
    t = time.time()
    for _ in range(n):
        fn()
    if DEVICE == "cuda": torch.cuda.synchronize()
    return (time.time() - t) / n * 1000                     # ms

# %% [markdown]
# ## 1. 大きいモデル vs 軽量モデル、FP32 vs FP16

# %%
teacher = spandrel.ModelLoader().load_from_file(
    download("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth")).to(DEVICE).eval()
compact = spandrel.ModelLoader().load_from_file(
    download("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth")).to(DEVICE).eval()

hr = load_portrait()                          # 顔が大きく写った高画質画像（512x512）
lr = hr.resize((128, 128), Image.BICUBIC)
x = to_tensor(lr).to(DEVICE)

rows = []
with torch.no_grad():
    for name, m in [("Real-ESRGAN x4plus", teacher), ("Real-ESRGAN Compact", compact)]:
        rows.append((name, "FP32", n_params(m.model), bench(lambda: m(x))))
        if DEVICE == "cuda":
            m.model.half()
            rows.append((name, "FP16", n_params(m.model), bench(lambda: m(x.half()))))
            m.model.float()
print(f"{'model':22s} {'prec':5s} {'params':>12s} {'time(128->512)':>15s}")
for r in rows:
    print(f"{r[0]:22s} {r[1]:5s} {r[2]:12,d} {r[3]:12.1f} ms")

# 画質も確認する（速さだけでなく、結果の見た目が許容できるか）
with torch.no_grad():
    sr1 = {"x4plus FP32": tensor_to_pil(teacher(x)), "Compact FP32": tensor_to_pil(compact(x))}
    if DEVICE == "cuda":
        compact.model.half(); sr1["Compact FP16"] = tensor_to_pil(compact(x.half())); compact.model.float()
box = (110, 40, 430, 360)                     # 顔の部分を拡大
show([lr.resize(hr.size, Image.BICUBIC).crop(box)] + [v.crop(box) for v in sr1.values()] + [hr.crop(box)],
     ["Bicubic"] + [f"{n}\n{psnr(v, hr):.2f} dB" for n, v in sr1.items()] + ["Ground truth"], size=3.2)

# %% [markdown]
# ## 2. 知識蒸留（Knowledge Distillation）
# 教師（Real-ESRGAN x4plus）の出力を「正解」として、パラメータ数 1/650 程度の小さな生徒ネットワークを学習します。
# 自社の画像（ラインカメラの画像など）で蒸留すれば、**その現場に特化した小型モデル** が作れます。

# %%
class TinySR(nn.Module):
    """3層の畳み込み + PixelShuffle による 4倍超解像（ESPCN 風）"""
    def __init__(self, c=32):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(3, c, 5, padding=2), nn.PReLU(c),
                                  nn.Conv2d(c, c, 3, padding=1), nn.PReLU(c),
                                  nn.Conv2d(c, 3 * 16, 3, padding=1))
        self.up = nn.PixelShuffle(4)
    def forward(self, x):
        base = nn.functional.interpolate(x, scale_factor=4, mode="bilinear", align_corners=False)
        return base + self.up(self.body(x))                    # 残差学習：補間との差だけ学ぶ

student = TinySR().to(DEVICE)
print(f"teacher params: {n_params(teacher.model):,}   student params: {n_params(student):,}")

# 学習データ：サンプル画像からランダムに切り出した低解像度パッチ（教師の出力が正解）
pool = [to_np(load_sample(n)) for n in ("astronaut", "coffee", "chelsea", "rocket")] + [to_np(hr)]
def random_lr_batch(bs=16, p=48):
    out = []
    for _ in range(bs):
        im = pool[np.random.randint(len(pool))]
        s = max(np.random.uniform(0.25, 0.5), (p + 1) / min(im.shape[:2]))
        small = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        y0, x0 = np.random.randint(0, small.shape[0] - p), np.random.randint(0, small.shape[1] - p)
        out.append(small[y0:y0 + p, x0:x0 + p])
    return torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).to(DEVICE)

opt = torch.optim.Adam(student.parameters(), 1e-3)
torch.backends.cudnn.benchmark = True                          # 毎回同じ大きさの入力なので、最速の計算方法を選ばせる
t = time.time()
for step in range(2000):
    xb = random_lr_batch()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        target = teacher(xb).float()                           # 教師の出力。教師は FP16 で動かす（1. で見たとおり結果はほぼ同じで、T4 では数倍速い）
    loss = nn.functional.l1_loss(student(xb), target)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 250 == 0:
        print(f"step {step}: L1 to teacher {loss.item():.4f}  ({time.time()-t:.0f}s)")
print(f"蒸留の学習時間: {time.time()-t:.0f}s")
student.eval()

with torch.no_grad():
    out_t = tensor_to_pil(teacher(x))
    out_s = tensor_to_pil(student(x))
    t_ms, s_ms = bench(lambda: teacher(x)), bench(lambda: student(x))
bic = lr.resize(hr.size, Image.BICUBIC)
box = (110, 40, 430, 360)                     # 顔の部分を拡大
show([bic.crop(box), out_t.crop(box), out_s.crop(box)],
     ["Bicubic", f"Teacher {t_ms:.1f}ms", f"Student {s_ms:.1f}ms (distilled)"], size=3.5)
print(f"生徒は教師の {t_ms/s_ms:.0f} 倍速。教師出力との PSNR: {psnr(out_s, out_t):.2f} dB")

# %% [markdown]
# ## 3. ONNX 変換と INT8 量子化（CPU推論）
# ONNX にすると PyTorch なしで、ONNX Runtime・TensorRT（Jetson）・OpenVINO（Intel）などで動かせます。
# INT8 量子化は重みと演算を 8bit 整数にして、**サイズ 1/4・CPU で高速化** を狙います（精度は少し落ちる）。

# %%
import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

assert "out_t" in globals(), "2. の蒸留が最後まで実行されていません（途中で止めた場合は、2. のセルをもう一度実行してください）"
student_cpu = TinySR().eval()
student_cpu.load_state_dict({k: v.cpu() for k, v in student.state_dict().items()})
dummy = torch.zeros(1, 3, 128, 128)
torch.onnx.export(student_cpu, dummy, "tiny_sr.onnx", input_names=["lr"], output_names=["sr"], opset_version=17,
                  dynamo=False)

class Calib(CalibrationDataReader):
    """量子化のスケールを決めるための代表データ（実運用では現場の画像を数十枚）"""
    def __init__(self):
        self.data = iter([{"lr": random_lr_batch(1, 128).cpu().numpy()} for _ in range(16)])
    def get_next(self):
        return next(self.data, None)

ort.quantization.shape_inference.quant_pre_process("tiny_sr.onnx", "tiny_sr_prep.onnx")
quantize_static("tiny_sr_prep.onnx", "tiny_sr_int8.onnx", Calib(), quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)

inp = {"lr": to_tensor(lr).numpy()}
res = {}
for name, path in [("ONNX FP32 (CPU)", "tiny_sr.onnx"), ("ONNX INT8 (CPU)", "tiny_sr_int8.onnx")]:
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, inp)[0]
    ms = bench(lambda: sess.run(None, inp), n=20)
    res[name] = to_pil(out[0].transpose(1, 2, 0))
    print(f"{name:18s} {os.path.getsize(path)/1024:7.1f} KB  {ms:7.1f} ms  PSNR vs teacher {psnr(res[name], out_t):.2f} dB")
with torch.no_grad():
    print(f"{'PyTorch FP32 (CPU)':18s} {'':7s}     {bench(lambda: student_cpu(torch.from_numpy(inp['lr']))):7.1f} ms")
show([out_t.crop(box), out_s.crop(box)] + [v.crop(box) for v in res.values()],
     ["Teacher", "Student (PyTorch)"] + [f"{n}\nvs teacher {psnr(v, out_t):.2f} dB" for n, v in res.items()], size=3.2)

# %% [markdown]
# **観察ポイント**：INT8 でファイルサイズは約 1/3 になります。速度は **CPU の種類やモデルの大きさで変わり**、このような小さいモデルでは速くならないこともあります（量子化・逆量子化の処理が相対的に重い）。
# 大きなモデルや INT8 演算に強いハードウェア（NPU, TensorRT on Jetson）で効果が出やすいです。**軽量化は必ず実機で実測** してください。

# %% [markdown]
# ## まとめ：軽量化の手段
# | 手段 | 効果 | 注意点 |
# |---|---|---|
# | 軽量アーキテクチャの選択 | 数倍〜数十倍速 | 画質は少し落ちる |
# | FP16 / BF16 | GPUで約2倍速・メモリ半分 | 一部モデルで数値不安定 |
# | 知識蒸留 | 現場特化の小型モデル | 教師の誤り（ハルシネーション）も継承 |
# | INT8 量子化 | サイズ1/4・CPU/NPUで高速 | キャリブレーションデータが重要 |
# | ONNX / TensorRT 変換 | フレームワーク非依存・最適化 | 対応していない演算がある |
# | 拡散モデルのステップ削減（LCM, Turbo, 1ステップ蒸留） | 数十倍速 | 品質・多様性とのトレードオフ |
#
# **Jetson での実装例**：PyTorch → ONNX → `trtexec --onnx=tiny_sr.onnx --fp16`（TensorRT エンジン化）
