# %% [markdown]
# # 00. はじめに：実行環境の確認（最初にこれを実行してください）
#
# 本講座「生成AIによる画像復元技術と最新AIモデルの応用」の演習ノートブックです。
#
# ## 使い方（Google Colab）
# 1. **［ファイル］→［ドライブにコピーを保存］** で自分用のコピーを作る（元のノートブックは編集できません）
# 2. **［ランタイム］→［ランタイムのタイプを変更］→「T4 GPU」** を選んで保存
# 3. 上のセルから順に ▶ ボタン（または Shift+Enter）で実行
# 4. 終わったら **［ランタイム］→［ランタイムを接続解除して削除］**
#
# ## 使い方（ローカルの GPU PC）
# `docs/ローカル環境構築ガイド.pdf` の手順で Python 環境を作り、Jupyter Lab / VS Code で開いて実行します。コードは Colab と共通です。
#
# ## ノートブック一覧
# | No. | 内容 | 推奨環境 |
# |---|---|---|
# | 01 | 劣化モデルと古典的復元（Wiener・補間） | CPU可 |
# | 02 | GAN・拡散モデル・CLIP の仕組み | T4 |
# | 03 | 超解像：Real-ESRGAN vs 拡散モデル | T4 |
# | 04 | Inpainting / Outpainting・境界のなじませ | T4 |
# | 05 | Deblurring / Denoising（DPIR・SCUNet） | T4 |
# | 06 | 条件付け：img2img・ControlNet・IP-Adapter | T4 |
# | 07 | 外観検査・異常検知・疑似欠陥生成 | T4 |
# | 08 | 評価指標とハルシネーション対策 | T4 |
# | 09 | 軽量化：蒸留・量子化・ONNX | T4 |
# | 10 | LoRA による現場特化チューニング | L4推奨（有料） |
# | 11 | マルチモーダル：言葉で指示する復元・VLM | L4推奨（有料） |
# | 12 | 物理モデルを組み込んだ AI：スマホ→一眼・塑性変形 | T4 |
# | 13 | 奥行きで変わるピンボケ：場所ごとの逆カーネル | T4 |
# | 14 | VLM による外観検査：違和感を言葉で報告 | T4 |
# | 15 | ComfyUI on Colab：SUPIR による高品質復元 | A100推奨（有料・ハイメモリ） |

# %% [setup]

# %% [markdown]
# ## 実行環境の詳細チェック

# %%
import shutil, platform
print("Python  :", platform.python_version())
try:
    import psutil
    print(f"システムRAM: {psutil.virtual_memory().total / 1024**3:.1f} GB")
except ImportError:
    pass
print(f"ディスク空き: {shutil.disk_usage('.').free / 1024**3:.0f} GB")

if DEVICE == "cuda":
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nGPU メモリ {vram:.1f} GB で実行できるノートブック:")
    print("  01〜09: OK" if vram >= 7 else "  01〜05, 09 程度（他はメモリ不足の可能性）")
    print("  10〜14: OK" if vram >= 14 else "  10〜14: メモリ不足の可能性（L4以上を推奨）")
    print("  15    : OK（システムRAMも 40GB 以上必要）" if vram >= 22 else "  15    : GPU メモリ不足（A100 / L4 + ハイメモリを推奨）")
else:
    print("\n⚠ GPU なし：ノートブック 01 以外は非常に遅くなります。")

# %% [markdown]
# ## 簡単な動作テスト：GPU で行列計算

# %%
a = torch.randn(4096, 4096, device=DEVICE, dtype=DTYPE)
t = time.time()
for _ in range(20):
    b = a @ a
if DEVICE == "cuda":
    torch.cuda.synchronize()
dt = time.time() - t
print(f"4096x4096 行列積 x20: {dt:.2f}s  （約 {20 * 2 * 4096**3 / dt / 1e12:.1f} TFLOPS）")

# %% [markdown]
# ## 結果を Google Drive に保存する方法（Colab）
# ランタイムが切断されると、作業フォルダのファイルはすべて消えます。残したい結果は Drive に保存してください。

# %%
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")                       # 初回は許可を求める画面が出ます
    os.makedirs("/content/drive/MyDrive/genai_restoration", exist_ok=True)
    load_sample("astronaut").save("/content/drive/MyDrive/genai_restoration/test.png")
    print("Drive に保存しました: MyDrive/genai_restoration/test.png")
else:
    print("ローカル実行なので Drive マウントは不要です。")

# %% [markdown]
# ## Colab の有料枠について（要点）
# - 有料枠は **コンピューティングユニット（CU）のプリペイド制**。GPU ランタイムに **接続している間** 減り続けます（計算していなくても）
# - T4 は約 1〜2 CU/時、A100 は約 5〜15 CU/時（変動します。右上のリソース表示で確認）
# - **使い終わったら［ランタイムを接続解除して削除］** ／ 月額プランは自動更新なので不要になったら解約
# - 詳しくは `docs/Colab有料枠ガイド.pdf`
