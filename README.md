# 生成AIによる画像復元技術と最新AIモデルの応用 ― 演習教材

日本テクノセンター オンラインセミナー（2026年11月12日）講師：永田 毅（明治学院大学 情報数理学部）

スライドで説明した技術を、**Google Colab** または **社内のGPU搭載PC** で再現するための教材です。

## 1. Google Colab ノートブック

リンクを開いたら **［ファイル］→［ドライブにコピーを保存］** してから、上のセルから順に実行してください。

| No. | 内容 | 推奨環境 | Colab で開く |
|---|---|---|---|
| 00 | はじめに：実行環境の確認 | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/00_setup_check.ipynb) |
| 01 | 劣化モデルと古典的復元（Wiener・補間） | CPU 可 | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/01_degradation_classical.ipynb) |
| 02 | GAN・拡散モデル・CLIP の仕組み | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/02_gan_diffusion_basics.ipynb) |
| 03 | 超解像：Real-ESRGAN vs 拡散モデル | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/03_super_resolution.ipynb) |
| 04 | Inpainting / Outpainting・境界のなじませ | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/04_inpainting_outpainting.ipynb) |
| 05 | Deblurring / Denoising（DPIR・SCUNet） | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/05_deblur_denoise.ipynb) |
| 06 | 条件付け：img2img・ControlNet・IP-Adapter | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/06_conditioning_controlnet_ipadapter.ipynb) |
| 07 | 外観検査・異常検知・疑似欠陥生成 | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/07_anomaly_detection.ipynb) |
| 08 | 評価指標とハルシネーション対策 | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/08_metrics_hallucination.ipynb) |
| 09 | 軽量化：蒸留・量子化・ONNX | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/09_lightweight_distill_quantize.ipynb) |
| 10 | LoRA による現場特化チューニング | L4 推奨（有料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/10_lora_finetuning.ipynb) |
| 11 | マルチモーダル：言葉で指示する復元・VLM | L4 推奨（有料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/11_multimodal_vlm.ipynb) |
| 12 | 物理モデルを組み込んだ AI：スマホ→一眼の高画質化・塑性変形した部品の復元（約 60 分。DRUNet の調整に時間がかかる） | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/12_physics_informed_restoration.ipynb) |
| 13 | 奥行きで変わるピンボケの復元：場所ごとの逆カーネル（最小二乗法） | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/13_spatially_varying_deblur.ipynb) |
| 14 | VLM による外観検査：異常マップ＋言葉による報告 | T4（無料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/14_vlm_visual_inspection.ipynb) |
| 15 | ComfyUI on Colab：SUPIR | A100＋ハイメモリ（有料） | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/takeshi-nagata-mgu/genai-image-restoration-seminar/blob/main/notebooks/15_comfyui_on_colab_supir.ipynb) |

**有料枠について**：必ず [docs/Colab有料枠ガイド.pdf](docs/Colab有料枠ガイド.pdf) を読んでから使ってください（料金・時間制限・切断時の注意）。

## 2. ComfyUI ワークフロー（ローカル PC 用）

[comfyui_workflows/](comfyui_workflows/README.md) を参照。JSON を ComfyUI の画面にドラッグ＆ドロップするだけで使えます。

## 3. 社内の GPU PC で動かす

[docs/ローカル環境構築ガイド.pdf](docs/ローカル環境構築ガイド.pdf) を参照。ノートブックはローカルでもそのまま動きます。
講師の PC の動作環境（GPU・ライブラリの版）と、同じ環境を作るコマンドは、ガイドの「C. 講師の PC の動作環境」にあります。ライブラリの版の一覧は `requirements.txt` です。

**実行例**：`notebooks_executed/` には、講師の PC（RTX 5090 Laptop）で実行した結果つきのノートブックがあります。
有料枠が必要なノートブック（10, 11, 15）も、実行せずに結果だけ確認できます（GitHub 上でそのまま表示されます）。

## フォルダ構成

```
notebooks/            演習ノートブック（Colab / ローカル共通）
notebooks_executed/   実行結果つきのノートブック（閲覧用）
comfyui_workflows/    ComfyUI ワークフローと講座用カスタムノード
docs/                 Colab 有料枠ガイド、ローカル環境構築ガイド（PDF）
requirements.txt      ローカル用のライブラリ一覧（講師の PC で動作を確認した版）
nb_src/               ノートブックの元ソース（# %% 形式）と生成・テスト用スクリプト、ガイドの元原稿（docs_src/）
```

## 注意
- 教材のコードは学習目的のサンプルです。業務で使う場合は、各モデルの**ライセンス**（SUPIR・FLUX.1 [dev]・CodeFormer・DifFace・Qwen2.5-VL-3B などは非商用／研究用）を確認してください。一覧は docs/ローカル環境構築ガイド.pdf の末尾にあります。
- 14・15 で使うデータ・モデル：Kodak 画像（評価用フリー画像）、合金部品の画像（コードで生成）、Depth Anything V2 Small（Apache 2.0）、DRUNet（KAIR、MIT）、VisA データセット（CC BY 4.0）、Qwen3-VL-4B-Instruct（Apache 2.0）。
- 社外秘の画像をクラウド（Colab）にアップロードする前に、社内の情報セキュリティ規程を確認してください。
- 動作確認環境：ローカル Windows 11 / RTX 5090 Laptop GPU (24GB) / Python 3.12 / PyTorch 2.11 (CUDA 12.8)
