# ComfyUI ワークフロー（配布用）

ComfyUI の画面に JSON ファイルを **ドラッグ＆ドロップ** すると読み込まれます。
赤枠のノードが出たら **Manager → Install Missing Custom Nodes** で不足ノードを入れて再起動し、
下表のモデルを所定のフォルダに置いてください。

> **RTX 50 シリーズ（Blackwell）の場合**：xformers が非対応のことがあります。`run_nvidia_gpu.bat` の起動オプションに `--disable-xformers` を付けてください（エラー例：`No operator found for memory_efficient_attention_forward`）。

## 1. 標準ノードだけで動くワークフロー（まずはここから）

| ファイル | 内容 | 必要なモデル（置き場所） |
|---|---|---|
| `upscale_realesrgan.json` | Real-ESRGAN による 4 倍超解像 | `RealESRGAN_x4plus.pth` → `models/upscale_models/` |
| `inpaint_sd15.json` | 不要物・キズの除去（SD1.5 Inpainting）。Load Image を右クリック →「Open in MaskEditor」で消したい部分を塗る | `sd-v1-5-inpainting.ckpt` → `models/checkpoints/` |
| `outpaint_sd15.json` | 画角拡張（上下左右の拡張幅を指定） | 同上 |

- これらは **API 形式**（ノード ID と入力値の辞書）で保存しています。画面に読み込むと自動でノードが配置されます。
- Inpainting / Outpainting では、生成結果を **マスク外は元画像のまま** 合成しています（ImageCompositeMasked）。

## 2. 講師が解説論文で使った比較ワークフロー（UI 形式）

| ファイル | 内容 | 必要なカスタムノード |
|---|---|---|
| `sr_comparison_workflow.json` | 劣化画像の作成 → Real-ESRGAN / StableSR / DiffBIR / SUPIR の比較（論文の図4） | ComfyUI-SUPIR, ComfyUI-DiffBIR, Comfyui-StableSR, seminar_nodes |
| `face_sr_comparison_workflow.json` | 顔画像：GFPGAN / CodeFormer / DiffBIR / SUPIR / **DifFace** の比較（論文の図5＋DifFace） | 上記＋ comfyui_gfpgan, facerestore_cf、DifFace リポジトリ（下記） |
| `sr_SUPIR.json` | SUPIR 単体（タイル処理版、1024px 程度以上の画像向け） | ComfyUI-SUPIR, seminar_nodes |

`api/` フォルダには、同じワークフローの API 形式版（Python から実行する用）があります。

### カスタムノードの入手先
| ノード | URL |
|---|---|
| ComfyUI-SUPIR | https://github.com/kijai/ComfyUI-SUPIR |
| ComfyUI-DiffBIR | https://github.com/jtscmw01/ComfyUI-DiffBIR |
| Comfyui-StableSR | https://github.com/gameltb/Comfyui-StableSR |
| comfyui_gfpgan | https://github.com/comfyorg/comfyui_gfpgan |
| facerestore_cf（CodeFormer） | https://github.com/mav-rik/facerestore_cf |
| **seminar_nodes**（劣化画像の作成：縮小＋ノイズ、ぼかし／**DifFace 顔復元**） | 本フォルダの `custom_nodes/seminar_nodes` を `ComfyUI/custom_nodes/` にコピー |
| DifFace 本体（DifFace ノードが使う） | `ComfyUI` フォルダで `git clone https://github.com/zsyOAOA/DifFace`（ZIP を展開して `ComfyUI/DifFace` に置いてもよい）。重み約 0.6GB は初回実行時に自動ダウンロード |

### モデル（主なもの）
| モデル | 置き場所 | 入手先 |
|---|---|---|
| `sd_xl_base_1.0.safetensors` | checkpoints | huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 |
| `SUPIR-v0Q_fp16.safetensors` | checkpoints | huggingface.co/Kijai/SUPIR_pruned |
| `v2-1_768-ema-pruned.safetensors`, `stablesr_768v_000139.ckpt` | checkpoints / stablesr | Comfyui-StableSR の README を参照 |
| DiffBIR の重み | models/diffbir | ComfyUI-DiffBIR の README を参照（初回自動ダウンロード） |
| `RealESRGAN_x4plus.pth` | upscale_models | github.com/xinntao/Real-ESRGAN/releases |
| `GFPGANv1.4.pth`, `codeformer.pth` | facerestore_models | 各ノードの README を参照 |
| DifFace の重み（`iddpm_ffhq512_ema500000.pth`, `swinir_restoration512_L1.pth`） | `ComfyUI/DifFace/weights/` | 初回実行時に自動ダウンロード |

**ライセンス注意**：SUPIR（独自ライセンス）、CodeFormer・StableSR・DifFace（S-Lab License）は **非商用** です。業務での利用前に必ず確認してください。

### トラブルシューティング
| 症状 | 対処 |
|---|---|
| SUPIR で `maximum recursion depth exceeded` | 同じセッションで SUPIR を繰り返し実行すると起きることがある。ComfyUI を再起動するか、画面右上の「Unload Models」（モデルとキャッシュの解放）を実行してから再実行 |
| `No operator found for memory_efficient_attention_forward` | RTX 50 系で xformers 非対応。起動オプションに `--disable-xformers` |
| SUPIR で `The size of tensor a must match ...` | 画像がタイルサイズより小さい。`sampler` を `RestoreDPMPP2MSampler`（タイルなし）にするか、1024px 程度以上の画像を使う |
| メモリ不足（OOM） | SUPIR の `fp8_unet` を true、`use_tiled_vae` を true、画像を小さく |
| DifFace ノードで「DifFace が見つかりません」 | `ComfyUI/DifFace` に DifFace のリポジトリを置く。別の場所なら `difface_dir` に絶対パスを指定 |
| DifFace ノードで `No module named ...` | ポータブル版の Python に不足ライブラリを入れる：`.\python_embeded\python.exe -m pip install omegaconf einops timm gdown`（docs/ローカル環境構築ガイド.pdf の A-4） |

## 3. Python から実行する（バッチ処理）

```bash
# ComfyUI を起動した状態で（画像は ComfyUI の input フォルダに置く）
python tools/run_api.py upscale_realesrgan.json --image img001.png img002.png img003.png
```

- `tools/run_api.py`：API 形式のワークフローを実行するスクリプト
- `tools/ui_to_api.py`：UI 形式 → API 形式への変換（画面の Workflow → Export (API) と同等）
- `tools/make_core_workflows.py`：標準ノードのワークフローを生成するスクリプト

## 動作確認
2026年10月、ComfyUI 0.x（frontend 1.42）/ RTX 5090 Laptop GPU で、`upscale_realesrgan`・`inpaint_sd15`・`outpaint_sd15`・`sr_SUPIR`（1024px 入力）・`sr_comparison_workflow`・`face_sr_comparison_workflow`（DifFace を含む全手法）の実行を確認。

顔比較ワークフローの処理時間（512px、モデル読み込み後）：GFPGAN・CodeFormer 1秒未満、DiffBIR 約5秒（VRAM 約8GB）、SUPIR 約23秒（1024px、約9GB）、DifFace 約10秒（毎回のモデル読み込みを含む、約1.6GB）。
