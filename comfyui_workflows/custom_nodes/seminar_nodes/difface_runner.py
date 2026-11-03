# DifFace（Yue & Loy, TPAMI 2024）を別プロセスで実行する補助スクリプト。
# difface_node.py から呼ばれる。DifFace のリポジトリには utils / models など
# ComfyUI と同名のパッケージがあるため、ComfyUI 本体とは別プロセスで動かす。
#
#   python difface_runner.py <DifFaceフォルダ> <入力フォルダ> <出力フォルダ> <start_t> <eta> <seed> <fp16:0/1>
import os
import sys
from pathlib import Path

difface_dir, in_dir, out_dir = (str(Path(p).resolve()) for p in sys.argv[1:4])
start_t, eta, seed, fp16 = int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]), sys.argv[7] == "1"

os.chdir(difface_dir)                    # 設定ファイル・重みは DifFace フォルダからの相対パス
sys.path.insert(0, difface_dir)

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from basicsr.utils.download_util import load_file_from_url
from sampler import DifFaceSampler

cfg = OmegaConf.load("configs/sample/iddpm_ffhq512_swinir.yaml")
cfg.seed = seed
cfg.aligned = True
cfg.diffusion.params.timestep_respacing = "ddim250"

# 重みが無ければ公式リリースから自動ダウンロード（合計 約 0.6 GB）
URL = "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/"
for ckpt, name in [(cfg.model_ir.ckpt_path, "swinir_restoration512_L1.pth"),
                   (cfg.model.ckpt_path, "iddpm_ffhq512_ema500000.pth")]:
    if not Path(ckpt).exists():
        load_file_from_url(url=URL + name, model_dir=str(Path(ckpt).parent), progress=True, file_name=name)

sampler = DifFaceSampler(cfg, im_size=cfg.model.params.image_size, use_fp16=fp16)

os.makedirs(out_dir, exist_ok=True)
for f in sorted(Path(in_dir).glob("*.png")):
    x = np.asarray(Image.open(f).convert("RGB"), dtype=np.float32) / 255.0
    y0 = torch.from_numpy(x).permute(2, 0, 1)[None].cuda()          # 1x3xHxW, [0,1]
    with torch.no_grad():
        out, _ = sampler.sample_func_ir_aligned(y0, start_timesteps=start_t, need_restoration=True, eta=eta)
    out = (out[0].permute(1, 2, 0).float().cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    Image.fromarray(out).save(Path(out_dir) / f.name)
    print("restored", f.name, flush=True)
