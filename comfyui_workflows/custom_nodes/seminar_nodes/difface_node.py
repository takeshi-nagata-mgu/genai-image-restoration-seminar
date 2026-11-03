# DifFace 顔画像復元ノード（セミナー配布用）
#   DifFace: Z. Yue and C. C. Loy, "DifFace: Blind Face Restoration with Diffused Error Contraction", TPAMI 2024
#   https://github.com/zsyOAOA/DifFace （コード・重みのライセンスは同リポジトリを参照）
#
# 準備：DifFace のリポジトリを ComfyUI フォルダの中に取得する（ComfyUI/DifFace）
#   cd ComfyUI
#   git clone https://github.com/zsyOAOA/DifFace  （ZIP をダウンロードして展開してもよい）
# 別の場所に置いた場合は difface_dir に絶対パスを指定する。重みは初回実行時に自動ダウンロード。
# 入力は「顔が中央に写った 512x512 程度の正方形画像」（位置合わせ済みの顔）を想定。
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

try:
    import comfy.model_management as mm
except ImportError:  # ComfyUI 外でテストするとき
    mm = None

RUNNER = Path(__file__).with_name("difface_runner.py")
COMFY_ROOT = Path(__file__).resolve().parents[2]  # ComfyUI/custom_nodes/seminar_nodes → ComfyUI


class DifFaceRestoreNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "difface_dir": ("STRING", {"default": "DifFace",
                                           "tooltip": "DifFace のフォルダ。相対パスは ComfyUI フォルダ基準"}),
                "start_timesteps": ("INT", {"default": 100, "min": 1, "max": 249,
                                            "tooltip": "どこまでノイズ化してから逆拡散するか（大きいほど生成の自由度が高い）"}),
                "eta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1, "control_after_generate": True}),
                "use_fp16": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "restore"
    CATEGORY = "seminar/restoration"

    def restore(self, image, difface_dir, start_timesteps, eta, seed, use_fp16):
        ddir = Path(difface_dir)
        if not ddir.is_absolute():
            ddir = COMFY_ROOT / ddir
        if not (ddir / "sampler.py").exists():
            raise FileNotFoundError(
                f"DifFace が見つかりません: {ddir}\n"
                "git clone https://github.com/zsyOAOA/DifFace を実行し、そのフォルダを difface_dir に指定してください。")
        if mm is not None:  # 別プロセスで GPU を使うので、ComfyUI 側のモデルを一旦解放
            mm.unload_all_models()
            mm.soft_empty_cache()

        work = Path(tempfile.mkdtemp(prefix="difface_"))
        try:
            (work / "in").mkdir()
            for i, im in enumerate(image):  # image: B x H x W x C, [0,1]
                arr = (im.cpu().numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
                Image.fromarray(arr).save(work / "in" / f"{i:04d}.png")
            cmd = [sys.executable, str(RUNNER), str(ddir), str(work / "in"), str(work / "out"),
                   str(start_timesteps), str(eta), str(seed), "1" if use_fp16 else "0"]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                raise RuntimeError("DifFace の実行に失敗しました:\n" + r.stdout[-2000:] + r.stderr[-3000:])
            outs = [np.asarray(Image.open(work / "out" / f"{i:04d}.png").convert("RGB"), dtype=np.float32) / 255.0
                    for i in range(image.shape[0])]
        finally:
            shutil.rmtree(work, ignore_errors=True)
        return (torch.from_numpy(np.stack(outs)),)


NODE_CLASS_MAPPINGS = {"DifFaceRestoreNode": DifFaceRestoreNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DifFaceRestoreNode": "DifFace 顔画像復元 (seminar)"}
