"""標準ノードだけで動く配布用ワークフロー（API 形式）を生成する。
ComfyUI の画面にドラッグ＆ドロップすると、そのまま読み込まれます。
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

NEG = "blurry, low quality, watermark, text, deformed"

upscale = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}, "_meta": {"title": "入力画像"}},
    "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x4plus.pth"}},
    "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
    "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "upscale_realesrgan"}},
}

def sd15_inpaint_core(image_ref, mask_ref, prompt, prefix, composite=True):
    wf = {
        "10": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd-v1-5-inpainting.ckpt"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": prompt}, "_meta": {"title": "プロンプト（生成したい内容）"}},
        "12": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 1], "text": NEG}, "_meta": {"title": "ネガティブプロンプト（消したい物も書く）"}},
        "13": {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": image_ref, "vae": ["10", 2], "mask": mask_ref, "grow_mask_by": 8}},
        "14": {"class_type": "KSampler", "inputs": {"model": ["10", 0], "positive": ["11", 0], "negative": ["12", 0],
               "latent_image": ["13", 0], "seed": 0, "steps": 30, "cfg": 7.0, "sampler_name": "dpmpp_2m",
               "scheduler": "karras", "denoise": 1.0}},
        "15": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["10", 2]}},
        "16": {"class_type": "GrowMask", "inputs": {"mask": mask_ref, "expand": 6, "tapered_corners": True}},
        # マスク外は元画像のまま残す（VAE による全体の微妙な変化を防ぐ）
        "17": {"class_type": "ImageCompositeMasked", "inputs": {"destination": image_ref, "source": ["15", 0],
               "x": 0, "y": 0, "resize_source": False, "mask": ["16", 0]}},
        "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": prefix}},
    }
    return wf

# Inpainting：Load Image で画像を選び、右クリック →「Open in MaskEditor」で消したい部分を塗る
inpaint = {"1": {"class_type": "LoadImage", "inputs": {"image": "input_masked.png"},
                 "_meta": {"title": "入力画像（MaskEditor で消したい部分を塗る）"}}}
inpaint.update(sd15_inpaint_core(["1", 0], ["1", 1], "plain gray wall, empty background, photo", "inpaint_sd15"))

# Outpainting：上下左右に何ピクセル広げるかを指定
outpaint = {"1": {"class_type": "LoadImage", "inputs": {"image": "input.png"}, "_meta": {"title": "入力画像"}},
            "2": {"class_type": "ImagePadForOutpaint", "inputs": {"image": ["1", 0], "left": 128, "top": 0,
                  "right": 128, "bottom": 64, "feathering": 40}, "_meta": {"title": "広げる幅（px）"}}}
outpaint.update(sd15_inpaint_core(["2", 0], ["2", 1], "a cup of coffee on a wooden table, cafe, photo", "outpaint_sd15"))

for name, wf in [("upscale_realesrgan", upscale), ("inpaint_sd15", inpaint), ("outpaint_sd15", outpaint)]:
    with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
        json.dump(wf, f, ensure_ascii=False, indent=1)
    print("wrote", name + ".json")
