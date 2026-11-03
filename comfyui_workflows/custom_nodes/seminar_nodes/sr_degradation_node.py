import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class SRDegradationNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05}),
                "multiple_of": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1}),
                "noise_sigma": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.5}),
                # ComfyUI標準のシード発生器と完全に一致する正しい定義
                # これによりフロントエンドが引数の位置を誤認するのを防ぎます
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "enabled": (["true", "false"],),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "target_width", "target_height")
    FUNCTION = "degrade"
    CATEGORY = "image"

    # バックエンドのバリデーションを無効化し、"randomize" などの文字列通過を許容する
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def degrade(self, image, scale_factor=0.25, multiple_of=16, noise_sigma=10.0, noise_seed=0, enabled="true"):
        B, H, W, C = image.shape
        device = image.device

        # シード値に "randomize" などの非数値オブジェクトが混入した場合の安全なフォールバック
        if not isinstance(noise_seed, (int, float)):
            try:
                noise_seed = int(float(str(noise_seed)))
            except (ValueError, TypeError):
                # 完全な文字列だった場合は、その場で安全な乱数を生成
                noise_seed = int(torch.randint(0, 2147483647, (1,)).item())

        # 確実に32bit符号付き整数の範囲に丸める (PyTorch Generatorの制約回避)
        noise_seed = int(noise_seed) % 2147483647

        if enabled == "false":
            safe_W = max(multiple_of, round(W / multiple_of) * multiple_of)
            safe_H = max(multiple_of, round(H / multiple_of) * multiple_of)
            return (image, safe_W * 4, safe_H * 4)

        # 1. 縮小サイズの計算
        raw_H = H * scale_factor
        raw_W = W * scale_factor

        # 2. 指定された倍数への丸め処理
        new_W = max(multiple_of, round(raw_W / multiple_of) * multiple_of)
        new_H = max(multiple_of, round(raw_H / multiple_of) * multiple_of)

        # チャンネルファーストへ変換 (B, C, H, W)
        x = image.permute(0, 3, 1, 2).contiguous()
        
        # アンチエイリアス縮小
        x_down = TF.resize(
            x, 
            size=[new_H, new_W], 
            interpolation=TF.InterpolationMode.BICUBIC, 
            antialias=True
        )
        x_down = x_down.clamp(0.0, 1.0)

        # ガウシアンノイズ付加
        if noise_sigma > 0.0:
            generator = torch.Generator(device=device).manual_seed(noise_seed)
            noise = torch.randn(x_down.shape, generator=generator,
                                device=device, dtype=x_down.dtype) * (noise_sigma / 255.0)
            x_down = (x_down + noise).clamp(0.0, 1.0)

        result = x_down.permute(0, 2, 3, 1).contiguous()
        
        # 4倍の出力ターゲットサイズ
        target_width = new_W * 4
        target_height = new_H * 4

        return (result, target_width, target_height)


NODE_CLASS_MAPPINGS = {
    "SRDegradationNode": SRDegradationNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SRDegradationNode": "SR Degradation (Downscale + Gaussian Noise)",
}