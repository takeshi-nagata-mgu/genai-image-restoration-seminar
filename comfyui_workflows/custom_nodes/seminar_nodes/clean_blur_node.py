import torch
import torch.nn.functional as F
import math

class CleanGaussianBlurNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "kernel_size": ("INT", {"default": 5, "min": 3, "max": 25, "step": 2}),
                "sigma": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_blur"
    CATEGORY = "image/filters"

    def apply_blur(self, image, kernel_size, sigma):
        # 偶数のカーネルサイズを奇数に強制補正（数学的整合性のため）
        if kernel_size % 2 == 0:
            kernel_size += 1

        B, H, W, C = image.shape
        device = image.device
        dtype = image.dtype

        # チャンネルファーストに変換 (B, C, H, W)
        x = image.permute(0, 3, 1, 2).contiguous()

        # 1次元のガウスカーネルを作成
        grid = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-grid**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # 2次元のガウスカーネルに拡張 (kernel_size, kernel_size)
        kernel_2d = kernel_1d.view(-1, 1) * kernel_1d.view(1, -1)
        
        # Conv2Dの重み形状に変形 (C, 1, kernel_size, kernel_size)
        blur_kernel = kernel_2d.expand(C, 1, kernel_size, kernel_size).contiguous()

        # パディング幅の計算
        padding = kernel_size // 2

        # OpenGLを一切介さず、純粋なPyTorchの数式（畳み込み）のみでボカシを計算
        x_blurred = F.conv2d(x, blur_kernel, groups=C, padding=padding)
        x_blurred = x_blurred.clamp(0.0, 1.0)

        # 元のComfyUI画像形状 (B, H, W, C) に戻して出力
        result = x_blurred.permute(0, 2, 3, 1).contiguous()
        return (result,)


NODE_CLASS_MAPPINGS = {
    "CleanGaussianBlurNode": CleanGaussianBlurNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CleanGaussianBlurNode": "Clean Gaussian Blur (PyTorch-Native)",
}