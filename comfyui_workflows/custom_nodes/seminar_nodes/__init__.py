# セミナー配布用カスタムノード
#   SRDegradationNode     : 縮小（倍率・16の倍数に丸め）＋ガウスノイズ
#   CleanGaussianBlurNode : PyTorch の畳み込みだけで行うガウスぼかし
#   DifFaceRestoreNode    : DifFace による顔画像復元（DifFace のリポジトリが別途必要）
# ComfyUI/custom_nodes/ にこのフォルダごと置いて ComfyUI を再起動してください。
from .sr_degradation_node import NODE_CLASS_MAPPINGS as _m1, NODE_DISPLAY_NAME_MAPPINGS as _d1
from .clean_blur_node import NODE_CLASS_MAPPINGS as _m2, NODE_DISPLAY_NAME_MAPPINGS as _d2
from .difface_node import NODE_CLASS_MAPPINGS as _m3, NODE_DISPLAY_NAME_MAPPINGS as _d3

NODE_CLASS_MAPPINGS = {**_m1, **_m2, **_m3}
NODE_DISPLAY_NAME_MAPPINGS = {**_d1, **_d2, **_d3}
