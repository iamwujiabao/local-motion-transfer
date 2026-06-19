# third_party/

Vendor the MimicMotion repository here so that:

    third_party/MimicMotion/inference.py

exists, and place its weights under:

    third_party/MimicMotion/models/
      ├── DWPose/{yolox_l.onnx, dw-ll_ucoco_384.onnx}
      └── MimicMotion_1-1.pth

See the top-level README → Setup. `config.py:MIMICMOTION_REPO` points here by
default; change it if you keep the repo elsewhere. MimicMotion is licensed
separately — see its own LICENSE file.
