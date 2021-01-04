import torch
from pycls.core.config import cfg
from pycls.models.model_zoo import regnety

size = cfg.TRAIN.IM_SIZE
models = ["800MF_W25_SE"]

for m in models:
    print("Model {}".format(m))
    model = regnety(m, pretrained=False)
    x = torch.randn(8, 3, 224, 224, requires_grad=True)
    out = model(x)
    torch.onnx.export(model, x, m + ".onnx", opset_version=11)

