from pycls.core.config import cfg
from pycls.models.model_zoo import regnety

size = cfg.TRAIN.IM_SIZE
cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
models = ["800MF", "800MF_NO_SE", "800MF_C_SE", "800MF_W_SE", "800MF_W1_SE", "800MF_SE_GAP", "800MF_SE_GAP1", "800MF_W25_SE"]

for m in models:
    print("Model {}".format(m))
    model = regnety(m, pretrained=False)
    print(model.complexity(cx))
