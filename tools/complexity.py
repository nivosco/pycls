from pycls.core.config import cfg
from pycls.models.model_zoo import regnety, effnet

size = cfg.TRAIN.IM_SIZE

#models = ["800MF", "800MF_NO_SE", "800MF_C_SE", "800MF_W_SE", "800MF_W1_SE", "800MF_SE_GAP", "800MF_SE_GAP1", "800MF_SE_GAP_DW", "800MF_W25_SE"]

models = ["3.2GF", "3.2GF_W25_SE", "1.6GF", "1.6GF_W25_SE", "800MF", "800MF_W25_SE", "600MF", "600MF_W25_SE", "400MF", "400MF_W25_SE", "200MF", "200MF_W25_SE"]

for m in models:
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    print("Model {}".format(m))
    model = regnety(m, pretrained=False)
    print(model.complexity(cx))

models = ["B0", "B0_WSE", "B1", "B1_WSE", "B2", "B2_WSE", "B3", "B3_WSE"]

for m in models:
    cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
    print("Model {}".format(m))
    model = effnet(m, pretrained=False)
    print(model.complexity(cx))
