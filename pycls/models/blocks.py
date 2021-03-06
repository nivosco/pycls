#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Common model blocks."""

import numpy as np
import torch
import torch.nn as nn
from pycls.core.config import cfg
from torch.nn import Module


# ----------------------- Shortcuts for common torch.nn layers ----------------------- #


def conv2d(w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Helper for building a conv2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    s, p, g, b = stride, (k - 1) // 2, groups, bias
    return nn.Conv2d(w_in, w_out, k, stride=s, padding=p, groups=g, bias=b)


def norm2d(w_in):
    """Helper for building a norm2d layer."""
    return nn.BatchNorm2d(num_features=w_in, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)


def pool2d(_w_in, k, *, stride=1):
    """Helper for building a pool2d layer."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    return nn.MaxPool2d(k, stride=stride, padding=(k - 1) // 2)


def gap2d(_w_in):
    """Helper for building a gap2d layer."""
    return nn.AdaptiveAvgPool2d((1, 1))


def wap2d(_w_in):
    """Helper for building a wap2d layer."""
    return nn.AdaptiveAvgPool2d((None, 1))


def linear(w_in, w_out, *, bias=False):
    """Helper for building a linear layer."""
    return nn.Linear(w_in, w_out, bias=bias)


def activation():
    """Helper for building an activation layer."""
    activation_fun = cfg.MODEL.ACTIVATION_FUN.lower()
    if activation_fun == "relu":
        return nn.ReLU(inplace=cfg.MODEL.ACTIVATION_INPLACE)
    elif activation_fun == "silu" or activation_fun == "swish":
        try:
            return torch.nn.SiLU()
        except AttributeError:
            return SiLU()
    else:
        raise AssertionError("Unknown MODEL.ACTIVATION_FUN: " + activation_fun)


# --------------------------- Complexity (cx) calculations --------------------------- #


def conv2d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    if type(k) == tuple:
        flops += k[0] * k[1] * w_in  * h + 1
        params += k[0] * k[1] + 1
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}
    else:
        assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
        h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
        flops += k * k * w_in * w_out * h * w // groups + (w_out if bias else 0)
        params += k * k * w_in * w_out // groups + (w_out if bias else 0)
        acts += w_in * w * k
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def conv1d_cx(cx, w_in, w_out, k, *, stride=1, groups=1, bias=False):
    """Accumulates complexity of conv2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += k * w_in * w_out * h * w // groups + (w_out if bias else 0)
    params += k * w_in * w_out // groups + (w_out if bias else 0)
    acts += w_in * w * k
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}
    

def norm2d_cx(cx, w_in):
    """Accumulates complexity of norm2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    params += 2 * w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def pool2d_cx(cx, w_in, k, *, stride=1):
    """Accumulates complexity of pool2d into cx = (h, w, flops, params, acts)."""
    assert k % 2 == 1, "Only odd size kernels supported to avoid padding issues."
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    h, w = (h - 1) // stride + 1, (w - 1) // stride + 1
    acts += w_in * w * k
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


def wap2d_cx(cx, _w_in):
    """Accumulates complexity of wap2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    acts += _w_in * w
    return {"h": h, "w": 1, "flops": flops, "params": params, "acts": acts}


def gap2d_cx(cx, _w_in):
    """Accumulates complexity of gap2d into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    acts += _w_in * w * h
    return {"h": 1, "w": 1, "flops": flops, "params": params, "acts": acts}


def linear_cx(cx, w_in, w_out, *, bias=False):
    """Accumulates complexity of linear into cx = (h, w, flops, params, acts)."""
    h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
    flops += w_in * w_out + (w_out if bias else 0)
    params += w_in * w_out + (w_out if bias else 0)
    acts += w_in
    return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


# ---------------------------------- Shared blocks ----------------------------------- #


class SiLU(Module):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""

    # Note: will be part of Pytorch 1.7, at which point can remove this.

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def get_fca_weights(shape, freq=16):
    _, c, h, w = shape
    channels_per_freq = c // freq
    weights = np.ones((1,freq, h, w), np.float32)
    for i in range(1,freq):
        for wh in range(h):
            for ww in range(w):
                weights[0, i, wh, ww] = np.cos((np.pi * wh / h) * (wh + 0.5)) * np.cos((np.pi * ww / w) * (ww + 0.5))
    return torch.repeat_interleave(torch.from_numpy(weights), channels_per_freq, dim=1).cuda().float()


class FCA_SE(Module):
    """FcaNet Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(FCA_SE, self).__init__()
        self.pre_computed_weights = None
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.pre_computed_weights is None:
            self.pre_computed_weights = get_fca_weights([int(i) for i in x.shape])
        sq = torch.unsqueeze(torch.unsqueeze(torch.sum(x * self.pre_computed_weights, dim=[2,3]), axis=-1), axis=-1)
        ex = self.f_ex(sq)
        return x * ex

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class SE(Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = gap2d(w_in)
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = gap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class C_SE(Module):
    """Channel Squeeze-and-Excitation (cSE) block: 1x1, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(C_SE, self).__init__()
        self.f_ex = nn.Sequential(
            conv2d(w_in, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(x)

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = conv2d_cx(cx, w_in, 1, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class SE_GAP(Module):
    """Squeeze-and-Excitation without GAP (SE_GAP) block: 3x3, Act, 3x3, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE_GAP, self).__init__()
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 3, bias=True),
            activation(),
            conv2d(w_se, w_in, 3, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(x)

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = conv2d_cx(cx, w_in, w_se, 3, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 3, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class SE_GAP1(Module):
    """Squeeze-and-Excitation without GAP (SE_GAP) block: 3x3, Act, 3x3, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(SE_GAP1, self).__init__()
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_se, 1, bias=True),
            activation(),
            conv2d(w_se, w_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(x)

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class SE_GAP_DW(Module):
    """Squeeze-and-Excitation without GAP (SE_GAP) block: dw3x3, Act, dw3x3, Sigmoid."""

    def __init__(self, w_in):
        super(SE_GAP_DW, self).__init__()
        self.f_ex = nn.Sequential(
            conv2d(w_in, w_in, 3, groups=w_in, bias=True),
            activation(),
            conv2d(w_in, w_in, 3, groups=w_in, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(x)

    @staticmethod
    def complexity(cx, w_in):
        h, w = cx["h"], cx["w"]
        cx = conv2d_cx(cx, w_in, w_in, 3, groups=w_in, bias=True)
        cx = conv2d_cx(cx, w_in, w_in, 3, groups=w_in, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class EW_SE(Module):
    """Efficient Width Squeeze-and-Excitation (EW_SE) block: AvgPool, 3x1, Act, 3x1, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(EW_SE, self).__init__()
        self.avg_pool = None
        self.f_ex = nn.Sequential(
            nn.Conv2d(w_in, w_se, 1, stride=1, padding=0, bias=True),
            activation(),
            nn.Conv2d(w_se, w_in, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.avg_pool is None:
            w = int(x.shape[-1])
            c = int(x.shape[1])
            self.avg_pool = nn.AvgPool2d((7, w), stride=(7, w), padding=0, ceil_mode=True)
        sq = self.avg_pool(x)
        ex = self.f_ex(sq)
        return x * torch.repeat_interleave(ex, 7, dim=-2)[:,:,:int(x.shape[-2]),:]

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = wap2d_cx(cx, w_in)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv2d_cx(cx, w_in, w_se, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx



class W_SE(Module):
    """Width Squeeze-and-Excitation (W_SE) block: AvgPool, 3x1, Act, 3x1, Sigmoid."""

    def __init__(self, w_in, w_se, add=False):
        super(W_SE, self).__init__()
        self.avg_pool = wap2d(w_in)
        self.add = add
        if self.add:
            self.f_ex = nn.Sequential(
                nn.Conv1d(w_in, w_se, 3, stride=1, padding=1, bias=True),
                activation(),
                nn.Conv1d(w_se, w_in, 3, stride=1, padding=1, bias=True),
            )
        else:
            self.f_ex = nn.Sequential(
                nn.Conv1d(w_in, w_se, 3, stride=1, padding=1, bias=True),
                activation(),
                nn.Conv1d(w_se, w_in, 3, stride=1, padding=1, bias=True),
                nn.Sigmoid(),
            )

    def forward(self, x):
        sq = torch.squeeze(self.avg_pool(x))
        ex = torch.unsqueeze(self.f_ex(sq), -1)
        if self.add:
            return x + ex
        else:
            return x * ex

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = wap2d_cx(cx, w_in)
        cx = conv1d_cx(cx, w_in, w_se, 3, bias=True)
        cx = conv1d_cx(cx, w_se, w_in, 3, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class W1_SE(Module):
    """Width Squeeze-and-Excitation (W_SE) block: AvgPool, 3x1, Act, 3x1, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(W1_SE, self).__init__()
        self.avg_pool = wap2d(w_in)
        self.f_ex = nn.Sequential(
            nn.Conv1d(w_in, w_se, 1, stride=1, padding=0, bias=True),
            activation(),
            nn.Conv1d(w_se, w_in, 1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * torch.unsqueeze(self.f_ex(torch.squeeze(self.avg_pool(x))), -1)

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = wap2d_cx(cx, w_in)
        cx = conv1d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv1d_cx(cx, w_se, w_in, 1, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class W13_SE(Module):
    """Width Squeeze-and-Excitation (W_SE) block: AvgPool, 1x1, Act, 3x1, Sigmoid."""

    def __init__(self, w_in, w_se):
        super(W13_SE, self).__init__()
        self.avg_pool = wap2d(w_in)
        self.f_ex = nn.Sequential(
            nn.Conv1d(w_in, w_se, 1, stride=1, padding=0, bias=True),
            activation(),
            nn.Conv1d(w_se, w_in, 3, stride=1, padding=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * torch.unsqueeze(self.f_ex(torch.squeeze(self.avg_pool(x))), -1)

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx = wap2d_cx(cx, w_in)
        cx = conv1d_cx(cx, w_in, w_se, 1, bias=True)
        cx = conv1d_cx(cx, w_se, w_in, 3, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


# ---------------------------------- Miscellaneous ----------------------------------- #


def adjust_block_compatibility(ws, bs, gs):
    """Adjusts the compatibility of widths, bottlenecks, and groups."""
    assert len(ws) == len(bs) == len(gs)
    assert all(w > 0 and b > 0 and g > 0 for w, b, g in zip(ws, bs, gs))
    vs = [int(max(1, w * b)) for w, b in zip(ws, bs)]
    gs = [int(min(g, v)) for g, v in zip(gs, vs)]
    ms = [np.lcm(g, b) if b > 1 else g for g, b in zip(gs, bs)]
    vs = [max(m, int(round(v / m) * m)) for v, m in zip(vs, ms)]
    ws = [int(v / b) for v, b in zip(vs, bs)]
    assert all(w * b % g == 0 for w, b, g in zip(ws, bs, gs))
    return ws, bs, gs


def init_weights(m):
    """Performs ResNet-style weight initialization."""
    if isinstance(m, nn.Conv2d):
        # Note that there is no bias due to BN
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
    elif isinstance(m, nn.BatchNorm2d):
        zero_init_gamma = cfg.BN.ZERO_INIT_FINAL_GAMMA
        zero_init_gamma = hasattr(m, "final_bn") and m.final_bn and zero_init_gamma
        m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(mean=0.0, std=0.01)
        m.bias.data.zero_()


def drop_connect(x, drop_ratio):
    """Drop connect (adapted from DARTS)."""
    keep_ratio = 1.0 - drop_ratio
    mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
    mask.bernoulli_(keep_ratio)
    x.div_(keep_ratio)
    x.mul_(mask)
    return x
