# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
from __future__ import annotations
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Union



__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "Concat2",
    "ADD",
    "SimAM",
    "ShuffleAttention",
    "GAM_Attention",
    "CBAM2",
    "CoordAtt",
    "ECA",
    "SEAttention",
    "GLCBAM",
    "S2Attention",
    "SKAttention",
    "GLF",
    "NAM",
    "GCBAM",
    "SACBAM",
    "MdC2f",
    "C2f_Invo",
    "IN",
    "Multiin",
    "EFBlock",
    "LiteAlign2",
    "LiteRefine2",
    "WeightedAdd",
    "LiteRefine2Ctx",
    "IRPromptRGB",
    "IRPriorWeightedAdd",
    "P3LocalVerify",
    "DetectionACMLite",
    "ResidualWeightedAdd",
    "ResidualWeightedAddStable",
    "LiteRefine2Mid",
    "ERF",
    "TSCI",
    "TSCF",
    "IACMFusion",
    "RawCueAddC",
    "RawCueMap",
    "RawCueAddD",
    "TSCIv2RawCueFusion",
    "_TSCIv3MatchedWindowAttention",
    "TSCISS2DContext",
    "TSCIv3RawCueSS2DFusion",
    "TSCIv4SharedWindowMambaFusion",
    "DSSF_SS2D",
    "CMSSMahalanobisWindowInteraction",
    "LASCIModule",
    "LASCIDSSF",
    "LASCICECC",
    "LASCIModulev2",
    "LASCIModulev3",
    "IRGuidedSelectiveOffset",
    "ProgressiveSCI", 
    "SCISelect", 
    "SCIPairFuse",
    "FRCSFModule",
    "FreqMoERouterSpecific",
    "FreqMoEFullIndependent",
    "FreqMoEFullIndependentV2",
    "FreqMoEFullIndependentV2Residual",
    "FreqCoupledWaveletFusion",
)


class P3LocalVerify(nn.Module):
    """
    P3 小头局部验证块
    目标：
    1) 只补局部纹理/边缘/轮廓
    2) 不做强上下文传播
    3) 用零初始化残差，训练初期尽量贴近当前 strongest baseline
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        c = c1 if c2 is None else c2
        assert c1 == c, f"P3LocalVerify expects same in/out channels, got {c1} vs {c}"

        hidden = max(c // 2, 32)
        gate_hidden = max(c // 4, 16)

        # 先压缩到 hidden，降低计算量
        self.pre = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 两个局部分支：3x3 偏细节，5x5 偏稍大局部轮廓
        self.dw3 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 融合回原通道
        self.mix = nn.Sequential(
            nn.Conv2d(hidden * 2, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # 通道门控：决定哪些通道值得被增强
        self.chan_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, gate_hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(gate_hidden, c, 1, bias=True),
            nn.Sigmoid()
        )

        # 空间门控：决定哪些位置值得被增强
        self.spa_gate = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 零初始化残差强度，尽量不破坏现有 baseline
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h = self.pre(x)

        f3 = self.dw3(h)
        f5 = self.dw5(h)

        delta = self.mix(torch.cat([f3, f5], dim=1))

        avg_map = torch.mean(delta, dim=1, keepdim=True)
        max_map = torch.max(delta, dim=1, keepdim=True)[0]

        sg = self.spa_gate(torch.cat([avg_map, max_map], dim=1))   # [B,1,H,W]
        cg = self.chan_gate(delta)                                 # [B,C,1,1]

        out = x + self.gamma * cg * sg * delta
        return out

class IRPriorWeightedAdd(nn.Module):
    """
    IR-guided fusion prior for WeightedAdd
    输入:  (rgb, ir)
    输出:  fused tensor

    设计目标：
    1) 不直接改写 RGB / IR 内容
    2) 只让 IR 生成空间先验，去偏置融合权重
    3) 初始尽量接近原始 WeightedAdd，降低破坏 strongest baseline 的风险
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)

        # 原始 WeightedAdd 的全局模态打分
        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1, bias=True)   # -> [B,2,1,1]
        )

        # IR -> spatial prior
        # 只生成一个 [B,1,H,W] 的空间先验图，表示“这里更可能该信 IR”
        self.ir_prior = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # 强度系数，初始为 0，保证训练初期退化为原始 WeightedAdd
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]
        diff = torch.abs(rgb - ir)

        # 全局模态分数: [B,2,1,1]
        s = self.score(torch.cat([rgb, ir, diff], dim=1))

        # IR 空间先验: [B,1,H,W]
        p = self.ir_prior(ir)

        # 用 IR 先验去偏置融合权重：
        # prior 大的位置，更偏向 IR；prior 小的位置，更偏向 RGB
        bias = torch.cat([-p, p], dim=1)  # [B,2,H,W]

        # 利用广播把 [B,2,1,1] 扩展到空间维
        logits = s + self.alpha * bias
        w = torch.softmax(logits, dim=1)

        wr = w[:, 0:1]   # [B,1,H,W]
        wi = w[:, 1:2]   # [B,1,H,W]

        out = wr * rgb + wi * ir
        return out

class IRPromptRGB(nn.Module):#没用
    """
    Infrared-guided RGB Prompt Module

    目标：
    1) 利用 IR 生成结构/显著性提示
    2) 只增强 RGB，不强行改动 IR
    3) 保持轻量、稳定，便于直接插入你当前 strongest baseline

    输入:  (rgb, ir)
    输出:  (rgb_prompted, ir)

    设计动机：
    - IR 更擅长提供目标轮廓/热显著区域
    - RGB 更擅长提供细粒度语义和纹理
    - 因此先让 IR 提示 RGB “该看哪里”，再进入后续对齐-细化-融合链路
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)

        # 先对 IR 做一个轻量投影，稳定提示生成
        self.ir_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # IR -> spatial prompt: [B,1,H,W]
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # IR -> channel prompt: [B,C,1,1]
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c, 1, bias=True),
            nn.Sigmoid()
        )

        # RGB 轻量变换支路
        self.rgb_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # 额外结合模态差异，避免只靠 IR 一路做提示
        self.diff_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # 输出校正，避免提示过硬
        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # 残差强度初始化为0，保证开局接近当前基线
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        ir_feat = self.ir_proj(ir)
        rgb_feat = self.rgb_proj(rgb)
        diff = self.diff_proj(torch.abs(rgb - ir))

        ms = self.spatial_gate(ir_feat)      # [B,1,H,W]
        mc = self.channel_gate(ir_feat)      # [B,C,1,1]

        # IR 提示 + 模态差异辅助
        prompt = ms * mc * (rgb_feat + diff)
        prompt = self.out_proj(prompt)

        rgb_out = rgb + self.alpha * prompt
        ir_out = ir   # 第一版先不改 IR，保持“IR 引导 RGB”的非对称设计

        return (rgb_out, ir_out)

class SelectiveScan2D(nn.Module):
    """
    轻量版四方向 selective state-space context branch
    适合先在 P5 上验证“上下文补语义”是否有效
    """
    def __init__(self, c):
        super().__init__()
        self.in_proj = nn.Conv2d(c, c, 1, bias=False)
        self.alpha_proj = nn.Conv2d(c, c, 1, bias=True)

        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

    def scan_lr(self, x, alpha):
        # x, alpha: [B, C, H, W]
        B, C, H, W = x.shape
        state = torch.zeros(B, C, H, device=x.device, dtype=x.dtype)
        outs = []
        for j in range(W):
            a = alpha[:, :, :, j]
            u = x[:, :, :, j]
            state = a * state + (1.0 - a) * u
            outs.append(state.unsqueeze(-1))
        return torch.cat(outs, dim=-1)

    def scan_rl(self, x, alpha):
        B, C, H, W = x.shape
        state = torch.zeros(B, C, H, device=x.device, dtype=x.dtype)
        outs = [None] * W
        for j in range(W - 1, -1, -1):
            a = alpha[:, :, :, j]
            u = x[:, :, :, j]
            state = a * state + (1.0 - a) * u
            outs[j] = state.unsqueeze(-1)
        return torch.cat(outs, dim=-1)

    def scan_tb(self, x, alpha):
        B, C, H, W = x.shape
        state = torch.zeros(B, C, W, device=x.device, dtype=x.dtype)
        outs = []
        for i in range(H):
            a = alpha[:, :, i, :]
            u = x[:, :, i, :]
            state = a * state + (1.0 - a) * u
            outs.append(state.unsqueeze(-2))
        return torch.cat(outs, dim=-2)

    def scan_bt(self, x, alpha):
        B, C, H, W = x.shape
        state = torch.zeros(B, C, W, device=x.device, dtype=x.dtype)
        outs = [None] * H
        for i in range(H - 1, -1, -1):
            a = alpha[:, :, i, :]
            u = x[:, :, i, :]
            state = a * state + (1.0 - a) * u
            outs[i] = state.unsqueeze(-2)
        return torch.cat(outs, dim=-2)

    def forward(self, x):
        u = self.in_proj(x)
        alpha = torch.sigmoid(self.alpha_proj(x))  # 输入相关 gate

        lr = self.scan_lr(u, alpha)
        rl = self.scan_rl(u, alpha)
        tb = self.scan_tb(u, alpha)
        bt = self.scan_bt(u, alpha)

        y = (lr + rl + tb + bt) * 0.25
        y = self.out_proj(y)
        return y


class LiteRefine2Ctx(nn.Module):
    """
    上下文增强版 LiteRefine2
    目标：
    1) 保留原 LiteRefine2 的局部细化能力
    2) 引入四方向 selective-scan 上下文补语义
    3) 用 gate 控制什么时候启用上下文，减少背景噪声干扰

    输入:  (rgb, ir)
    输出:  (rgb_refined, ir_refined)
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 2, 32)

        # 双流拼接后先压到 hidden
        self.pre = nn.Sequential(
            nn.Conv2d(c * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 局部分支：保留原版局部建模能力
        self.local_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=max(hidden // 16, 1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 上下文分支：四方向 selective scan
        self.context_branch = SelectiveScan2D(hidden)

        # 不确定性 / 置信门控
        # 输入 local, context, |local-context|
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.Sigmoid()
        )

        # 两支路融合
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 输出两路更新量
        self.out_proj = nn.Conv2d(hidden, c * 2, 1, bias=True)

        # 残差强度，初始化为0，保证开局贴近你当前有效基线
        self.beta_r = nn.Parameter(torch.zeros(1))
        self.beta_i = nn.Parameter(torch.zeros(1))
        self.gamma_ctx = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        h = self.pre(torch.cat([rgb, ir], dim=1))

        local_feat = self.local_branch(h)
        ctx_feat = self.context_branch(h)

        gate = self.ctx_gate(torch.cat([local_feat, ctx_feat, torch.abs(local_feat - ctx_feat)], dim=1))

        # 只在 gate 认为有必要时引入上下文
        ctx_enhanced = local_feat + self.gamma_ctx * gate * ctx_feat

        fused = self.fuse(torch.cat([local_feat, ctx_enhanced], dim=1))
        update = self.out_proj(fused)

        dr, di = torch.chunk(update, 2, dim=1)

        rgb_out = rgb + self.beta_r * dr
        ir_out = ir + self.beta_i * di

        return (rgb_out, ir_out)

class LiteAlign2(nn.Module):
    """
    轻量双模态软对齐
    输入: [rgb, ir]
    输出: (rgb_aligned, ir_aligned)
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )
        self.ir_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU()
        )

        # 用差异信息引导两路靠近
        self.mix = nn.Sequential(
            nn.Conv2d(c * 3, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(),
        )

        # 生成两路门控
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, c * 2, 1, bias=True),
            nn.Sigmoid()
        )

        # 用 0 初始化，保证一开始接近原 baseline
        self.gamma_r = nn.Parameter(torch.zeros(1))
        self.gamma_i = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        rgb = self.rgb_proj(rgb)
        ir = self.ir_proj(ir)

        diff = torch.abs(rgb - ir)
        fusion = torch.cat([rgb, ir, diff], dim=1)

        gates = self.gate(fusion)
        g_r, g_i = torch.chunk(gates, 2, dim=1)

        delta = self.mix(fusion)

        rgb_out = rgb + self.gamma_r * g_i * delta
        ir_out = ir + self.gamma_i * g_r * delta

        return (rgb_out, ir_out)


class LiteRefine2(nn.Module):
    """
    模仿 Add2:
    不是直接融合两路，而是先分别更新两路
    输入: (rgb, ir)
    输出: (rgb_refined, ir_refined)
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 2, 32)

        self.refine = nn.Sequential(
            nn.Conv2d(c * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=max(hidden // 16, 1), bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, c * 2, 1, bias=True)
        )

        self.beta_r = nn.Parameter(torch.zeros(1))
        self.beta_i = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        update = self.refine(torch.cat([rgb, ir], dim=1))
        dr, di = torch.chunk(update, 2, dim=1)

        rgb_out = rgb + self.beta_r * dr
        ir_out = ir + self.beta_i * di

        return (rgb_out, ir_out)

class LiteRefine2Mid(nn.Module):
    """
    P4 专用中尺度 refine 模块
    目标：
    1) 保留 LiteRefine2 的局部细化能力
    2) 在 P4 引入中尺度结构/轮廓建模
    3) 不做 full context scan，避免与 P5 的 LiteRefine2Ctx 重叠
    4) 用 gate 决定什么时候引入中尺度信息

    输入:  (rgb, ir)
    输出:  (rgb_refined, ir_refined)
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 2, 32)
        groups = max(hidden // 16, 1)

        # 双流拼接后先压到 hidden
        self.pre = nn.Sequential(
            nn.Conv2d(c * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 1) local branch：保留原 LiteRefine2 的局部建模
        self.local_branch = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 2) mid-shape branch：偏长轴/轮廓，适合 truck / freight_car / van
        self.mid_shape = nn.Sequential(
            nn.Conv2d(hidden, hidden, (1, 7), padding=(0, 3), groups=groups, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, (7, 1), padding=(3, 0), groups=groups, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 3) mid-region branch：补中尺度区域关系，不走 full global
        self.mid_region = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=groups, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 4) 先融合两个 mid 分支
        self.mid_fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 5) gate：决定 local 和 mid 哪个更该参与
        # 输入 local, mid, |local-mid|
        self.mid_gate = nn.Sequential(
            nn.Conv2d(hidden * 3, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 1, bias=True),
            nn.Sigmoid()
        )

        # 6) local + gated mid 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 2, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU()
        )

        # 7) 输出两路更新量
        self.out_proj = nn.Conv2d(hidden, c * 2, 1, bias=True)

        # 8) 零初始化残差强度，开局贴近 baseline
        self.beta_r = nn.Parameter(torch.zeros(1))
        self.beta_i = nn.Parameter(torch.zeros(1))
        self.gamma_mid = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rgb, ir = x[0], x[1]

        h = self.pre(torch.cat([rgb, ir], dim=1))

        local_feat = self.local_branch(h)

        mid_shape = self.mid_shape(h)
        mid_region = self.mid_region(h)
        mid_feat = self.mid_fuse(torch.cat([mid_shape, mid_region], dim=1))

        gate = self.mid_gate(torch.cat([
            local_feat,
            mid_feat,
            torch.abs(local_feat - mid_feat)
        ], dim=1))

        # 只在 gate 认为有必要时引入中尺度信息
        mid_enhanced = local_feat + self.gamma_mid * gate * mid_feat

        fused = self.fuse(torch.cat([local_feat, mid_enhanced], dim=1))
        update = self.out_proj(fused)

        dr, di = torch.chunk(update, 2, dim=1)

        rgb_out = rgb + self.beta_r * dr
        ir_out = ir + self.beta_i * di

        return (rgb_out, ir_out)


# class ResidualWeightedAdd(nn.Module):
#     """
#     比 WeightedAdd 多一个“局部修正残差”：
#     - 保留全局模态打分的稳定性
#     - 只在不确定/冲突区域启用局部修正
#     - 输入: (rgb, ir)
#     - 输出: fused
#     """
#     def __init__(self, c):
#         super().__init__()
#         hidden = max(c // 4, 16)
#         u_hidden = max(hidden // 2, 8)

#         # 1) 全局先验：基本继承原 WeightedAdd 的保守思路
#         self.global_score = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(c * 4, hidden, 1, bias=False),
#             nn.SiLU(),
#             nn.Conv2d(hidden, 2, 1, bias=True)
#         )

#         # 2) 局部修正：轻量 spatial residual
#         self.local_score = nn.Sequential(
#             nn.Conv2d(c * 4, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(),
#             nn.Conv2d(
#                 hidden, hidden, 3, padding=1,
#                 groups=max(hidden // 16, 1), bias=False
#             ),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden, 2, 1, bias=True)
#         )

#         # 3) 不确定性门控：决定哪些位置需要 local correction
#         self.uncertainty = nn.Sequential(
#             nn.Conv2d(4, u_hidden, 3, padding=1, bias=False),
#             nn.BatchNorm2d(u_hidden),
#             nn.SiLU(),
#             nn.Conv2d(u_hidden, 1, 1, bias=True),
#             nn.Sigmoid()
#         )

#         # 开局退化为原始 WeightedAdd，训练中再慢慢学习是否启用局部修正
#         self.alpha = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         rgb, ir = x[0], x[1]

#         diff = torch.abs(rgb - ir)
#         common = rgb * ir
#         fusion = torch.cat([rgb, ir, diff, common], dim=1)

#         # 全局分数 [B,2,1,1]
#         s_global = self.global_score(fusion)

#         # 局部分数 [B,2,H,W]
#         s_local = self.local_score(fusion)

#         # 低维不确定性图 [B,1,H,W]
#         u_in = torch.cat([
#             diff.mean(dim=1, keepdim=True),
#             common.mean(dim=1, keepdim=True),
#             rgb.abs().mean(dim=1, keepdim=True),
#             ir.abs().mean(dim=1, keepdim=True),
#         ], dim=1)
#         u = self.uncertainty(u_in)

#         # 只有在不确定区域才启用局部修正
#         score = s_global + self.alpha * u * torch.tanh(s_local)

#         w = torch.softmax(score, dim=1)
#         wr = w[:, 0:1]
#         wi = w[:, 1:2]

#         out = wr * rgb + wi * ir
#         return out

class ResidualWeightedAdd(nn.Module):
    """
    Self-contained ResidualWeightedAdd
    输入:  (rgb, ir)
    输出:  fused feature

    在最终融合时，引入 feature-level 的低光代理和 IR 显著性代理，让融合决策更懂“什么时候 RGB 可能不可靠，什么时候 IR 更值得信”

    思路:
    1) global 分支保留 WeightedAdd 的整图级裁决
    2) local 分支只看低维冲突/共识线索，做局部纠偏
    3) 模块内部自己构造 low_map / ir_map，不改外部输入接口
    """

    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)

        # -----------------------------
        # global 分支：整图级模态偏向
        # -----------------------------
        self.global_main = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),
            # nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.global_prior = nn.Sequential(
            nn.Conv2d(2, hidden, 1, bias=False),
            # nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )

        self.global_out = nn.Conv2d(hidden * 2, 2, 1, bias=True)

        # -----------------------------
        # local 分支：局部残差修正
        # 输入 6 通道:
        # diff_map, common_map, sim_map, energy_map, low_map, ir_map
        # -----------------------------
        self.local_score = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, 2, 1, bias=True)
        )

        # -----------------------------
        # u 分支：控制哪里需要局部修正
        # -----------------------------
        self.u_gate = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # -----------------------------
        # 局部修正强度
        # 初始接近 0，开局更像原始 WeightedAdd
        # alpha = 0.5 * sigmoid(alpha_raw)
        # -----------------------------
        self.alpha_raw = nn.Parameter(torch.tensor(-6.0))

    @staticmethod
    def _norm_map(x, eps=1e-6):
        """
        x: [B,1,H,W]
        对每张图做 0~1 归一化
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _build_internal_priors(self, rgb, ir):
        """
        从当前层 feature 内部构造两类代理先验:
        1) low_map: RGB 弱响应区域 -> 更像低光/弱纹理区域
        2) ir_map : IR 强响应区域  -> 更像 IR 显著区域
        """
        rgb_resp = rgb.abs().mean(dim=1, keepdim=True)   # [B,1,H,W]
        ir_resp = ir.abs().mean(dim=1, keepdim=True)     # [B,1,H,W]

        rgb_norm = self._norm_map(rgb_resp)
        ir_norm = self._norm_map(ir_resp)

        low_map = 1.0 - rgb_norm
        ir_map = ir_norm
        return low_map, ir_map

    def forward(self, x):
        rgb, ir = x[0], x[1]   # [B,C,H,W]

        # --------------------------------
        # 基础关系特征
        # --------------------------------
        diff = torch.abs(rgb - ir)

        diff_map = diff.mean(dim=1, keepdim=True)   # 冲突强度
        common_map = (F.relu(rgb) * F.relu(ir)).mean(dim=1, keepdim=True)  # 共激活
        sim_map = F.cosine_similarity(rgb, ir, dim=1, eps=1e-6).unsqueeze(1)  # 相似性
        energy_map = 0.5 * (
            rgb.abs().mean(dim=1, keepdim=True) +
            ir.abs().mean(dim=1, keepdim=True)
        )  # 整体响应强度

        # --------------------------------
        # 内部 low-light / IR prior
        # --------------------------------
        low_map, ir_map = self._build_internal_priors(rgb, ir)

        # --------------------------------
        # global 分支
        # --------------------------------
        g_main = self.global_main(torch.cat([rgb, ir, diff], dim=1))  # [B,h,1,1]

        g_prior = self.global_prior(
            torch.cat([
                F.adaptive_avg_pool2d(low_map, 1),
                F.adaptive_avg_pool2d(ir_map, 1)
            ], dim=1)
        )  # [B,h,1,1]

        s_global = self.global_out(torch.cat([g_main, g_prior], dim=1))  # [B,2,1,1]

        # --------------------------------
        # local / u 分支
        # --------------------------------
        local_in = torch.cat([
            diff_map,
            common_map,
            sim_map,
            energy_map,
            low_map,
            ir_map
        ], dim=1)  # [B,6,H,W]

        s_local = self.local_score(local_in)   # [B,2,H,W]
        u = self.u_gate(local_in)              # [B,1,H,W]

        alpha = 0.5 * torch.sigmoid(self.alpha_raw)

        # global 主判断 + local 残差修正
        logits = s_global + alpha * u * torch.tanh(s_local)

        # --------------------------------
        # 最终融合
        # --------------------------------
        w = torch.softmax(logits, dim=1)
        wr = w[:, 0:1]
        wi = w[:, 1:2]

        out = wr * rgb + wi * ir
        return out

class WeightedAdd(nn.Module):
    """
    对齐和双流更新后，再做自适应加权融合
    输入: (rgb, ir)
    输出: fused
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)

        self.score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1, bias=True)
        )

    def forward(self, x):
        rgb, ir = x[0], x[1]
        diff = torch.abs(rgb - ir)

        s = self.score(torch.cat([rgb, ir, diff], dim=1))
        w = torch.softmax(s, dim=1)

        wr = w[:, 0:1]
        wi = w[:, 1:2]

        out = wr * rgb + wi * ir
        return out

class ResidualWeightedAddStable(nn.Module):
    """
    Stable 版最终融合模块
    输入:  (rgb, ir)
    输出:  fused tensor

    设计目标：
    1) 保留 WeightedAdd 的全局稳定性
    2) 只做轻量局部纠偏，不重做前面的交互/语义建模
    3) 用受限强度 + 稀疏不确定性 + 固定平均残差，降低偏置风险
    """
    def __init__(self, c):
        super().__init__()
        hidden = max(c // 4, 16)
        u_hidden = max(hidden // 2, 8)

        # 1) 全局模态打分：继承原始 WeightedAdd 主体
        self.global_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c * 3, hidden, 1, bias=False),   # rgb, ir, diff
            nn.SiLU(),
            nn.Conv2d(hidden, 2, 1, bias=True)         # [B,2,1,1]
        )

        # 2) 局部修正分支：很轻，只做 residual correction
        # 输入多加一个 common = rgb * ir，表示共识区域
        self.local_score = nn.Sequential(
            nn.Conv2d(c * 4, hidden, 1, bias=False),   # rgb, ir, diff, common
            nn.BatchNorm2d(hidden),
            nn.SiLU(),

            nn.Conv2d(
                hidden, hidden, 3, padding=1,
                groups=max(hidden // 16, 1), bias=False
            ),
            nn.BatchNorm2d(hidden),
            nn.SiLU(),

            nn.Conv2d(hidden, 2, 1, bias=True)         # [B,2,H,W]
        )

        # 3) 不确定性图：只在冲突/不确定区域放大 local correction
        self.uncertainty = nn.Sequential(
            nn.Conv2d(4, u_hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(u_hidden),
            nn.SiLU(),
            nn.Conv2d(u_hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 4) 局部修正强度，限制上限，避免过激
        # 实际使用时 alpha = 0.3 * sigmoid(alpha_raw)
        self.alpha_raw = nn.Parameter(torch.zeros(1))

        # 5) 保留一部分固定平均融合，抑制偏置
        # out = (1-beta) * avg + beta * adaptive
        self.beta = 0.8   # 可后续再调 0.7 / 0.8 / 0.85

    def forward(self, x):
        rgb, ir = x[0], x[1]

        diff = torch.abs(rgb - ir)
        common = rgb * ir

        # -------- global --------
        s_global = self.global_score(torch.cat([rgb, ir, diff], dim=1))   # [B,2,1,1]

        # -------- local residual correction --------
        s_local = self.local_score(torch.cat([rgb, ir, diff, common], dim=1))  # [B,2,H,W]

        # -------- uncertainty map --------
        u_in = torch.cat([
            diff.mean(dim=1, keepdim=True),          # 模态冲突强度
            common.mean(dim=1, keepdim=True),        # 模态共识强度
            rgb.abs().mean(dim=1, keepdim=True),     # RGB 响应
            ir.abs().mean(dim=1, keepdim=True),      # IR 响应
        ], dim=1)

        u = self.uncertainty(u_in)   # [B,1,H,W]
        u = u * u                    # 稀疏化，只在更不确定区域才放大修正

        # -------- limited correction strength --------
        alpha = 0.3 * torch.sigmoid(self.alpha_raw)

        logits = s_global + alpha * u * torch.tanh(s_local)
        w = torch.softmax(logits, dim=1)

        wr = w[:, 0:1]
        wi = w[:, 1:2]

        adaptive = wr * rgb + wi * ir
        avg_base = 0.5 * (rgb + ir)

        out = (1.0 - self.beta) * avg_base + self.beta * adaptive
        return out

class IN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Multiin(nn.Module):  # stereo attention block
    def __init__(self, out=1):
        super().__init__()
        self.out = out

    def forward(self, x):
        x1, x2 = x[:, :3, :, :], x[:, 3:, :, :]
        if self.out == 1:
            x = x1
        else:
            x = x2
        return x


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class EFBlock(nn.Module):
    def __init__(self, c1, c2, reduction=16):
        super(EFBlock, self).__init__()
        self.mask_map_r = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.mask_map_i = nn.Conv2d(c1 // 2, 1, 1, 1, 0, bias=True)
        self.softmax = nn.Softmax(-1)
        self.bottleneck1 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.bottleneck2 = nn.Conv2d(c1 // 2, c2 // 2, 3, 1, 1, bias=False)
        self.se = SE_Block(c2, reduction)

    def forward(self, x):
        x_left_ori, x_right_ori = x[:, :3, :, :], x[:, 3:, :, :]
        x_left = x_left_ori * 0.5
        x_right = x_right_ori * 0.5

        x_mask_left = torch.mul(self.mask_map_r(x_left), x_left)
        x_mask_right = torch.mul(self.mask_map_i(x_right), x_right)

        out_IR = self.bottleneck1(x_mask_right + x_right_ori)
        out_RGB = self.bottleneck2(x_mask_left + x_left_ori)  # RGB
        out = self.se(torch.cat([out_RGB, out_IR], 1))

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.SiLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
class FEM(nn.Module):
    def __init__(self, in_planes, out_planes, n=3,stride=1, scale=0.1, map_reduce=4):
        super(FEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, 2 * inter_planes, kernel_size=1, stride=stride),
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=1),
            BasicConv(2*inter_planes, 2*inter_planes , kernel_size=(1, 3), stride=stride, padding=(0, 1)),
            BasicConv(2*inter_planes, 2 * inter_planes, kernel_size=(3, 1), stride=stride, padding=(1, 0)),
        )



    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        return out
    
class C2f_FEM(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList([*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n//2)),FEM(self.c,self.c)] )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
import numpy as np
import torch
from torch import nn
from torch.nn import init

# https://arxiv.org/abs/2108.01072
def spatial_shift1(x):
    b,w,h,c = x.size()
    x[:,1:,:,:c//4] = x[:,:w-1,:,:c//4]
    x[:,:w-1,:,c//4:c//2] = x[:,1:,:,c//4:c//2]
    x[:,:,1:,c//2:c*3//4] = x[:,:,:h-1,c//2:c*3//4]
    x[:,:,:h-1,3*c//4:] = x[:,:,1:,3*c//4:]
    return x


def spatial_shift2(x):
    b,w,h,c = x.size()
    x[:,:,1:,:c//4] = x[:,:,:h-1,:c//4]
    x[:,:,:h-1,c//4:c//2] = x[:,:,1:,c//4:c//2]
    x[:,1:,:,c//2:c*3//4] = x[:,:w-1,:,c//2:c*3//4]
    x[:,:w-1,:,3*c//4:] = x[:,1:,:,3*c//4:]
    return x


class SplitAttention(nn.Module):
    def __init__(self,channel=512,k=3):
        super().__init__()
        self.channel=channel
        self.k=k
        self.mlp1=nn.Linear(channel,channel,bias=False)
        self.gelu=nn.GELU()
        self.mlp2=nn.Linear(channel,channel*k,bias=False)
        self.softmax=nn.Softmax(1)
    
    def forward(self,x_all):
        b,k,h,w,c=x_all.shape
        x_all=x_all.reshape(b,k,-1,c) 
        a=torch.sum(torch.sum(x_all,1),1) 
        hat_a=self.mlp2(self.gelu(self.mlp1(a))) 
        hat_a=hat_a.reshape(b,self.k,c) 
        bar_a=self.softmax(hat_a) 
        attention=bar_a.unsqueeze(-2) 
        out=attention*x_all 
        out=torch.sum(out,1).reshape(b,h,w,c)
        return out
#NAM
class NAM(nn.Module):
    def __init__(self, channels,c2, t=16):
        super(NAM, self).__init__()
        self.channels = channels
        self.conv=Conv(channels,c2,1,1)
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
    def forward(self, x):
        x=torch.cat(x,1)
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual  #
        x=self.conv(x)
        return x
    
    
class GLF(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        self.conv=Conv(c1,c2,1,1)
        self.d=1

        self.avg_pool = nn.AdaptiveAvgPool2d(1) #全局池化
        # 全局特征提取
        self.fc1 = nn.Sequential(
         
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel ),
            nn.Sigmoid()
        )
        # 局部特征提取
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction,1,1),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel,1,1),
            nn.BatchNorm2d(channel),
        )



    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, _, _ = x.size()
        
        # 全局特征mul
        y = self.avg_pool(x)
        y = self.fc1(y).view(b, c, 1, 1)

        #局部特征
        y1= self.fc2(x)

        x=x * y.expand_as(x) 
        #局部特征add
        x=torch.add(x, y1)
        
        x=self.conv(x)

        return x
    




from collections import OrderedDict


class SKAttention(nn.Module):

    def __init__(self,c1,c2, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)



    def forward(self, x):

        x=torch.cat(x,1)
        bs, c, _, _ = x.size()
        conv_outs=[]
        ### split
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        V=self.conv(V)
        return V


    


class S2Attention(nn.Module):

    def __init__(self, c1,c2,channels=512 ):
        super().__init__()
        channels=c1
        self.conv=Conv(c1,c2,1,1)

        self.mlp1 = nn.Linear(channels,channels*3)
        self.mlp2 = nn.Linear(channels,channels)
        self.split_attention = SplitAttention(c1)

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b,c,w,h = x.size()
        x=x.permute(0,2,3,1)
        x = self.mlp1(x)
        x1 = spatial_shift1(x[:,:,:,:c])
        x2 = spatial_shift2(x[:,:,:,c:c*2])
        x3 = x[:,:,:,c*2:]
        x_all=torch.stack([x1,x2,x3],1)
        a = self.split_attention(x_all)
        x = self.mlp2(a)
        x=x.permute(0,3,1,2)
        x=self.conv(x)
        return x
  
  

 
 
###################### EffectiveSE     ####     end   by  AI&CV  ###############################

import numpy as np
import torch
from torch import nn
from torch.nn import init

class ChannelAttentionModule(nn.Module):
    def __init__(self, c1, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = c1 // reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=c1, out_features=mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(in_features=mid_channel, out_features=c1)
        )
        self.act = nn.Sigmoid()
        #self.act=nn.SiLU()
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0),-1)).unsqueeze(2).unsqueeze(3)
        return self.act(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.act = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.act(self.conv2d(out))
        return out

class CBAM2(nn.Module):
    def __init__(self, c1,c2):
        super(CBAM2, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        x=self.conv(out)
        return x


class CSFM(nn.Module):
    def __init__(self, c1,c2):
        super(CSFM, self).__init__()
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        _,c,_,_=x[0].shape
        x3=x[0]
        x4=x[1]
        x=torch.cat(x, self.d) 
        out = self.channel_attention(x) * x
        x1, x2 = torch.split(out, c, dim =self.d)

        x1=x1*x3
        x2=x2*x4
        # x1+=x[0]
        # x2+=x[1]
        out=torch.add(x1,x2)
        # out = self.spatial_attention(out) * out
        
        return out

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output
    



class SACBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)

        x = self.channel_shuffle(x, 2)
        x_channel=self.channel_attention(x) * x
        out=self.spatial_attention(x_channel) * x_channel
        out=self.conv(out)
        return out
    


    
class GCBAM(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        


        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures

        # x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        # x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # # channel attention
        # x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        # x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        # x_channel = x_0 * self.sigmoid(x_channel)

        # # spatial attention
        # x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        # x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        # x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        x_channel=self.channel_attention(x) * x
        
        out=self.spatial_attention(x_channel) * x_channel
        # concatenate along channel axis
        # out = torch.cat([x_channel, x_spatial], dim=1) 
        # out = out.contiguous().view(b, -1, h, w)
        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out
    
    
# 局部CBAM
class GLCBAM(nn.Module):
    def __init__(self, c1,c2):
        super(GLCBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        mid_channel=c1//16
        
        #局部特征
        self.localConv = nn.Sequential(          
            nn.Conv2d(in_channels=c1, out_channels=mid_channel,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(mid_channel),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=mid_channel, out_channels=c1,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(c1),
        )

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.localConv(y)
        out=torch.add(local,out)
        
        x=self.conv(out)

        return x


class SACBAM(nn.Module):
    def __init__(self, c1,c2):
        super(SACBAM, self).__init__()
        self.conv=Conv(c1,c2,1,1)
        self.d=1 
        self.channel_attention = ChannelAttentionModule(c1)
        self.spatial_attention = SpatialAttentionModule()
        self.SA=ShuffleAttention(c1,c2)
        mid_channel=c1//16
        
 

    def forward(self, x):
        x=torch.cat(x, self.d) 
        y=x
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        
        local=self.SA(y)
        out=torch.add(local,out)
    
        x=self.conv(out)

        return x
    

class SEAttention(nn.Module):

    def __init__(self, c1,c2,channel=512, reduction=16):
        super().__init__()
        channel=c1
        # self.conv=Conv(c1,c2,1,1)
        self.d=1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        x=x * y.expand_as(x) 
        # x=self.conv(x)
        return x
    
class Concat2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        #self.conv=nn.Conv2d(c1,c2,1,1,bias=False)
        self.conv=Conv(c1,c2,1,1)

    def forward(self, x):
        x=torch.cat(x, self.d)
        x=self.conv(x)
        return x
class SA(nn.Module):

    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out
    
from torch.nn import init
from torch.nn.parameter import Parameter

class SimAM(torch.nn.Module):
    def __init__(self, c1,c2,e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
        self.d=1
        self.conv=Conv(c1,c2,1,1)

        

    def forward(self, x):
        x=torch.cat(x, self.d)
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        x= x * self.activaton(y)
        x=self.conv(x)
        return x






class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
 
    def forward(self, x):
        return self.relu(x + 3) / 6
 
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
 
    def forward(self, x):
        return x * self.sigmoid(x)
 
class CoordAtt(nn.Module):
    def __init__(self, inp,c2, reduction=32):
        super(CoordAtt, self).__init__()
        self.conv=Conv(inp,c2,1,1)
        oup = inp
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
 
        mip = max(8, inp // reduction)
 
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        
 
    def forward(self, x):
        x=torch.cat(x,dim=1)
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
 
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
 
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
 
        out = identity * a_w * a_h
 
        return self.conv(out)

import torch
from torch import nn
from torch.nn.parameter import Parameter
class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x
    
# class ECA(nn.Module):
#     """Constructs a ECA module.
#     Args:
#         channel: Number of channels of the input feature map
#         k_size: Adaptive selection of kernel size
#     """
#     def __init__(self, c1,c2, k_size=3):
#         super(ECA, self).__init__()
#         self.conv1=Conv(c1,c2,1,1)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
#         self.sigmoid = nn.Sigmoid()
 
#     def forward(self, x):
#         # feature descriptor on the global spatial information
#         x=torch.cat(x,dim=1)
#         y = self.avg_pool(x)
 
#         # Two different branches of ECA module
#         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
 
#         # Multi-scale information fusion
#         y = self.sigmoid(y)
 
#         return self.conv1(x * y.expand_as(x))
    
class GAM_Attention(nn.Module):
    # https://paperswithcode.com/paper/global-attention-mechanism-retain-information
    def __init__(self, c1, c2, group=True, rate=4):
        super(GAM_Attention, self).__init__()
        self.conv=Conv(c1,c2,1,1)

        c2=c1
        self.d=1
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1)
        )

        self.spatial_attention = nn.Sequential(

            nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(c1, int(c1 / rate),
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate) if group else nn.Conv2d(int(c1 / rate), c2,
                                                                                                     kernel_size=7,
                                                                                                     padding=3),
            nn.BatchNorm2d(c2)
        )

    def forward(self, x):
        x=torch.cat(x,dim=self.d)
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        # x_channel_att=channel_shuffle(x_channel_att,4) #last shuffle
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)  # last shuffle
        out = x * x_spatial_att
        # out=channel_shuffle(out,4) #last shuffle
        out=self.conv(out)
        return out
    
class ShuffleAttention(nn.Module):

    def __init__(self,c1,c2, channel=512, reduction=16, G=8):
        super().__init__()
        self.conv=Conv(c1,c2,1,1)
        channel=c1
        self.G = G
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        x=torch.cat(x,dim=1)
        b, c, h, w = x.size()
        # group into subfeatures
        x = x.view(b * self.G, -1, h, w)  # bs*G,c//G,h,w

        # channel_split
        x_0, x_1 = x.chunk(2, dim=1)  # bs*G,c//(2*G),h,w

        # channel attention
        x_channel = self.avg_pool(x_0)  # bs*G,c//(2*G),1,1
        x_channel = self.cweight * x_channel + self.cbias  # bs*G,c//(2*G),1,1
        x_channel = x_0 * self.sigmoid(x_channel)

        # spatial attention
        x_spatial = self.gn(x_1)  # bs*G,c//(2*G),h,w
        x_spatial = self.sweight * x_spatial + self.sbias  # bs*G,c//(2*G),h,w
        x_spatial = x_1 * self.sigmoid(x_spatial)  # bs*G,c//(2*G),h,w

        # concatenate along channel axis
        out = torch.cat([x_channel, x_spatial], dim=1)  # bs*G,c//G,h,w
        out = out.contiguous().view(b, -1, h, w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        out=self.conv(out)
        return out






        
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        # a 8400
        # c1=16 
        # 4 16 a 
        # 16 4 a 
        # 4 a
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))

class MdC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=i+1) for i in range(n))
        

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
class CDC2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        # Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)
        # 3 1 3 8 5 2  5 2     k= 3, 3, 5, and 5 and d= 1, 8, 2, and 3
        if n==1:
           # high pass d
           # 3 1 /3 8/5 3
           self.m = nn.ModuleList(Md(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0,deiltations=8))


        else :
           # low pass c
           # 3 1/3 8/ 5 2/ 5 3/ 3 3/ 5 5 
           self.m = nn.ModuleList((Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=2),
                                  Md(self.c, self.c, shortcut, g, k=((3, 3), (5, 5)), e=1.0,deiltations=3)) )

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    


class C2f_F(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = c1//4 # hidden channels
        self.c1=self.c*3
        self.cv1 = Conv(c1, c1, 1, 1)
        self.cv2 = Conv((4 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Conv(self.c,self.c,k=3,s=1) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x=self.cv1(x)
        c1=3*self.c
        x1 = x[:, :c1, :, :]  
        # 第二部分  
        x2 = x[:, c1:, :, :] 
        y=list([x1,x2])
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


import torch
import torch.nn as nn
from torch.nn import functional as F
 
 

 
class Involution(nn.Module):
 
    def __init__(self, c1, c2, kernel_size, stride):
        super(Involution, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.c1 = c1
        reduction_ratio = 4
        self.group_channels = 16
        self.groups = self.c1 // self.group_channels
        self.conv1 = Conv(
            c1, c1 // reduction_ratio, 1)
        self.conv2 = Conv(
            c1 // reduction_ratio,
            kernel_size ** 2 * self.groups,
            1, 1)
 
        if stride > 1:
            self.avgpool = nn.AvgPool2d(stride, stride)
        self.unfold = nn.Unfold(kernel_size, 1, (kernel_size - 1) // 2, stride)
 
    def forward(self, x):
        weight = self.conv2(self.conv1(x if self.stride == 1 else self.avgpool(x)))
        b, c, h, w = weight.shape
        weight = weight.view(b, self.groups, self.kernel_size ** 2, h, w).unsqueeze(2)
        out = self.unfold(x).view(b, self.groups, self.group_channels, self.kernel_size ** 2, h, w)
        out = (weight * out).sum(dim=3).view(b, self.c1, h, w)
 
        return out

from ultralytics.utils.torch_utils import make_divisible


class PKIModule_CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor
    

class PKIModule(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True) -> None:
        super().__init__()
        hidc = make_divisible(int(ouc * expansion), 8)
        
        self.pre_conv = Conv(inc, hidc)
        self.dw_conv = nn.ModuleList(nn.Conv2d(hidc, hidc, kernel_size=k, padding=autopad(k), groups=hidc) for k in kernel_sizes)
        self.pw_conv = Conv(hidc, hidc)
        self.post_conv = Conv(hidc, ouc)
        
        if with_caa:
            self.caa_factor = PKIModule_CAA(hidc, caa_kernel_size, caa_kernel_size)
        else:
            self.caa_factor = None
        
        self.add_identity = add_identity and inc == ouc
    
    def forward(self, x):
        x = self.pre_conv(x)
        
        y = x
        x = self.dw_conv[0](x)
        x = torch.sum(torch.stack([x] + [layer(x) for layer in self.dw_conv[1:]], dim=0), dim=0)
        x = self.pw_conv(x)
        
        if self.caa_factor is not None:
            y = self.caa_factor(y)
        if self.add_identity:
            y = x * y
            x = x + y
        else:
            x = x * y

        x = self.post_conv(x)
        return x
    


class C2f_PKIModule(C2f):
    def __init__(self, c1, c2, n=1, kernel_sizes=(3, 5, 7, 9, 11), expansion=1.0, with_caa=True, caa_kernel_size=11, add_identity=True, g=1, e=0.5):
        super().__init__(c1, c2, n, True, g, e)
        self.m = nn.ModuleList(PKIModule(self.c, self.c, kernel_sizes, expansion, with_caa, caa_kernel_size, add_identity) for _ in range(n))

class ShuffleNetV2(nn.Module):
    def __init__(self, inp, oup, stride):  # ch_in, ch_out, stride
        super().__init__()

        self.stride = stride

        branch_features = oup // 2 # 输出的一半
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride == 2:
            # copy input
            self.branch1 = nn.Sequential(
                nn.Conv2d(inp, inp, kernel_size=3, stride=self.stride, padding=1, groups=inp),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True))
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride == 2) else branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            #Dw卷积
            nn.Conv2d(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1, groups=branch_features),
            nn.BatchNorm2d(branch_features),
            #Pw
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = self.channel_shuffle(out, 2)

        return out

    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        return out
    
class C2f_Shufflenet(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(ShuffleNetV2(self.c, self.c,1) for _ in range(n))

class C2f_Invo(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(InvoConv(self.c, self.c,1) for _ in range(n))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class InvoConv(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Involution(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class Md(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5,deiltations=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g,d=deiltations)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class ADD(nn.Module):
    #  Add two tensors
    
    def __init__(self, arg):
        super(ADD,self).__init__()
        # 128 256 512
        self.arg = arg
  
    def forward(self, x):
        return torch.add(x[0], x[1])



class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x

class LocalGlobalAttention(nn.Module):
    def __init__(self, output_dim, patch_size):
        super().__init__()
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.mlp1 = nn.Linear(patch_size*patch_size, output_dim // 2)
        self.norm = nn.LayerNorm(output_dim // 2)
        self.mlp2 = nn.Linear(output_dim // 2, output_dim)
        self.conv = nn.Conv2d(output_dim, output_dim, kernel_size=1)
        self.prompt = torch.nn.parameter.Parameter(torch.randn(output_dim, requires_grad=True)) 
        self.top_down_transform = torch.nn.parameter.Parameter(torch.eye(output_dim), requires_grad=True)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        P = self.patch_size

        # Local branch
        local_patches = x.unfold(1, P, P).unfold(2, P, P)  # (B, H/P, W/P, P, P, C)
        local_patches = local_patches.reshape(B, -1, P*P, C)  # (B, H/P*W/P, P*P, C)
        local_patches = local_patches.mean(dim=-1)  # (B, H/P*W/P, P*P)

        local_patches = self.mlp1(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.norm(local_patches)  # (B, H/P*W/P, input_dim // 2)
        local_patches = self.mlp2(local_patches)  # (B, H/P*W/P, output_dim)

        local_attention = F.softmax(local_patches, dim=-1)  # (B, H/P*W/P, output_dim)
        local_out = local_patches * local_attention # (B, H/P*W/P, output_dim)

        cos_sim = F.normalize(local_out, dim=-1) @ F.normalize(self.prompt[None, ..., None], dim=1)  # B, N, 1
        mask = cos_sim.clamp(0, 1)
        local_out = local_out * mask
        local_out = local_out @ self.top_down_transform

        # Restore shapes
        local_out = local_out.reshape(B, H // P, W // P, self.output_dim)  # (B, H/P, W/P, output_dim)
        local_out = local_out.permute(0, 3, 1, 2)
        local_out = F.interpolate(local_out, size=(H, W), mode='bilinear', align_corners=False)
        output = self.conv(local_out)

        return output

class ECA(nn.Module):
    def __init__(self,in_channel,gamma=2,b=1):
        super(ECA, self).__init__()
        k=int(abs((math.log(in_channel,2)+b)/gamma))
        kernel_size=k if k % 2 else k+1
        padding=kernel_size//2
        self.pool=nn.AdaptiveAvgPool2d(output_size=1)
        self.conv=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        out=self.pool(x)
        out=out.view(x.size(0),1,x.size(1))
        out=self.conv(out)
        out=out.view(x.size(0),x.size(1),1,1)
        return out*x

# https://mp.weixin.qq.com/s/26H0PgN5sikD1MoSkIBJzg
class PPA(nn.Module):
    def __init__(self, in_features, filters) -> None:
         super().__init__()

         self.skip = Conv(in_features, filters, act=False)
         self.c1 = Conv(filters, filters, 3)
         self.c2 = Conv(filters, filters, 3)
         self.c3 = Conv(filters, filters, 3)
         self.sa = SpatialAttentionModule()
         self.cn = ECA(filters)
         self.lga2 = LocalGlobalAttention(filters, 2)
         self.lga4 = LocalGlobalAttention(filters, 4)

         self.drop = nn.Dropout2d(0.1)
         self.bn1 = nn.BatchNorm2d(filters)
         self.silu = nn.SiLU()

    def forward(self, x):
        x_skip = self.skip(x)
        x_lga2 = self.lga2(x_skip)
        x_lga4 = self.lga4(x_skip)
        x1 = self.c1(x)
        x2 = self.c2(x1)
        x3 = self.c3(x2)
        x = x1 + x2 + x3 + x_skip + x_lga2 + x_lga4
        x = self.cn(x)
        x = self.sa(x)
        x = self.drop(x)
        x = self.bn1(x)
        x = self.silu(x)
        return x


class C2f_PPA(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(PPA(self.c, self.c) for _ in range(n))

from timm.models.layers import DropPath


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        # only for inference
        x = x.clone()  # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        # for training/inference
        # x = x.clone()  # !!! Keep the original input intact for the residual connection later
        # x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        # return x
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class Faster_Block(nn.Module):
    def __init__(self,
                 inc,
                 dim,
                 n_div=4,
                 mlp_ratio=1,
                 drop_path=0.1,
                 layer_scale_init_value=0.0,
                 pconv_fw_type='split_cat'
                 ):
        super().__init__()

        self.dim = dim
        self.mlp_ratio = mlp_ratio
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.n_div = n_div

        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = [
            Conv(dim, mlp_hidden_dim, 1),
            # nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        ]

        self.mlp = nn.Sequential(*mlp_layer)

        self.spatial_mixing = Partial_conv3(
            dim,
            n_div,
            pconv_fw_type
        )

        # self.adjust_channel = None
        # if inc != dim:
        #     self.adjust_channel = Conv(inc, dim, 1)

        # if layer_scale_init_value > 0:
        #     self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        #     self.forward = self.forward_layer_scale
        # else:
        #     self.forward = self.forward

    def forward(self, x):
        # if self.adjust_channel is not None:
        #     x = self.adjust_channel(x)
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(self.mlp(x))
        
        return self.mlp(x)

    def forward_layer_scale(self, x):
        # shortcut = x
        x = self.spatial_mixing(x)
        # x = shortcut + self.drop_path(
        #     self.layer_scale.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        return x


class C2f_Faster(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(Faster_Block(self.c, self.c) for _ in range(n))















class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv)
        self.fusion_bn = nn.Sequential(*fusion_bn)

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()
    

    def forward(self, x):
        

        x1 = self.primary_conv(x)  # mg
        x2 = self.cheap_operation(x1)
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v



class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn   # 激活函数
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
 
    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)



class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )
          

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x) #
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)
    

class RepGhostModule(nn.Module):
    def __init__(
            self, inp, oup, kernel_size=1, dw_size=3, stride=1, relu=True, deploy=False, reparam_bn=True,
            reparam_identity=False
    ):
        super(RepGhostModule, self).__init__()
        init_channels = oup
        new_channels = oup
        self.deploy = deploy
        # 1x1 conv + bn + SiLU
        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False,
            ),
            nn.BatchNorm2d(init_channels),
            nn.SiLU(inplace=True) if relu else nn.Sequential(),
        )
        fusion_conv = []
        fusion_bn = []
        if not deploy and reparam_bn:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.BatchNorm2d(init_channels))
        if not deploy and reparam_identity:
            fusion_conv.append(nn.Identity())
            fusion_bn.append(nn.Identity())

        self.fusion_conv = nn.Sequential(*fusion_conv) #indentity
        self.fusion_bn = nn.Sequential(*fusion_bn) #fusion bn

        # dwconv BN Silu
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=deploy,
            ),
            nn.BatchNorm2d(new_channels) if not deploy else nn.Sequential(),
            # nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )
        if deploy:
            self.cheap_operation = self.cheap_operation[0]
        if relu:
            self.relu = nn.SiLU(inplace=False)
        else:
            self.relu = nn.Sequential()

    def forward(self, x):
        x1 = self.primary_conv(x)  # conv1x1 SiLu
        x2 = self.cheap_operation(x1) # dw BN SiLu
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            x2 = x2 + bn(conv(x1))# indentity x1 + bn
        return self.relu(x2)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.cheap_operation[0], self.cheap_operation[1])
        for conv, bn in zip(self.fusion_conv, self.fusion_bn):
            kernel, bias = self._fuse_bn_tensor(conv, bn, kernel3x3.shape[0], kernel3x3.device)
            kernel3x3 += self._pad_1x1_to_3x3_tensor(kernel)
            bias3x3 += bias
        return kernel3x3, bias3x3

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    @staticmethod
    def _fuse_bn_tensor(conv, bn, in_channels=None, device=None):
        in_channels = in_channels if in_channels else bn.running_mean.shape[0]
        device = device if device else bn.weight.device
        if isinstance(conv, nn.Conv2d):
            kernel = conv.weight
            assert conv.bias is None
        else:
            assert isinstance(conv, nn.Identity)
            kernel_value = np.zeros((in_channels, 1, 1, 1), dtype=np.float32)
            for i in range(in_channels):
                kernel_value[i, 0, 0, 0] = 1
            kernel = torch.from_numpy(kernel_value).to(device)

        if isinstance(bn, nn.BatchNorm2d):
            running_mean = bn.running_mean
            running_var = bn.running_var
            gamma = bn.weight
            beta = bn.bias
            eps = bn.eps
            std = (running_var + eps).sqrt()
            t = (gamma / std).reshape(-1, 1, 1, 1)
            return kernel * t, beta - running_mean * gamma / std
        assert isinstance(bn, nn.Identity)
        return kernel, torch.zeros(in_channels).to(kernel.device)

    def switch_to_deploy(self):
        if len(self.fusion_conv) == 0 and len(self.fusion_bn) == 0:
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.cheap_operation = nn.Conv2d(in_channels=self.cheap_operation[0].in_channels,
                                         out_channels=self.cheap_operation[0].out_channels,
                                         kernel_size=self.cheap_operation[0].kernel_size,
                                         padding=self.cheap_operation[0].padding,
                                         dilation=self.cheap_operation[0].dilation,
                                         groups=self.cheap_operation[0].groups,
                                         bias=True)
        self.cheap_operation.weight.data = kernel
        self.cheap_operation.bias.data = bias
        self.__delattr__('fusion_conv')
        self.__delattr__('fusion_bn')
        self.fusion_conv = []
        self.fusion_bn = []
        self.deploy = True


class RepGhostBottleneck(nn.Module):
    """RepGhost bottleneck w/ optional SE"""

    def __init__(
            self,
            in_chs,
            
            out_chs,
            dw_kernel_size=3,
            stride=1,
            se_ratio=0.0,
            shortcut=True,
            reparam=True,
            reparam_bn=True,
            reparam_identity=False,
            deploy=False,
    ):
        super(RepGhostBottleneck, self).__init__()
        mid_chs=in_chs//2
        has_se = se_ratio is not None and se_ratio > 0.0
        self.stride = stride
        self.enable_shortcut = shortcut
        self.in_chs = in_chs
        self.out_chs = out_chs

        # Point-wise expansion
        self.ghost1 = RepGhostModule(
            in_chs,
            mid_chs,
            relu=True,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs,
                mid_chs,
                dw_kernel_size,
                stride=stride,
                padding=(dw_kernel_size - 1) // 2,
                groups=mid_chs,
                bias=False,
            )
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = RepGhostModule(
            mid_chs,
            out_chs,
            relu=False,
            reparam_bn=reparam and reparam_bn,
            reparam_identity=reparam and reparam_identity,
            deploy=deploy,
        )

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias=False,
                ),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(
                    in_chs, out_chs, 1, stride=1,
                    padding=0, bias=False,
                ),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        residual = x
        x1 = self.ghost1(x)
        if self.stride > 1:
            x = self.conv_dw(x1)
            x = self.bn_dw(x)
        else:
            x = x1

        if self.se is not None:
            x = self.se(x)

        # 2nd repghost bottleneck mg
        x = self.ghost2(x)
        if not self.enable_shortcut and self.in_chs == self.out_chs and self.stride == 1:
            return x
        return x + self.shortcut(residual)

class C2f_RG(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(RepGhostBottleneck(self.c, self.c) for _ in range(n))




### bifpn##


class GSConv(nn.Module):
    # GSConv https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, p, g, d, Conv.default_act)
        self.cv2 = Conv(c_, c_, 5, 1, p, c_, d, Conv.default_act)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # shuffle
        # y = x2.reshape(x2.shape[0], 2, x2.shape[1] // 2, x2.shape[2], x2.shape[3])
        # y = y.permute(0, 2, 1, 3, 4)
        # return y.reshape(y.shape[0], -1, y.shape[3], y.shape[4])

        b, n, h, w = x2.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__(c1, c2, k, s, p, g, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU()(self.shuf(x2))

class GSBottleneck(nn.Module):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__()
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 1, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class GSBottleneckns(GSBottleneck):
    # GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1, e=0.5):
        super().__init__(c1, c2, k, s, e)
        c_ = int(c2*e)
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConvns(c1, c_, 1, 1),
            GSConvns(c_, c2, 3, 1, act=False))
        
class GSBottleneckC(GSBottleneck):
    # cheap GS Bottleneck https://github.com/AlanLi1997/slim-neck-by-gsconv
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__(c1, c2, k, s)
        self.shortcut = DWConv(c1, c2, k, s, act=False)

class VoVGSCSP(nn.Module):
    # VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.gsb = nn.Sequential(*(GSBottleneck(c_, c_, e=1.0) for _ in range(n)))
        self.res = Conv(c_, c_, 3, 1, act=False)
        self.cv3 = Conv(2 * c_, c2, 1)

    def forward(self, x):
        x1 = self.gsb(self.cv1(x))
        y = self.cv2(x)
        return self.cv3(torch.cat((y, x1), dim=1))

class VoVGSCSPns(VoVGSCSP):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.gsb = nn.Sequential(*(GSBottleneckns(c_, c_, e=1.0) for _ in range(n)))

class VoVGSCSPC(VoVGSCSP):
    # cheap VoVGSCSP module with GSBottleneck
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2)
        c_ = int(c2 * 0.5)  # hidden channels
        self.gsb = GSBottleneckC(c_, c_, 1, 1)


class SDI(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # self.convs = nn.ModuleList([nn.Conv2d(channel, channels[0], kernel_size=3, stride=1, padding=1) for channel in channels])
        self.convs = nn.ModuleList([GSConv(channel, channels[0]) for channel in channels])

    def forward(self, xs):
        ans = torch.ones_like(xs[0])
        target_size = xs[0].shape[2:]
        for i, x in enumerate(xs):
            if x.shape[-1] > target_size[-1]:
                x = F.adaptive_avg_pool2d(x, (target_size[0], target_size[1]))
            elif x.shape[-1] < target_size[-1]:
                x = F.interpolate(x, size=(target_size[0], target_size[1]),
                                      mode='bilinear', align_corners=True)
            ans = ans * self.convs[i](x)
        return ans
    







class Fusion(nn.Module):
    def __init__(self, inc_list, fusion='bifpn') -> None:
        super().__init__()
        
        assert fusion in ['weight', 'adaptive', 'concat', 'bifpn', 'SDI']
        self.fusion = fusion
        
        if self.fusion == 'bifpn':
            self.fusion_weight = nn.Parameter(torch.ones(len(inc_list), dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()
            self.epsilon = 1e-4
        elif self.fusion == 'SDI':
            self.SDI = SDI(inc_list)
        else:
            self.fusion_conv = nn.ModuleList([Conv(inc, inc, 1) for inc in inc_list])

            if self.fusion == 'adaptive':
                self.fusion_adaptive = Conv(sum(inc_list), len(inc_list), 1)
        
    
    def forward(self, x):
        if self.fusion in ['weight', 'adaptive']:
            for i in range(len(x)):
                x[i] = self.fusion_conv[i](x[i])
        if self.fusion == 'weight':
            return torch.sum(torch.stack(x, dim=0), dim=0)
        elif self.fusion == 'adaptive':
            fusion = torch.softmax(self.fusion_adaptive(torch.cat(x, dim=1)), dim=1)
            x_weight = torch.split(fusion, [1] * len(x), dim=1)
            return torch.sum(torch.stack([x_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'concat':
            return torch.cat(x, dim=1)
        elif self.fusion == 'bifpn':
            fusion_weight = self.relu(self.fusion_weight.clone())
            fusion_weight = fusion_weight / (torch.sum(fusion_weight, dim=0))
            return torch.sum(torch.stack([fusion_weight[i] * x[i] for i in range(len(x))], dim=0), dim=0)
        elif self.fusion == 'SDI':
            return self.SDI(x)
        
###### bifpn###


 

class Fusion_module(nn.Module):
    '''
    基于注意力的自适应特征聚合 Fusion_Module
    '''

    def __init__(self, channels=64, r=4):
        super(Fusion_module, self).__init__()

        inter_channels = int(channels // r)

        self.Recalibrate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2 * channels, 2 * inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * inter_channels, 2 * channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2 * channels),
            nn.Sigmoid(),
        )

        self.channel_agg = nn.Sequential(
            nn.Conv2d(2 * channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            )

        self.local_att = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        _, c, _, _ = x1.shape
        input = torch.cat([x1, x2], dim=1)
        recal_w = self.Recalibrate(input)
        recal_input = recal_w * input ## 先对特征进行一步自校正
        recal_input = recal_input + input
        x1, x2 = torch.split(recal_input, c, dim =1)
        agg_input = self.channel_agg(recal_input) ## 进行特征压缩 因为只计算一个特征的权重
        local_w = self.local_att(agg_input)  ## 局部注意力 即spatial attention
        global_w = self.global_att(agg_input) ## 全局注意力 即channel attention
        w = self.sigmoid(local_w * global_w) ## 计算特征x1的权重
        xo = w * x1 + (1 - w) * x2 ## fusion results ## 特征聚合
        return xo
class Concat3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,c2,dimension=1):
        super().__init__()
        self.d = dimension#沿着哪个维度进行拼接
        self.Fm=Fusion_module(channels=c2)


    def forward(self, x):
        # x1=self.conv1(x[0])
        # x2=self.conv2(x[1])

        x=self.Fm(x[0],x[1])
        # x=torch.cat([x1,x2], self.d)

        return x
################空###################

class RIFusion(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, c1,r=16,dimension=1):
        super().__init__()

    def forward(self, x):
        return x
  
    
# class RIFusion(nn.Module):
#     # Concatenate a list of tensors along dimension
#     def __init__(self, c1,r=16,dimension=1):
#         super().__init__()
#         self.c1=c1*2
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(self.c1, self.c1 // r, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(self.c1 // r, self.c1, bias=False),
#             nn.Sigmoid()
#             # nn.Sigmoid(inplace=True)
#         )
#     def forward(self, x):
#         # return x
#         b, _, _, _ = x.size()
#         y = self.avg_pool(x).view(b, self.c1)
#         y = self.fc(y).view(b, self.c1, 1, 1)
  
#         x1=x*y
#         return x+torch.cat((x1[:,self.c1//2:,...],x1[:,:self.c1//2,...]),dim=1)


################SOD迁移的ACM模块###################
#如果特征图高宽不能整除 small_win，就先 pad 到能整除。
def _autopad_to_multiple(x, multiple):
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


def _remove_pad(x, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return x
    return x[..., : x.shape[-2] - pad_h, : x.shape[-1] - pad_w]

#它把当前特征图切成小窗口 token 序列。输入x: [B, C, H, W]，输出[B, Nw, win*win, C]
def _window_partition(x, win):
    b, c, h, w = x.shape
    x = x.view(b, c, h // win, win, w // win, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, (h // win) * (w // win), win * win, c)

#做完相关性增强后，增强结果还是窗口形态：[B, Nw, small_win*small_win, C]，这个函数把它拼回特征图形态：[B, C, H, W]
def _window_reverse(windows, win, H, W):
    b, num_win, tokens, c = windows.shape
    nh, nw = H // win, W // win
    x = windows.view(b, nh, nw, win, win, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, H, W)

#小窗口 query，对应另一模态更大的 search regione，大窗口是围绕小窗口扩出来的，用来覆盖位置偏移和尺度差异。
def _extract_large_windows(x, small_win, large_win):
    half = (large_win - small_win) // 2
    extra = (large_win - small_win) - half
    #先在原图周围 pad
    x_pad = F.pad(x, (half, extra, half, extra), mode="replicate")
    #以 small_win 的步长 抽取 large_win 的大窗口
    patches = F.unfold(x_pad, kernel_size=large_win, stride=small_win)  # [B, C*L*L, Nw]
    b, _, nw = patches.shape
    c = x.shape[1]
    patches = patches.transpose(1, 2).contiguous().view(b, nw, c, large_win * large_win)
    return patches.permute(0, 1, 3, 2).contiguous()  # [B, Nw, L*L, C]
#特征图 → 小窗口 token / 大窗口 token → 局部相关性 → 拼回特征图

#用高层语义引导低中层跨模态相关性，让注意力更聚焦目标而不是背景
class _LiteSemanticGuidance(nn.Module):
    def __init__(self, in_ch, top_ch=None):
        super().__init__()
        # print("LiteSemanticGuidance top_ch =", top_ch, "in_ch =", in_ch) 
        top_ch = in_ch if top_ch is None else top_ch 

        self.rgb_proj = nn.Conv2d(top_ch, in_ch, 1, bias=False)
        self.ir_proj  = nn.Conv2d(top_ch, in_ch, 1, bias=False)

        self.rgb_gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )
        self.ir_gate = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, ir_feat, rgb_top=None, ir_top=None):
        #如果没给 rgb_top / ir_top，直接返回原特征，不做引导
        if rgb_top is None or ir_top is None:
            return rgb_feat, ir_feat

        target_size = rgb_feat.shape[-2:]
        rgb_top = F.interpolate(rgb_top, size=target_size, mode="bilinear", align_corners=False)
        ir_top  = F.interpolate(ir_top,  size=target_size, mode="bilinear", align_corners=False)

        rgb_top = self.rgb_proj(rgb_top)
        ir_top  = self.ir_proj(ir_top)

        sem = torch.cat([rgb_top, ir_top], dim=1)
        return rgb_feat * self.rgb_gate(sem), ir_feat * self.ir_gate(sem)

# class _LiteSemanticGuidance(nn.Module):
#     def __init__(self, in_ch, top_ch=None):
#         super().__init__()
#         top_ch = in_ch if top_ch is None else top_ch
#         self.rgb_gate = nn.Sequential(
#             nn.Conv2d(top_ch * 2, in_ch, 1, bias=False),
#             nn.BatchNorm2d(in_ch),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(in_ch, in_ch, 1, bias=True),
#             nn.Sigmoid(),
#         )
#         self.ir_gate = nn.Sequential(
#             nn.Conv2d(top_ch * 2, in_ch, 1, bias=False),
#             nn.BatchNorm2d(in_ch),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(in_ch, in_ch, 1, bias=True),
#             nn.Sigmoid(),
#         )

#     def forward(self, rgb_feat, ir_feat, rgb_top=None, ir_top=None):
#         if rgb_top is None or ir_top is None:
#             return rgb_feat, ir_feat
#         target_size = rgb_feat.shape[-2:]
#         rgb_top = F.interpolate(rgb_top, size=target_size, mode="bilinear", align_corners=False)
#         ir_top = F.interpolate(ir_top, size=target_size, mode="bilinear", align_corners=False)
#         sem = torch.cat([rgb_top, ir_top], dim=1)
#         return rgb_feat * self.rgb_gate(sem), ir_feat * self.ir_gate(sem)

#最关键，真正的ACM计算。小窗口 query，对应另一模态更大的 search region，在该区域内做相关性增强。
#注意力计算在 token 级别，和窗口大小无关，所以能适应不同尺度的特征图。
class _AsymmetricLocalCorrelation(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, src_small, ref_large):
        b, nwin, nq, c = src_small.shape
        nk = ref_large.shape[2]
        #局部多头注意力计算
        q = self.q_proj(src_small).view(b, nwin, nq, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(ref_large).view(b, nwin, nk, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(ref_large).view(b, nwin, nk, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(b, nwin, nq, c)
        #RGB 当前局部区域，去 IR 更大邻域里找最相关的内容，再把这些 IR 信息加权聚回来增强 RGB。
        #反过来就是IR 当前局部区域，去 RGB 更大邻域里找相关内容，再增强 IR。
        return self.out_proj(out)


class DetectionACMLite(nn.Module):
    """
    输入: [rgb, ir, rgb_top, ir_top]
    输出: (rgb_out, ir_out)
    args:
        c, embed_dim, top_c, small_win, large_win, num_heads
    """
    def __init__(self, c, embed_dim=128, top_c=None, small_win=4, large_win=6, num_heads=4):
        super().__init__()
        # print("DetectionACMLite init:", c, embed_dim, top_c, small_win, large_win, num_heads)
        top_c = c if top_c is None else top_c
        self.small_win = small_win
        self.large_win = large_win

        self.sg = _LiteSemanticGuidance(c, top_c)

        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        self.rgb_from_ir = _AsymmetricLocalCorrelation(embed_dim, num_heads)
        self.ir_from_rgb = _AsymmetricLocalCorrelation(embed_dim, num_heads)

        self.rgb_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.ffn_rgb = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ffn_ir = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        # 情况1：接在 LiteAlign2 后面
        # x = [(rgb, ir), rgb_top, ir_top]
        if isinstance(x, list) and len(x) == 3 and isinstance(x[0], (list, tuple)):
            rgb, ir = x[0]
            rgb_top, ir_top = x[1], x[2]

        # 情况2：直接替换 LiteAlign2
        # x = [rgb, ir, rgb_top, ir_top]
        elif isinstance(x, list) and len(x) == 4:
            rgb, ir, rgb_top, ir_top = x[0], x[1], x[2], x[3]

        # 情况3：只给两路特征，不给 top 语义
        # x = [rgb, ir]
        elif isinstance(x, list) and len(x) == 2:
            rgb, ir = x[0], x[1]
            rgb_top, ir_top = None, None

        else:
            raise ValueError(f"DetectionACMLite got unexpected input type/len: {type(x)}, len={len(x) if isinstance(x, list) else 'NA'}")

        # print("ACM input shapes:", rgb.shape, ir.shape, rgb_top.shape, ir_top.shape)

        #保存残差
        rgb_res, ir_res = rgb, ir
        #语义引导
        rgb, ir = self.sg(rgb, ir, rgb_top, ir_top)
        #投影到embed_dim（相关性空间）
        rgb = self.rgb_in(rgb)
        ir = self.ir_in(ir)
        #pad到整除窗口大小
        rgb, pad_hw = _autopad_to_multiple(rgb, self.small_win)
        ir, _ = _autopad_to_multiple(ir, self.small_win)

        #划小窗口，抽大窗口
        B, C, H, W = rgb.shape

        rgb_small = _window_partition(rgb, self.small_win)
        ir_small = _window_partition(ir, self.small_win)

        rgb_large = _extract_large_windows(rgb, self.small_win, self.large_win)
        ir_large = _extract_large_windows(ir, self.small_win, self.large_win)

        #LayerNorm
        rgb_small_n = self.rgb_norm(rgb_small)
        ir_small_n = self.ir_norm(ir_small)
        rgb_large_n = self.rgb_norm(rgb_large)
        ir_large_n = self.ir_norm(ir_large)

        #双向非对称跨模态相关
        rgb_enh = self.rgb_from_ir(rgb_small_n, ir_large_n) + rgb_small
        ir_enh = self.ir_from_rgb(ir_small_n, rgb_large_n) + ir_small

        #拼回特征图并去 pad
        rgb_enh = _window_reverse(rgb_enh, self.small_win, H, W)
        ir_enh = _window_reverse(ir_enh, self.small_win, H, W)

        rgb_enh = _remove_pad(rgb_enh, pad_hw)
        ir_enh = _remove_pad(ir_enh, pad_hw)

        #回写特征图+ffn残差增强
        rgb_out = self.rgb_out(rgb_enh) + rgb_res
        ir_out = self.ir_out(ir_enh) + ir_res

        rgb_out = self.act(rgb_out + self.ffn_rgb(rgb_out))
        ir_out = self.act(ir_out + self.ffn_ir(ir_out))

        return (rgb_out, ir_out)


############new_ACM####################
class ERF(nn.Module):
    """
    Evidence Routing Fusion, 三路证据路由融合模块

    输入:
        1) [rgb, ir]
           - 没有外部 cross 证据时，模块内部用轻量卷积生成 cross evidence
           - 适合 P3 保守版

        2) [rgb, ir, cross]
           - cross 来自 TSCI / Selective ACM
           - 适合 P4/P5 或完整方法版

    输出:
        fused feature: [B, C, H, W]

    核心思想:
        不是简单做 RGB / IR 二路加权，而是在三类证据之间做路由:
        1) RGB 单模态证据
        2) Thermal 单模态证据
        3) Cross-modal 修正证据

        f_out = w_r * rgb + w_t * ir + w_c * cross

    设计要点:
        - q_rgb / q_ir: 单模态可靠性估计
        - u: 模态冲突程度估计
        - router: 根据可靠性和冲突状态生成三路权重
    """

    def __init__(self, c, hidden=None, init_cross_bias=-1.5):
        super().__init__()
        hidden = max(c // 4, 16) if hidden is None else hidden

        self.c = c

        # -------------------------------------------------
        # 1) 当没有外部 cross evidence 时，内部构造一个轻量 cross 分支
        #    注意：这不是强交互，只是给 ERF-only 版本一个第三路候选证据
        # -------------------------------------------------
        self.cross_fallback = nn.Sequential(
            nn.Conv2d(c * 3, hidden, 1, bias=False),   # rgb, ir, |rgb-ir|
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # -------------------------------------------------
        # 2) RGB / IR 单模态可靠性估计
        #    输出 q_r, q_t: [B,1,H,W]
        #    这里的可靠性不是类别置信度，而是“当前位置该不该信该模态”
        # -------------------------------------------------
        self.rgb_quality = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.ir_quality = nn.Sequential(
            nn.Conv2d(c, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # -------------------------------------------------
        # 3) 冲突估计分支
        #    输入 6 个低维关系图:
        #    diff_map       : RGB/IR 差异强度
        #    incons_map     : 方向不一致程度
        #    q_gap          : 两路可靠性差距
        #    rgb_energy     : RGB 响应强度
        #    ir_energy      : IR 响应强度
        #    common_map     : 正响应共激活
        #
        #    输出 u: [B,1,H,W]
        # -------------------------------------------------
        self.conflict_gate = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # -------------------------------------------------
        # 4) 三路路由器
        #    输入:
        #       q_r, q_t, u, |q_r-q_t|, diff_map, incons_map
        #    输出:
        #       logits: [B,3,H,W]
        #       对应 RGB / IR / Cross 三路得分
        # -------------------------------------------------
        self.router = nn.Sequential(
            nn.Conv2d(6, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden, 3, 1, bias=True)
        )

        # 初始化 router 最后一层:
        # - 权重为 0，开局主要靠 bias
        # - cross bias 较低，避免训练初期过度依赖 cross 证据
        last = self.router[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        with torch.no_grad():
            last.bias[2] = init_cross_bias

        # -------------------------------------------------
        # 5) 先验路由项
        #    手工提供一个非常弱的可解释先验:
        #       低冲突时更看 q_r/q_t
        #       高冲突且目标性较强时允许 cross 参与
        #    prior_scale 可学习，初始为 1
        # -------------------------------------------------
        self.prior_scale = nn.Parameter(torch.tensor(1.0))

        # -------------------------------------------------
        # 6) 输出轻量整理
        #    gamma 初始为 0，保证训练开始时不会强行改变路由融合结果
        # -------------------------------------------------
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _norm_map(x, eps=1e-6):
        """
        对每张图的单通道 map 做 0~1 归一化

        x: [B,1,H,W]
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _build_relation_maps(self, rgb, ir, q_r, q_t):
        """
        构造低维关系图，用于冲突估计和路由决策
        """
        # 1) 模态差异强度
        diff_map = torch.abs(rgb - ir).mean(dim=1, keepdim=True)
        diff_map = self._norm_map(diff_map)

        # 2) 方向不一致程度
        # cosine similarity ∈ [-1,1]，转成 [0,1] 后再取 1-sim
        cos_map = F.cosine_similarity(rgb, ir, dim=1, eps=1e-6).unsqueeze(1)
        sim01 = 0.5 * (cos_map + 1.0)
        incons_map = 1.0 - sim01.clamp(0.0, 1.0)

        # 3) 两路可靠性的差距
        q_gap = torch.abs(q_r - q_t)

        # 4) 单模态响应强度
        rgb_energy = self._norm_map(rgb.abs().mean(dim=1, keepdim=True))
        ir_energy = self._norm_map(ir.abs().mean(dim=1, keepdim=True))

        # 5) 正响应共激活
        # 这里只作为辅助线索，不把它直接等同于语义共识
        common_map = (F.relu(rgb) * F.relu(ir)).mean(dim=1, keepdim=True)
        common_map = self._norm_map(common_map)

        return diff_map, incons_map, q_gap, rgb_energy, ir_energy, common_map

    def forward(self, x):
        """
        x:
            [rgb, ir]
            或 [rgb, ir, cross]
        """
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"ERF expects list/tuple input, but got {type(x)}")

        if len(x) == 2:
            rgb, ir = x[0], x[1]
            diff = torch.abs(rgb - ir)

            # 没有外部 cross 时，内部生成轻量 cross evidence
            # cross = self.cross_fallback(torch.cat([rgb, ir, diff], dim=1))
            cross_delta = self.cross_fallback(torch.cat([rgb, ir, diff], dim=1))
            cross = 0.5 * (rgb + ir) + 0.1 * cross_delta

        elif len(x) == 3:
            rgb, ir, cross = x[0], x[1], x[2]

            # 若 cross 空间尺寸和 rgb 不同，则插值到当前尺度
            if cross.shape[-2:] != rgb.shape[-2:]:
                cross = F.interpolate(cross, size=rgb.shape[-2:], mode="nearest")

        else:
            raise ValueError(f"ERF expects 2 or 3 inputs, but got len={len(x)}")

        # -------------------------------------------------
        # Step 1: 单模态可靠性估计
        # -------------------------------------------------
        q_r = self.rgb_quality(rgb)   # [B,1,H,W]
        q_t = self.ir_quality(ir)     # [B,1,H,W]

        # -------------------------------------------------
        # Step 2: 构造差异性 / 不一致性 / 响应强度等低维关系图
        # -------------------------------------------------
        diff_map, incons_map, q_gap, rgb_energy, ir_energy, common_map = \
            self._build_relation_maps(rgb, ir, q_r, q_t)

        relation_in = torch.cat([
            diff_map,
            incons_map,
            q_gap,
            rgb_energy,
            ir_energy,
            common_map
        ], dim=1)  # [B,6,H,W]

        # -------------------------------------------------
        # Step 3: 模态冲突估计
        # -------------------------------------------------
        u = self.conflict_gate(relation_in)  # [B,1,H,W]

        # -------------------------------------------------
        # Step 4: 三路路由 logits
        # -------------------------------------------------
        route_in = torch.cat([
            q_r,
            q_t,
            u,
            q_gap,
            diff_map,
            incons_map
        ], dim=1)  # [B,6,H,W]

        learned_logits = self.router(route_in)  # [B,3,H,W]

        # 可解释先验:
        # - 低冲突区域: 根据 q_r/q_t 在 RGB 和 IR 之间路由
        # - 高冲突区域: cross 证据才更有资格参与
        objness = torch.maximum(q_r, q_t)

        z_r = q_r * (1.0 - u)
        z_t = q_t * (1.0 - u)
        z_c = objness * u

        prior_logits = torch.cat([z_r, z_t, z_c], dim=1)

        logits = learned_logits + self.prior_scale * prior_logits

        # -------------------------------------------------
        # Step 5: 三路权重
        # -------------------------------------------------
        weights = torch.softmax(logits, dim=1)

        w_r = weights[:, 0:1]
        w_t = weights[:, 1:2]
        w_c = weights[:, 2:3]

        # -------------------------------------------------
        # Step 6: 三路证据路由融合
        # -------------------------------------------------
        fused = w_r * rgb + w_t * ir + w_c * cross

        # 轻量整理，gamma=0 初始化，训练初期不破坏 fused
        out = fused + self.gamma * self.refine(fused)

        return out

# old版本，缺乏可解释性
class TSCI(nn.Module):
    """
    TSCI: Target-aware Selective Cross-modal Interaction

    输入:
        x = [rgb, ir]
        rgb: [B, C, H, W]
        ir : [B, C, H, W]

    输出:
        cross: [B, C, H, W]

    核心思想:
        1. 不再让所有窗口都执行跨模态相关性交互；
        2. 先计算窗口级“目标相关冲突分数”；
        3. 在全图窗口中统一选择 Top-K 高冲突窗口；
        4. 只对这些高冲突窗口执行 small-window / large-window 跨模态交互；
        5. 输出 cross-modal evidence，交给 ERF 做三路证据路由。

    YAML 推荐:
        - [[15, 16], 1, TSCI, [256, 128, 4, 7, 4, 0.25]]
        参数含义:
            [c, embed_dim, small_win, large_win, num_heads, topk_ratio]

    注意:
        large_win 建议使用奇数，比如 7。
        如果 large_win=6，F.unfold 的中心对齐和窗口数量会更麻烦，不建议第一版使用。
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=7,
        num_heads=4,
        topk_ratio=0.25,
        min_topk=1,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        # assert large_win % 2 == 1, \
        #     "large_win 建议为奇数，例如 7；偶数窗口会导致中心对齐不稳定"
        assert large_win >= small_win, \
            f"large_win={large_win} should be >= small_win={small_win}"

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk

        # -------------------------------------------------
        # 1) 输入投影到相关性计算空间
        # -------------------------------------------------
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        # token 级归一化，用于 attention 前稳定训练
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        # -------------------------------------------------
        # 2) 多头局部跨模态相关
        #    RGB small window -> IR large window
        #    IR  small window -> RGB large window
        # -------------------------------------------------
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # -------------------------------------------------
        # 3) 目标性 / 可靠性估计
        #    用来避免“纯背景差异大”也被当成高冲突窗口
        # -------------------------------------------------
        hidden_q = max(embed_dim // 4, 16)

        self.rgb_quality = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_q, 1, bias=False),
            nn.BatchNorm2d(hidden_q),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_q, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.ir_quality = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_q, 1, bias=False),
            nn.BatchNorm2d(hidden_q),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_q, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # -------------------------------------------------
        # 4) 窗口级冲突评分器
        #    输入 3 个窗口级量:
        #       diff_score       : 两模态差异强度
        #       incons_score     : 两模态方向不一致程度
        #       obj_score        : 目标相关性 / 至少一路模态可靠性
        # -------------------------------------------------
        self.conflict_mlp = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1, bias=True),
        )

        # -------------------------------------------------
        # 5) 将 selected windows 的双向交互结果转成 cross evidence
        #    注意这里输入是:
        #       rgb_delta_embed
        #       ir_delta_embed
        #       |rgb_delta_embed - ir_delta_embed|
        # -------------------------------------------------
        self.delta_to_c = nn.Sequential(
            nn.Conv2d(embed_dim * 3, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),

            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # 初始为 0，训练开始时 TSCI 输出约等于 0.5*(rgb+ir)，更稳定
        self.gamma = nn.Parameter(torch.zeros(1))

        # 对 cross evidence 做轻量整理
        self.cross_refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    # =====================================================
    # 工具函数
    # =====================================================
    @staticmethod
    def _pad_to_multiple(x, multiple):
        """
        将 H/W pad 到 multiple 的整数倍，避免窗口划分时尺寸不整除。
        返回:
            x_pad
            pad_hw = (pad_h, pad_w)
        """
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        return x, (pad_h, pad_w)

    @staticmethod
    def _remove_pad(x, pad_hw):
        """
        去掉 _pad_to_multiple 增加的 padding。
        """
        pad_h, pad_w = pad_hw
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

    @staticmethod
    def _window_partition(x, win):
        """
        非重叠小窗口划分。

        输入:
            x: [B, C, H, W]

        输出:
            windows: [B, L, win*win, C]
            gh, gw : 小窗口网格数
        """
        B, C, H, W = x.shape
        assert H % win == 0 and W % win == 0, \
            f"H={H}, W={W} must be divisible by win={win}"

        gh, gw = H // win, W // win

        x = x.view(B, C, gh, win, gw, win)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(B, gh * gw, win * win, C)

        return windows, gh, gw

    @staticmethod
    def _window_reverse(windows, gh, gw, win):
        """
        将窗口序列还原为二维特征图。

        输入:
            windows: [B, L, win*win, C]

        输出:
            x: [B, C, H, W]
        """
        B, L, N, C = windows.shape
        assert L == gh * gw
        assert N == win * win

        x = windows.view(B, gh, gw, win, win, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, gh * win, gw * win)

        return x

    # @staticmethod
    # def _extract_large_windows(x, small_win, large_win):
    #     """
    #     以 small_win 为步长，在每个小窗口中心附近提取另一模态 large_win x large_win 大邻域。

    #     输入:
    #         x: [B, C, H, W]

    #     输出:
    #         patches: [B, L, large_win*large_win, C]
    #     """
    #     B, C, H, W = x.shape
    #     pad = large_win // 2

    #     # 对 odd large_win:
    #     # H 可被 small_win 整除时，unfold 后窗口数量正好是 H/small_win * W/small_win
    #     patches = F.unfold(
    #         x,
    #         kernel_size=large_win,
    #         stride=small_win,
    #         padding=pad
    #     )  # [B, C*K, L]

    #     patches = patches.transpose(1, 2).contiguous()  # [B, L, C*K]
    #     patches = patches.view(B, patches.shape[1], C, large_win * large_win)
    #     patches = patches.permute(0, 1, 3, 2).contiguous()  # [B, L, K, C]

    #     return patches
    @staticmethod
    def _extract_large_windows(x, small_win, large_win):
        """
        以 small_win 为步长，提取 large_win x large_win 大邻域。

        支持 large_win 为偶数，例如 small_win=4, large_win=6。

        设计含义：
        large_win 不是以单个像素为中心的奇数卷积核，
        而是围绕当前 small window 扩展出的更大邻域。
        例如 small_win=4, large_win=6 表示：
        在 4x4 当前窗口周围额外引入少量上下文，形成 6x6 邻域。
        """
        B, C, H, W = x.shape

        assert large_win >= small_win, \
            f"large_win={large_win} should be >= small_win={small_win}"

        # 为了保证 unfold 后的窗口数等于 H/small_win * W/small_win，
        # 需要满足: H + pad_total - large_win = H - small_win
        # 因此 pad_total = large_win - small_win
        pad_total = large_win - small_win
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        patches = F.unfold(
            x,
            kernel_size=large_win,
            stride=small_win,
            padding=0
        )  # [B, C*K, L]

        patches = patches.transpose(1, 2).contiguous()  # [B, L, C*K]
        patches = patches.view(B, patches.shape[1], C, large_win * large_win)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # [B, L, K, C]

        return patches

    @staticmethod
    def _norm_01(x, eps=1e-6):
        """
        对 [B, L] 或 [B,1,H,W] 做 per-sample 0~1 归一化。
        """
        if x.ndim == 2:
            x_min = x.amin(dim=1, keepdim=True)
            x_max = x.amax(dim=1, keepdim=True)
        elif x.ndim == 4:
            x_min = x.amin(dim=(2, 3), keepdim=True)
            x_max = x.amax(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unsupported x.ndim={x.ndim}")

        return (x - x_min) / (x_max - x_min + eps)

    def _get_topk_num(self, num_windows):
        """
        根据 topk_ratio 得到 top-k 数量。
        - topk_ratio <= 0: 不选择窗口，TSCI 退化为 base cross
        - 0 < topk_ratio <= 1: 按比例选择
        - topk_ratio > 1: 按绝对数量选择
        """
        if self.topk_ratio <= 0:
            return 0

        if self.topk_ratio <= 1:
            k = int(round(num_windows * self.topk_ratio))
        else:
            k = int(self.topk_ratio)

        k = max(self.min_topk, k)
        k = min(k, num_windows)
        return k

    def _local_cross_attention(self, q_win, kv_win):
        """
        对 selected windows 做局部多头跨模态 attention。

        输入:
            q_win : [K, Nq, C]
            kv_win: [K, Nk, C]

        输出:
            out: [K, Nq, C]
        """
        K, Nq, C = q_win.shape
        Nk = kv_win.shape[1]

        q = self.q_proj(q_win)
        k = self.k_proj(kv_win)
        v = self.v_proj(kv_win)

        q = q.view(K, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(K, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(K, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(K, Nq, C)

        return self.o_proj(out)

    def _compute_conflict_score(self, rgb_small, ir_small, rgb_e, ir_e, gh, gw):
        """
        计算窗口级目标相关冲突分数。

        输入:
            rgb_small, ir_small: [B, L, Ns, E]
            rgb_e, ir_e        : [B, E, H, W]

        输出:
            conflict: [B, L]
        """
        # -------------------------------------------------
        # 1) 差异性: 窗口内 RGB/IR 的 L1 差异
        # -------------------------------------------------
        diff_score = torch.abs(rgb_small - ir_small).mean(dim=(2, 3))  # [B,L]
        diff_score = self._norm_01(diff_score)

        # -------------------------------------------------
        # 2) 不一致性: 用余弦方向差异替代简单相乘
        # -------------------------------------------------
        rgb_vec = rgb_small.mean(dim=2)  # [B,L,E]
        ir_vec = ir_small.mean(dim=2)    # [B,L,E]

        cos_score = F.cosine_similarity(rgb_vec, ir_vec, dim=-1, eps=1e-6)  # [-1,1]
        sim01 = 0.5 * (cos_score + 1.0)
        incons_score = 1.0 - sim01.clamp(0.0, 1.0)  # [B,L]

        # -------------------------------------------------
        # 3) 目标相关性 / 可靠性:
        #    至少一路模态看起来像目标相关区域，才更值得进入交互
        # -------------------------------------------------
        q_rgb = self.rgb_quality(rgb_e)  # [B,1,H,W]
        q_ir = self.ir_quality(ir_e)     # [B,1,H,W]

        q_rgb_win = F.avg_pool2d(
            q_rgb,
            kernel_size=self.small_win,
            stride=self.small_win
        ).view(q_rgb.shape[0], -1)  # [B,L]

        q_ir_win = F.avg_pool2d(
            q_ir,
            kernel_size=self.small_win,
            stride=self.small_win
        ).view(q_ir.shape[0], -1)   # [B,L]

        obj_score = torch.maximum(q_rgb_win, q_ir_win)  # [B,L]

        # -------------------------------------------------
        # 4) MLP 生成冲突分数
        #    输入: [差异性, 不一致性, 目标性]
        # -------------------------------------------------
        conflict_in = torch.stack(
            [diff_score, incons_score, obj_score],
            dim=-1
        )  # [B,L,3]

        conflict = torch.sigmoid(self.conflict_mlp(conflict_in)).squeeze(-1)  # [B,L]

        # 目标性作为外部门控，避免纯背景伪冲突被选中
        conflict = conflict * obj_score

        return conflict

    # =====================================================
    # forward
    # =====================================================
    def forward(self, x):
        """
        x = [rgb, ir]

        返回:
            cross evidence: [B, C, H, W]
        """
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                f"TSCI expects input [rgb, ir], but got type={type(x)}, "
                f"len={len(x) if isinstance(x, (list, tuple)) else 'NA'}"
            )

        rgb, ir = x[0], x[1]
        assert rgb.shape == ir.shape, \
            f"TSCI expects rgb and ir with same shape, got {rgb.shape} and {ir.shape}"

        B, C, H0, W0 = rgb.shape

        # -------------------------------------------------
        # Step 0: 基础 cross evidence
        # 低冲突区域即使不做 ACM，cross 也不是 0，而是保守的平均证据。
        # ERF 后续会决定是否使用 cross。
        # -------------------------------------------------
        base_cross = 0.5 * (rgb + ir)

        # -------------------------------------------------
        # Step 1: 投影到嵌入空间
        # -------------------------------------------------
        rgb_e = self.rgb_in(rgb)  # [B,E,H,W]
        ir_e = self.ir_in(ir)

        # -------------------------------------------------
        # Step 2: pad 到 small_win 的整数倍
        # -------------------------------------------------
        rgb_e, pad_hw = self._pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = self._pad_to_multiple(ir_e, self.small_win)

        _, E, H, W = rgb_e.shape

        # -------------------------------------------------
        # Step 3: 小窗口与大邻域窗口
        # -------------------------------------------------
        rgb_small, gh, gw = self._window_partition(rgb_e, self.small_win)
        ir_small, _, _ = self._window_partition(ir_e, self.small_win)

        rgb_large = self._extract_large_windows(
            rgb_e, self.small_win, self.large_win
        )
        ir_large = self._extract_large_windows(
            ir_e, self.small_win, self.large_win
        )

        # 检查 large window 数量是否和 small window 数量一致
        # 如果这里报错，通常是 large_win 不是奇数导致的。
        assert rgb_large.shape[1] == rgb_small.shape[1], \
            f"large windows L={rgb_large.shape[1]} != small windows L={rgb_small.shape[1]}; " \
            f"please use odd large_win, e.g. 7"

        L = rgb_small.shape[1]

        # -------------------------------------------------
        # Step 4: 窗口级冲突评分
        # -------------------------------------------------
        conflict = self._compute_conflict_score(
            rgb_small, ir_small, rgb_e, ir_e, gh, gw
        )  # [B,L]

        # -------------------------------------------------
        # Step 5: 全图统一 Top-K 选择高冲突窗口
        # -------------------------------------------------
        k = self._get_topk_num(L)

        rgb_delta_win = torch.zeros_like(rgb_small)  # [B,L,Ns,E]
        ir_delta_win = torch.zeros_like(ir_small)

        if k > 0:
            topk_idx = torch.topk(
                conflict,
                k=k,
                dim=1,
                largest=True,
                sorted=False
            ).indices  # [B,K]

            # 逐 batch 处理，保证每张图按自己的冲突分布选 top-k
            for b in range(B):
                sel = topk_idx[b]  # [K]

                # 取高冲突窗口
                rgb_q = rgb_small[b, sel]   # [K,Ns,E]
                ir_q = ir_small[b, sel]

                rgb_kv = rgb_large[b, sel]  # [K,Nl,E]
                ir_kv = ir_large[b, sel]

                # LayerNorm 稳定 token 表达
                rgb_q = self.rgb_norm(rgb_q)
                ir_q = self.ir_norm(ir_q)
                rgb_kv = self.rgb_norm(rgb_kv)
                ir_kv = self.ir_norm(ir_kv)

                # RGB 从 IR 大邻域取回补充证据
                rgb_delta_win[b, sel] = self._local_cross_attention(rgb_q, ir_kv)

                # IR 从 RGB 大邻域取回补充证据
                ir_delta_win[b, sel] = self._local_cross_attention(ir_q, rgb_kv)

        # -------------------------------------------------
        # Step 6: 窗口更新还原为二维特征图
        # -------------------------------------------------
        #只有 Top-K 窗口有跨模态 attention 结果；未选中的窗口不是原特征，而是 0。
        #原始 RGB 和 Thermal 特征并没有在 TSCI 内被修改，它们会原样和 TSCI 生成的 cross evidence 一起送入 ERF。
        rgb_delta = self._window_reverse(
            rgb_delta_win, gh, gw, self.small_win
        )  # [B,E,H,W]

        ir_delta = self._window_reverse(
            ir_delta_win, gh, gw, self.small_win
        )  # [B,E,H,W]

        # 去掉 padding，恢复到输入尺度
        rgb_delta = self._remove_pad(rgb_delta, pad_hw)
        ir_delta = self._remove_pad(ir_delta, pad_hw)

        # -------------------------------------------------
        # Step 7: 生成 cross-modal 修正证据
        # -------------------------------------------------
        delta_cat = torch.cat(
            [
                rgb_delta,
                ir_delta,
                torch.abs(rgb_delta - ir_delta)
            ],
            dim=1
        )  # [B,3E,H,W]

        cross_delta = self.delta_to_c(delta_cat)  # [B,C,H,W]

        # 用窗口级 conflict 生成一个空间门控图，只在高冲突区域强化 cross_delta
        conflict_map = conflict.view(B, 1, gh, gw)
        conflict_map = F.interpolate(
            conflict_map,
            size=(H, W),
            mode="nearest"
        )
        conflict_map = self._remove_pad(conflict_map, pad_hw)

        # base_cross 保证低冲突区域也有稳定 cross 证据；
        # gamma 初始为 0，训练初期不会强行引入 cross_delta。
        cross = base_cross + self.gamma * conflict_map * cross_delta

        # 轻量整理，beta 初始为 0，不破坏初始稳定性
        cross = cross + self.beta * self.cross_refine(cross)

        return cross

class TSCF(nn.Module):
    """
    TSCF: Target-aware Selective Cross-modal Fusion

    输入:
        x = [rgb, ir]
        rgb: [B, C, H, W]
        ir : [B, C, H, W]

    输出:
        fused: [B, C, H, W]

    核心思想:
        1. 先计算窗口级冲突分数，只选择 Top-K 高冲突窗口；
        2. 只在这些窗口上执行双向跨模态 attention；
        3. 将 RGB<-IR 和 IR<-RGB 两个方向的更新量分别回写到原模态；
        4. 根据跨模态 attention 的匹配置信度融合两路增强特征；
        5. 不再构造第三路 cross evidence，也不再使用 ERF 三路路由。
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=6,
        num_heads=4,
        topk_ratio=0.25,
        min_topk=1,
        tau=0.7,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        assert large_win >= small_win, \
            f"large_win={large_win} should be >= small_win={small_win}"

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.topk_ratio = topk_ratio
        self.min_topk = min_topk
        self.tau = tau

        # -----------------------------
        # 1. 输入投影到 attention 空间
        # -----------------------------
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        # -----------------------------
        # 2. 多头局部跨模态 attention
        # -----------------------------
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # -----------------------------
        # 3. 窗口响应门控
        # 注意：这里不再强称 objectness，只作为有效响应代理
        # -----------------------------
        hidden_q = max(embed_dim // 4, 16)

        self.rgb_quality = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_q, 1, bias=False),
            nn.BatchNorm2d(hidden_q),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_q, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        self.ir_quality = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_q, 1, bias=False),
            nn.BatchNorm2d(hidden_q),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_q, 1, 3, padding=1, bias=True),
            nn.Sigmoid()
        )

        # -----------------------------
        # 4. 窗口级冲突评分
        # 输入: diff_score, incons_score, response_gate
        # -----------------------------
        self.conflict_mlp = nn.Sequential(
            nn.Linear(3, 16, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1, bias=True),
        )

        # -----------------------------
        # 5. 将两个方向的 embed 更新量映射回原始通道 C
        # 不再合成 cross evidence，而是分别回写到 RGB / IR
        # -----------------------------
        self.rgb_delta_to_c = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.ir_delta_to_c = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # -----------------------------
        # 6. 更新强度
        # 初始化为 0，训练初期退化为 0.5*(rgb+ir)
        # -----------------------------
        self.lambda_rgb = nn.Parameter(torch.zeros(1))
        self.lambda_ir = nn.Parameter(torch.zeros(1))

        # -----------------------------
        # 7. attention-confidence fusion 强度
        # 初始化为 0，训练初期两路等权融合
        # -----------------------------
        self.conf_scale = nn.Parameter(torch.zeros(1))

    # =====================================================
    # 工具函数
    # =====================================================
    @staticmethod
    def _pad_to_multiple(x, multiple):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        return x, (pad_h, pad_w)

    @staticmethod
    def _remove_pad(x, pad_hw):
        pad_h, pad_w = pad_hw
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

    @staticmethod
    def _window_partition(x, win):
        B, C, H, W = x.shape
        assert H % win == 0 and W % win == 0, \
            f"H={H}, W={W} must be divisible by win={win}"

        gh, gw = H // win, W // win

        x = x.view(B, C, gh, win, gw, win)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(B, gh * gw, win * win, C)

        return windows, gh, gw

    @staticmethod
    def _window_reverse(windows, gh, gw, win):
        B, L, N, C = windows.shape
        assert L == gh * gw
        assert N == win * win

        x = windows.view(B, gh, gw, win, win, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, gh * win, gw * win)

        return x

    @staticmethod
    def _extract_large_windows(x, small_win, large_win):
        """
        支持 small_win=4, large_win=6 或 7。
        large_win 表示围绕当前 small window 扩展出的更大邻域。
        """
        B, C, H, W = x.shape

        assert large_win >= small_win, \
            f"large_win={large_win} should be >= small_win={small_win}"

        pad_total = large_win - small_win
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        pad_top = pad_total // 2
        pad_bottom = pad_total - pad_top

        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        patches = F.unfold(
            x,
            kernel_size=large_win,
            stride=small_win,
            padding=0
        )  # [B, C*K, L]

        patches = patches.transpose(1, 2).contiguous()  # [B, L, C*K]
        patches = patches.view(B, patches.shape[1], C, large_win * large_win)
        patches = patches.permute(0, 1, 3, 2).contiguous()  # [B, L, K, C]

        return patches

    @staticmethod
    def _norm_01(x, eps=1e-6):
        if x.ndim == 2:
            x_min = x.amin(dim=1, keepdim=True)
            x_max = x.amax(dim=1, keepdim=True)
        elif x.ndim == 4:
            x_min = x.amin(dim=(2, 3), keepdim=True)
            x_max = x.amax(dim=(2, 3), keepdim=True)
        else:
            raise ValueError(f"Unsupported x.ndim={x.ndim}")

        return (x - x_min) / (x_max - x_min + eps)

    def _get_topk_num(self, num_windows):
        if self.topk_ratio <= 0:
            return 0

        if self.topk_ratio <= 1:
            k = int(round(num_windows * self.topk_ratio))
        else:
            k = int(self.topk_ratio)

        k = max(self.min_topk, k)
        k = min(k, num_windows)
        return k

    def _local_cross_attention_with_conf(self, q_win, kv_win):
        """
        对 selected windows 做局部多头跨模态 attention，
        同时返回 attention 匹配置信度。

        输入:
            q_win : [K, Nq, C]
            kv_win: [K, Nk, C]

        输出:
            out       : [K, Nq, C]
            conf_token: [K, Nq, 1]
        """
        K, Nq, C = q_win.shape
        Nk = kv_win.shape[1]

        q = self.q_proj(q_win)
        k = self.k_proj(kv_win)
        v = self.v_proj(kv_win)

        q = q.view(K, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(K, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(K, Nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)  # [K, heads, Nq, Nk]

        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(K, Nq, C)
        out = self.o_proj(out)

        # -----------------------------
        # attention 置信度
        # 熵越低，说明注意力越集中，匹配越明确
        # -----------------------------
        eps = 1e-6
        entropy = -(attn * torch.log(attn + eps)).sum(dim=-1)  # [K, heads, Nq]
        entropy = entropy / torch.log(torch.tensor(float(Nk), device=attn.device, dtype=attn.dtype) + eps)

        conf = 1.0 - entropy               # [K, heads, Nq]
        conf = conf.mean(dim=1).unsqueeze(-1)  # [K, Nq, 1]
        conf = conf.clamp(0.0, 1.0)

        return out, conf

    def _compute_conflict_score(self, rgb_small, ir_small, rgb_e, ir_e):
        """
        计算窗口级冲突分数。
        注意：这里的 response_gate 不是真实 objectness，
        而是窗口有效响应门控。
        """
        # 1. 幅值差异
        diff_score = torch.abs(rgb_small - ir_small).mean(dim=(2, 3))  # [B,L]
        diff_score = self._norm_01(diff_score)

        # 2. 方向不一致性
        rgb_vec = rgb_small.mean(dim=2)  # [B,L,E]
        ir_vec = ir_small.mean(dim=2)    # [B,L,E]

        cos_score = F.cosine_similarity(rgb_vec, ir_vec, dim=-1, eps=1e-6)
        sim01 = 0.5 * (cos_score + 1.0)
        incons_score = 1.0 - sim01.clamp(0.0, 1.0)  # [B,L]

        # 3. 窗口有效响应门控
        q_rgb = self.rgb_quality(rgb_e)  # [B,1,H,W]
        q_ir = self.ir_quality(ir_e)     # [B,1,H,W]

        q_rgb_win = F.avg_pool2d(
            q_rgb,
            kernel_size=self.small_win,
            stride=self.small_win
        ).view(q_rgb.shape[0], -1)

        q_ir_win = F.avg_pool2d(
            q_ir,
            kernel_size=self.small_win,
            stride=self.small_win
        ).view(q_ir.shape[0], -1)

        # 仍然保留 max 思路：只要一路有有效响应，就允许参与冲突判断
        response_gate = torch.maximum(q_rgb_win, q_ir_win)

        conflict_in = torch.stack(
            [diff_score, incons_score, response_gate],
            dim=-1
        )  # [B,L,3]

        conflict = torch.sigmoid(self.conflict_mlp(conflict_in)).squeeze(-1)

        # response_gate 作为外部门控，抑制低响应背景窗口
        conflict = conflict * response_gate

        return conflict

    # =====================================================
    # forward
    # =====================================================
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                f"TSCF expects input [rgb, ir], but got type={type(x)}, "
                f"len={len(x) if isinstance(x, (list, tuple)) else 'NA'}"
            )

        rgb, ir = x[0], x[1]
        assert rgb.shape == ir.shape, \
            f"TSCF expects rgb and ir with same shape, got {rgb.shape} and {ir.shape}"

        B, C, H0, W0 = rgb.shape

        # -----------------------------
        # Step 1: 投影到 embed 空间
        # -----------------------------
        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)

        # -----------------------------
        # Step 2: pad 到 small_win 整数倍
        # -----------------------------
        rgb_e, pad_hw = self._pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = self._pad_to_multiple(ir_e, self.small_win)

        _, E, H, W = rgb_e.shape

        # -----------------------------
        # Step 3: 小窗口 / 大邻域窗口
        # -----------------------------
        rgb_small, gh, gw = self._window_partition(rgb_e, self.small_win)
        ir_small, _, _ = self._window_partition(ir_e, self.small_win)

        rgb_large = self._extract_large_windows(rgb_e, self.small_win, self.large_win)
        ir_large = self._extract_large_windows(ir_e, self.small_win, self.large_win)

        assert rgb_large.shape[1] == rgb_small.shape[1], \
            f"large windows L={rgb_large.shape[1]} != small windows L={rgb_small.shape[1]}"

        L = rgb_small.shape[1]

        # -----------------------------
        # Step 4: 窗口级冲突评分
        # -----------------------------
        conflict = self._compute_conflict_score(
            rgb_small, ir_small, rgb_e, ir_e
        )  # [B,L]

        k = self._get_topk_num(L)

        # 更新量窗口，未选中窗口保持 0
        rgb_delta_win = torch.zeros_like(rgb_small)  # [B,L,Ns,E]
        ir_delta_win = torch.zeros_like(ir_small)

        # attention 置信度窗口，未选中窗口为 0
        rgb_conf_win = torch.zeros(
            B, L, self.small_win * self.small_win, 1,
            device=rgb.device,
            dtype=rgb.dtype
        )
        ir_conf_win = torch.zeros_like(rgb_conf_win)

        # -----------------------------
        # Step 5: 只对 Top-K 窗口做双向跨模态 attention
        # -----------------------------
        if k > 0:
            topk_idx = torch.topk(
                conflict,
                k=k,
                dim=1,
                largest=True,
                sorted=False
            ).indices  # [B,K]

            for b in range(B):
                sel = topk_idx[b]

                rgb_q = rgb_small[b, sel]   # [K,Ns,E]
                ir_q = ir_small[b, sel]

                rgb_kv = rgb_large[b, sel]  # [K,Nl,E]
                ir_kv = ir_large[b, sel]

                rgb_q = self.rgb_norm(rgb_q)
                ir_q = self.ir_norm(ir_q)
                rgb_kv = self.rgb_norm(rgb_kv)
                ir_kv = self.ir_norm(ir_kv)

                # RGB 从 IR 中取回补充信息
                rgb_out, rgb_conf = self._local_cross_attention_with_conf(rgb_q, ir_kv)

                # IR 从 RGB 中取回补充信息
                ir_out, ir_conf = self._local_cross_attention_with_conf(ir_q, rgb_kv)

                # rgb_delta_win[b, sel] = rgb_out
                # ir_delta_win[b, sel] = ir_out
                rgb_delta_win[b, sel] = rgb_out.to(dtype=rgb_delta_win.dtype)
                ir_delta_win[b, sel] = ir_out.to(dtype=ir_delta_win.dtype)

                # rgb_conf_win[b, sel] = rgb_conf
                # ir_conf_win[b, sel] = ir_conf
                rgb_conf_win[b, sel] = rgb_conf.to(dtype=rgb_conf_win.dtype)
                ir_conf_win[b, sel] = ir_conf.to(dtype=ir_conf_win.dtype)

        # -----------------------------
        # Step 6: 窗口更新量还原为二维特征图
        # -----------------------------
        rgb_delta = self._window_reverse(rgb_delta_win, gh, gw, self.small_win)
        ir_delta = self._window_reverse(ir_delta_win, gh, gw, self.small_win)

        rgb_delta = self._remove_pad(rgb_delta, pad_hw)
        ir_delta = self._remove_pad(ir_delta, pad_hw)

        # attention 置信度图
        rgb_conf_map = self._window_reverse(rgb_conf_win, gh, gw, self.small_win)
        ir_conf_map = self._window_reverse(ir_conf_win, gh, gw, self.small_win)

        rgb_conf_map = self._remove_pad(rgb_conf_map, pad_hw)
        ir_conf_map = self._remove_pad(ir_conf_map, pad_hw)

        rgb_conf_map = rgb_conf_map.to(dtype=rgb.dtype)
        ir_conf_map = ir_conf_map.to(dtype=rgb.dtype)

        # 冲突图
        conflict_map = conflict.view(B, 1, gh, gw)
        conflict_map = F.interpolate(conflict_map, size=(H, W), mode="nearest")
        conflict_map = self._remove_pad(conflict_map, pad_hw)
        conflict_map = conflict_map.to(dtype=rgb.dtype)

        # -----------------------------
        # Step 7: 将两个方向的更新量分别映射回 C 通道
        # -----------------------------
        rgb_update = self.rgb_delta_to_c(rgb_delta)
        ir_update = self.ir_delta_to_c(ir_delta)

        # -----------------------------
        # Step 8: 回写到原始两路特征，形成 cross-aware RGB / IR
        # -----------------------------
        rgb_enh = rgb + self.lambda_rgb * conflict_map * rgb_update
        ir_enh = ir + self.lambda_ir * conflict_map * ir_update

        # -----------------------------
        # Step 9: attention-confidence fusion
        # 置信度越高，说明该方向跨模态匹配越明确
        # conf_scale 初始为 0，因此初始为等权融合
        # -----------------------------
        fusion_logits = torch.cat([rgb_conf_map, ir_conf_map], dim=1)  # [B,2,H,W]
        fusion_logits = self.conf_scale * fusion_logits / max(self.tau, 1e-6)

        weights = torch.softmax(fusion_logits, dim=1)
        w_rgb = weights[:, 0:1]
        w_ir = weights[:, 1:2]

        fused = w_rgb * rgb_enh + w_ir * ir_enh

        return fused

class _IALocalCorrelationWithConf(nn.Module):
    """
    Asymmetric local cross-modal attention with interpretable confidence.

    输入:
        src_small: [B, L, Ns, C]
        ref_large: [B, L, Nl, C]

    输出:
        out:       [B, L, Ns, C]
        conf:      [B, L, Ns, 1]  attention 熵置信度
        attn_mean: [B, L, Ns, Nl] 跨 head 平均后的 attention，用于计算 match quality
    """
    def __init__(self, dim, num_heads=4, eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, src_small, ref_large):
        B, L, Ns, C = src_small.shape
        Nl = ref_large.shape[2]

        q = self.q_proj(src_small).view(B, L, Ns, self.num_heads, self.head_dim)
        k = self.k_proj(ref_large).view(B, L, Nl, self.num_heads, self.head_dim)
        v = self.v_proj(ref_large).view(B, L, Nl, self.num_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)  # [B,L,H,Ns,D]
        k = k.permute(0, 1, 3, 2, 4)  # [B,L,H,Nl,D]
        v = v.permute(0, 1, 3, 2, 4)  # [B,L,H,Nl,D]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)   # [B,L,H,Ns,Nl]

        out = attn @ v                # [B,L,H,Ns,D]
        out = out.permute(0, 1, 3, 2, 4).contiguous().view(B, L, Ns, C)
        out = self.out_proj(out)

        # attention-confidence：越集中，说明匹配越明确
        attn_mean = attn.mean(dim=2)  # [B,L,Ns,Nl]
        entropy = -(attn_mean * (attn_mean + self.eps).log()).sum(dim=-1, keepdim=True)
        conf = 1.0 - entropy / math.log(float(Nl))
        conf = conf.clamp(0.0, 1.0)   # [B,L,Ns,1]

        return out, conf, attn_mean


class IACMFusion(nn.Module):
    """
    IACMFusion: Interpretable Asymmetric Cross-modal Fusion

    核心:
    1) 保留大小窗口跨模态 attention；
    2) 不使用小网络预测融合权重；
    3) 使用 Qr、Qi、U、rho 解析式计算融合权重；
    4) 不包含 Mamba / VMamba。

    输入:
        x = [rgb, ir]
        rgb: [B,C,H,W]
        ir : [B,C,H,W]

    输出:
        fused: [B,C,H,W]
    """
    def __init__(
        self,
        c,
        embed_dim=64,
        small_win=4,
        large_win=6,
        num_heads=4,
        grad_lambda=0.5,
        eps=1e-6,
    ):
        super().__init__()
        assert large_win >= small_win, "large_win should be >= small_win"

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.grad_lambda = grad_lambda
        self.eps = eps

        # 投影到 attention 空间
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        self.rgb_from_ir = _IALocalCorrelationWithConf(embed_dim, num_heads, eps)
        self.ir_from_rgb = _IALocalCorrelationWithConf(embed_dim, num_heads, eps)

        # attention 输出映射回原始通道

        # v1 BN
        # self.rgb_delta_out = nn.Sequential(
        #     nn.Conv2d(embed_dim, c, 1, bias=False),
        #     nn.BatchNorm2d(c),
        # )
        # self.ir_delta_out = nn.Sequential(
        #     nn.Conv2d(embed_dim, c, 1, bias=False),
        #     nn.BatchNorm2d(c),
        # )

        # v2 BN 可能破坏 Top-k scatter 后的稀疏性：未选中的窗口本来是 0，但经过 BN 后不一定还是 0。
        self.rgb_delta_out = nn.Conv2d(embed_dim, c, 1, bias=False)
        self.ir_delta_out = nn.Conv2d(embed_dim, c, 1, bias=False)

        # 残差强度初始化为 0，训练初期不破坏原特征
        self.gamma_r = nn.Parameter(torch.zeros(1))
        self.gamma_i = nn.Parameter(torch.zeros(1))

        # 解析式融合权重的启用强度。
        # 不是黑盒路由，只是从 0.5/0.5 平滑过渡到解析式权重。
        self.weight_alpha_raw = nn.Parameter(torch.tensor(-2.0))

        self.out_refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    # -------------------------------------------------
    # basic helpers
    # -------------------------------------------------
    @staticmethod
    def _norm01(x, eps=1e-6):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + eps)

    def _pad_to_multiple(self, x, multiple):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (H, W)

    @staticmethod
    def _remove_pad(x, hw):
        H, W = hw
        return x[:, :, :H, :W]

    def _window_partition(self, x, win):
        B, C, H, W = x.shape
        assert H % win == 0 and W % win == 0
        gh, gw = H // win, W // win

        x = x.view(B, C, gh, win, gw, win)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, gh * gw, win * win, C)
        return x, gh, gw

    def _window_reverse(self, x_win, gh, gw, win):
        B, L, N, C = x_win.shape
        assert L == gh * gw
        assert N == win * win

        x = x_win.view(B, gh, gw, win, win, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, gh * win, gw * win)
        return x

    def _extract_large_windows(self, x, small_win, large_win):
        """
        使用非对称 padding + unfold 提取 large window。
        输出: [B, L, large_win*large_win, C]
        """
        B, C, H, W = x.shape
        total_pad = large_win - small_win
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        patches = F.unfold(
            x_pad,
            kernel_size=large_win,
            stride=small_win,
            padding=0,
        )  # [B, C*large*large, L]

        L = patches.shape[-1]
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(B, L, C, large_win * large_win)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches

    def _grad_mag(self, x):
        """
        x: [B,1,H,W]
        """
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    def _local_smooth(self, x, k):
        # 使用奇数核避免 even kernel 带来的尺寸偏移
        kk = k if k % 2 == 1 else k + 1
        return F.avg_pool2d(x, kernel_size=kk, stride=1, padding=kk // 2)

    # -------------------------------------------------
    # interpretable maps
    # -------------------------------------------------
    def _rgb_quality(self, rgb):
        """
        Feature-level RGB quality proxy.
        第一版不改数据流，所以这里用 RGB 分支特征的局部响应和局部梯度近似可见质量。
        后续如果传入原图，可以替换为 LAB-L / gray 的局部亮度质量。
        """
        resp = rgb.abs().mean(dim=1, keepdim=True)
        resp = self._norm01(resp, self.eps)

        grad = self._grad_mag(resp)
        grad = self._norm01(grad, self.eps)

        q = resp * (1.0 + self.grad_lambda * grad)
        q = self._local_smooth(q, self.small_win)
        q = self._norm01(q, self.eps)
        return q.clamp(0.0, 1.0)

    def _ir_quality(self, ir):
        """
        IR quality proxy:
        使用局部热显著性 + 热结构。
        避免只要热响应强就盲目信 IR。
        """
        resp = ir.abs().mean(dim=1, keepdim=True)
        resp = self._norm01(resp, self.eps)

        local_small = self._local_smooth(resp, self.small_win)
        local_large = self._local_smooth(resp, self.large_win)
        saliency = torch.abs(local_small - local_large)
        saliency = self._norm01(saliency, self.eps)

        grad = self._grad_mag(resp)
        grad = self._norm01(grad, self.eps)

        q = saliency * (1.0 + self.grad_lambda * grad)
        q = self._local_smooth(q, self.small_win)
        q = self._norm01(q, self.eps)
        return q.clamp(0.0, 1.0)

    def _conflict_map(self, rgb, ir):
        """
        U: 模态冲突图。
        同时考虑幅值差异和方向不一致。
        """
        diff = torch.abs(rgb - ir).mean(dim=1, keepdim=True)
        diff = self._norm01(diff, self.eps)

        cos = F.cosine_similarity(rgb, ir, dim=1, eps=self.eps).unsqueeze(1)
        incons = (1.0 - cos) * 0.5
        incons = self._norm01(incons, self.eps)

        u = diff * incons
        u = self._local_smooth(u, self.small_win)
        u = self._norm01(u, self.eps)
        return u.clamp(0.0, 1.0)

    # -------------------------------------------------
    # forward
    # -------------------------------------------------
    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(f"IACMFusion expects [rgb, ir], but got {type(x)}")

        rgb, ir = x[0], x[1]
        assert rgb.shape == ir.shape, f"rgb and ir must have same shape, got {rgb.shape}, {ir.shape}"

        B, C, H0, W0 = rgb.shape

        # 原始特征 pad，供质量图和最终门控使用
        rgb_pad, pad_hw = self._pad_to_multiple(rgb, self.small_win)
        ir_pad, _ = self._pad_to_multiple(ir, self.small_win)

        # 质量图与冲突图，均是可解释空间矩阵
        Qr = self._rgb_quality(rgb_pad)       # [B,1,H,W]
        Qi = self._ir_quality(ir_pad)         # [B,1,H,W]
        U = self._conflict_map(rgb_pad, ir_pad)

        # 投影到 attention 空间
        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)

        rgb_e, _ = self._pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = self._pad_to_multiple(ir_e, self.small_win)

        _, E, H, W = rgb_e.shape

        # 小窗口 query，大窗口 key/value
        rgb_small, gh, gw = self._window_partition(rgb_e, self.small_win)
        ir_small, _, _ = self._window_partition(ir_e, self.small_win)

        rgb_large = self._extract_large_windows(rgb_e, self.small_win, self.large_win)
        ir_large = self._extract_large_windows(ir_e, self.small_win, self.large_win)

        # LayerNorm 稳定 token 表达
        rgb_small = self.rgb_norm(rgb_small)
        ir_small = self.ir_norm(ir_small)
        rgb_large = self.rgb_norm(rgb_large)
        ir_large = self.ir_norm(ir_large)

        # 双向 asymmetric correlation
        rgb_delta_win, rho_r_win, attn_r = self.rgb_from_ir(rgb_small, ir_large)
        ir_delta_win, rho_i_win, attn_i = self.ir_from_rgb(ir_small, rgb_large)

        # attention 残差还原到二维
        rgb_delta_e = self._window_reverse(rgb_delta_win, gh, gw, self.small_win)
        ir_delta_e = self._window_reverse(ir_delta_win, gh, gw, self.small_win)

        rho_r = self._window_reverse(rho_r_win, gh, gw, self.small_win)
        rho_i = self._window_reverse(rho_i_win, gh, gw, self.small_win)

        # 映射回原通道
        rgb_delta = self.rgb_delta_out(rgb_delta_e)
        ir_delta = self.ir_delta_out(ir_delta_e)

        # 用冲突图和 attention 置信度控制跨模态残差注入
        rgb_star = rgb_pad + self.gamma_r * U * rho_r * rgb_delta
        ir_star = ir_pad + self.gamma_i * U * rho_i * ir_delta

        # -------------------------------------------------
        # match quality:
        # RGB 从 IR 大窗口取信息时，实际匹配到的 IR 区域质量
        # IR 从 RGB 大窗口取信息时，实际匹配到的 RGB 区域质量
        # -------------------------------------------------
        Qi_large = self._extract_large_windows(Qi, self.small_win, self.large_win)  # [B,L,Nl,1]
        Qr_large = self._extract_large_windows(Qr, self.small_win, self.large_win)

        Qi_large = Qi_large.squeeze(-1)  # [B,L,Nl]
        Qr_large = Qr_large.squeeze(-1)

        Qi_match_win = (attn_r * Qi_large.unsqueeze(2)).sum(dim=-1, keepdim=True)  # [B,L,Ns,1]
        Qr_match_win = (attn_i * Qr_large.unsqueeze(2)).sum(dim=-1, keepdim=True)

        Qi_match = self._window_reverse(Qi_match_win, gh, gw, self.small_win)
        Qr_match = self._window_reverse(Qr_match_win, gh, gw, self.small_win)

        # -------------------------------------------------
        # analytical reliability
        # 低冲突：看本模态质量
        # 高冲突：看跨模态匹配置信度 * 匹配区域质量
        # -------------------------------------------------
        Rr = (1.0 - U) * Qr + U * rho_r * Qi_match
        Ri = (1.0 - U) * Qi + U * rho_i * Qr_match

        Rr = Rr.clamp(min=self.eps)
        Ri = Ri.clamp(min=self.eps)

        wr_analytical = Rr / (Rr + Ri + self.eps)
        wi_analytical = 1.0 - wr_analytical

        # 训练初期从 0.5/0.5 平滑过渡到解析式权重
        alpha = torch.sigmoid(self.weight_alpha_raw)
        wr = 0.5 + alpha * (wr_analytical - 0.5)
        wi = 1.0 - wr

        fused = wr * rgb_star + wi * ir_star

        # 轻量输出整理，beta 初始为 0
        fused = fused + self.beta * self.out_refine(fused)

        fused = self._remove_pad(fused, pad_hw)
        return fused

class RawCueMap(nn.Module):
    """
    从原始 6 通道输入中提取原图 cue。

    输入:
        x: [B, 6, H, W]
           x[:, 0:3] 是 RGB
           x[:, 3:6] 是 IR

    输出:
        cue: [B, 2, H, W]
             cue[:, 0:1] = RGB brightness / gray-Y
             cue[:, 1:2] = IR intensity
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert x.dim() == 4 and x.shape[1] >= 6, f"RawCueMap expects [B,6,H,W], got {x.shape}"

        rgb = x[:, 0:3]
        ir = x[:, 3:6]

        # RGB 灰度亮度，假设输入已经归一化到 0~1
        y = (
            0.299 * rgb[:, 0:1]
            + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3]
        )

        # IR intensity，如果 IR 是三通道复制图，取均值即可
        t = ir.mean(dim=1, keepdim=True)

        # 稳妥一点，限制到 0~1
        y = y.clamp(0.0, 1.0)
        t = t.clamp(0.0, 1.0)

        cue = torch.cat([y, t], dim=1)
        return cue


class RawCueAddC(nn.Module):
    """
    RawCue-ADD-C:
    使用原图 brightness / intensity 的局部均值增强 ADD。

    输入:
        x = [rgb_feat, ir_feat, raw_cue]

        rgb_feat: [B, C, H, W]
        ir_feat : [B, C, H, W]
        raw_cue : [B, 2, H0, W0]
                  cue[:,0:1] = RGB brightness
                  cue[:,1:2] = IR intensity

    输出:
        out: [B, C, H, W]

    公式:
        Fadd = Fr + Fi
        Qr = local_mean(resize(Y))
        Qi = local_mean(resize(T))
        Fout = Fadd + gamma * (Qr * Fr + Qi * Fi)
    """
    def __init__(self, c, small_win=4, gamma_init=0.05, eps=1e-6):
        super().__init__()
        self.c = c
        self.small_win = small_win
        self.eps = eps

        # gamma 初始很小，保证开局接近 ADD
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # 可选的轻量输出整理，初始为 0，不破坏 ADD
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def _local_mean(self, x, k):
        """
        x: [B,1,H,W]
        使用奇数核做局部均值，避免 even kernel 导致尺寸偏移。
        small_win=4 时实际用 5x5 平滑。
        """
        kk = k if k % 2 == 1 else k + 1
        return F.avg_pool2d(x, kernel_size=kk, stride=1, padding=kk // 2)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            f"RawCueAddC expects [rgb_feat, ir_feat, raw_cue], got {type(x)} with len={len(x) if isinstance(x, (list, tuple)) else 'NA'}"

        rgb, ir, cue = x[0], x[1], x[2]

        assert rgb.shape == ir.shape, f"rgb and ir feature shape mismatch: {rgb.shape} vs {ir.shape}"
        assert cue.dim() == 4 and cue.shape[1] == 2, f"cue should be [B,2,H,W], got {cue.shape}"

        B, C, H, W = rgb.shape
        assert C == self.c, f"RawCueAddC channel mismatch: module c={self.c}, input C={C}"

        # resize raw cue 到当前特征尺度
        cue = F.interpolate(cue, size=(H, W), mode="bilinear", align_corners=False)

        y = cue[:, 0:1]  # RGB brightness
        t = cue[:, 1:2]  # IR intensity

        # C 版本：只做局部均值，不做阈值函数
        Qr = self._local_mean(y, self.small_win).clamp(0.0, 1.0)
        Qi = self._local_mean(t, self.small_win).clamp(0.0, 1.0)


        # ADD 主路径
        fadd = rgb + ir

        # raw cue 残差增强
        cue_res = Qr * rgb + Qi * ir
        out = fadd + self.gamma * cue_res

        #新残差
        # comp_res = Qr * (1.0 - Qi) * rgb + Qi * (1.0 - Qr) * ir
        # out = fadd + self.gamma * comp_res

        # 初始 beta=0，所以刚开始不影响
        # out = out + self.beta * self.refine(out)
        out = fadd + self.beta * self.refine(fadd)
        return out

class RawCueAddD(nn.Module):
    """
    RawCue-ADD-D:
    在 ADD 主路径上加入原图 brightness / intensity 的局部质量增强。

    相比 RawCueAddC：
    1. Qr 不再等于局部平均亮度，而是亮度区间质量；
    2. Qi 不再等于局部平均热响应，而是热响应 + 热显著性；
    3. 主路径仍然是 ADD，不直接压制任何模态。

    输入:
        x = [rgb_feat, ir_feat, raw_cue]

        rgb_feat: [B, C, H, W]
        ir_feat : [B, C, H, W]
        raw_cue : [B, 2, H0, W0]
                  raw_cue[:, 0:1] = RGB brightness
                  raw_cue[:, 1:2] = IR intensity

    输出:
        out: [B, C, H, W]

    args: [c, small_win, large_win, gamma_init, low_thr, high_thr, smooth_s, thermal_alpha]
    small_win = 4，实际局部平均核为 5x5
    large_win = 8，实际局部上下文核为 9x9
    0.20 和 0.78 怎么来:亮度基本按 0–255 统计:
    0–50：低光 / 极暗
    50–200：有效可见光范围
    200–255：可能过亮 / 反光 / 过曝
    换算到0-1 50 / 255 ≈ 0.196 ；200 / 255 ≈ 0.784
    s 是 soft threshold 的平滑宽度。如果 s 很小，比如 0.02：亮度刚过阈值，Qr 会突然变化，接近硬阈值。过大则阈值边界很模糊，亮度质量区分不明显。
    0.06 在 0–1 亮度范围里大约对应 15 个灰度级的平滑宽度，既能保证一定的平滑过渡，又不会让亮度质量区分过于模糊。
    0.6：保留 IR 本身的热响应优势，0.4：引入局部突出性，减少热背景误增强
    """

    def __init__(
        self,
        c,
        small_win=4,
        large_win=8,
        gamma_init=0.05,
        low_thr=0.20,
        high_thr=0.78,
        smooth_s=0.06,
        thermal_alpha=0.6,
        eps=1e-6,
    ):
        super().__init__()
        self.c = c
        self.small_win = small_win
        self.large_win = large_win
        self.eps = eps

        self.low_thr = low_thr
        self.high_thr = high_thr
        self.smooth_s = smooth_s
        self.thermal_alpha = thermal_alpha

        # gamma 初始较小，保证开局接近 ADD
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        # 可选 refine，初始关闭
        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def _norm01(self, x):
        """
        对每张图单独做 0-1 归一化。
        x: [B, 1, H, W]
        """
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

    def _local_mean(self, x, k):
        """
        局部滑动平均。
        注意：如果 k 是偶数，会自动变成 k+1，保证输出尺寸不变。
        small_win=4 时，实际是 5x5 平均。
        """
        kk = k if k % 2 == 1 else k + 1
        return F.avg_pool2d(x, kernel_size=kk, stride=1, padding=kk // 2)

    def _brightness_quality(self, y):
        """
        RGB 亮度质量图。

        不是越亮越好，而是亮度处于 low_thr 和 high_thr 之间较可靠。
        """
        y_local = self._local_mean(y, self.small_win)

        q_dark = torch.sigmoid((y_local - self.low_thr) / self.smooth_s)
        q_over = torch.sigmoid((self.high_thr - y_local) / self.smooth_s)

        q = q_dark * q_over
        return q.clamp(0.0, 1.0)

    def _thermal_quality(self, t):
        """
        IR 热质量图。

        Qi = alpha * T_local + (1 - alpha) * T_saliency

        T_local:
            局部热响应强度。

        T_saliency:
            small window 和 large window 的差异，表示相对周围是否突出。
        """
        t_local = self._local_mean(t, self.small_win)
        t_context = self._local_mean(t, self.large_win)

        # 这里必须用 self._norm01，而不是 norm01
        t_local = self._norm01(t_local)
        t_context = self._norm01(t_context)

        t_saliency = torch.abs(t_local - t_context)
        t_saliency = self._norm01(t_saliency)

        alpha = self.thermal_alpha
        q = alpha * t_local + (1.0 - alpha) * t_saliency
        return q.clamp(0.0, 1.0)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            f"RawCueAddD expects [rgb_feat, ir_feat, raw_cue], got {type(x)}"

        rgb, ir, cue = x[0], x[1], x[2]

        assert rgb.shape == ir.shape, \
            f"rgb and ir feature shape mismatch: {rgb.shape} vs {ir.shape}"

        assert cue.dim() == 4 and cue.shape[1] == 2, \
            f"cue should be [B,2,H,W], got {cue.shape}"

        B, C, H, W = rgb.shape
        assert C == self.c, f"RawCueAddD channel mismatch: module c={self.c}, input C={C}"

        # resize raw cue 到当前特征尺度
        cue = F.interpolate(cue, size=(H, W), mode="bilinear", align_corners=False)

        y = cue[:, 0:1].clamp(0.0, 1.0)  # RGB brightness
        t = cue[:, 1:2].clamp(0.0, 1.0)  # IR intensity

        # D 版本的 Qr / Qi
        Qr = self._brightness_quality(y)
        Qi = self._thermal_quality(t)

        # ADD 主路径
        fadd = rgb + ir

        # raw cue 残差增强
        cue_res = Qr * rgb + Qi * ir

        out = fadd + self.gamma * cue_res

        # refine 初始关闭，beta=0
        out = out + self.beta * self.refine(out)

        return out

class _TSCIv2LocalAttention(nn.Module):
    """
    输入:
        src_small: [B, L, Ns, C]
        ref_large: [B, L, Nl, C]

    输出:
        delta: [B, L, Ns, C]
        其中 delta = matched_ref - src_small
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, src_small, ref_large):
        B, L, Ns, C = src_small.shape
        Nl = ref_large.shape[2]

        q = self.q_proj(src_small).view(B, L, Ns, self.num_heads, self.head_dim)
        k = self.k_proj(ref_large).view(B, L, Nl, self.num_heads, self.head_dim)
        v = self.v_proj(ref_large).view(B, L, Nl, self.num_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)  # [B,L,H,Ns,D]
        k = k.permute(0, 1, 3, 2, 4)  # [B,L,H,Nl,D]
        v = v.permute(0, 1, 3, 2, 4)  # [B,L,H,Nl,D]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        matched = attn @ v
        matched = matched.permute(0, 1, 3, 2, 4).contiguous().view(B, L, Ns, C)

        delta = matched - src_small
        delta = self.out_proj(delta)
        return delta


class TSCIv2RawCueFusion(nn.Module):
    """
    TSCI-v2 + RawCueAddC final fusion.
    [占位c,embed_dim,small_win,large_win,num_heads,topk_ratio,score_lambda,score_mode,r_fuse,eta_init,raw_gamma_init]
    输入:
        x = [rgb_feat, ir_feat, raw_cue]

        rgb_feat: [B, C, H, W]
        ir_feat : [B, C, H, W]
        raw_cue : [B, 2, H0, W0]
                  raw_cue[:, 0:1] = RGB brightness
                  raw_cue[:, 1:2] = IR intensity

    输出:
        fused feature: [B, C, H, W]

    窗口级评分组成:
        1) 特征冲突:
           C_i = 0.5 * Norm(D_i) + 0.5 * Norm(I_i)

        2) 局部质量:
           Q_i = 0.5 * Norm(local_std) + 0.5 * Norm(local_grad)

        3) 互补需求:
           N_i = 0.5 * ((1 - Qr_i) * Qt_i + (1 - Qt_i) * Qr_i)

        4) 可选可靠性门控:
           - simple:
               P_i = score_lambda * C_i + (1 - score_lambda) * N_i
           - with_r:
               R_i = max(Qr_i, Qt_i) or avg(Qr_i, Qt_i)
               P_i = R_i * (score_lambda * C_i + (1 - score_lambda) * N_i)
           - with_r_consistency:
               R_i = base_R_i * (1 - In_i)
               P_i = R_i * (score_lambda * C_i + (1 - score_lambda) * N_i)

    说明:
        - score_mode 控制最终窗口分数的计算方式
        - r_fuse 控制 R_i 的基础聚合方式（max / avg）
        - 推荐先用 simple 做基线，再试 with_r
    """
    def __init__(
        self,
        c,
        embed_dim=64,
        small_win=4,
        large_win=6,
        num_heads=4,
        topk_ratio=0.25,
        score_lambda=0.5,
        score_mode="with_r",          # "simple" | "with_r" | "with_r_consistency" | "with_r_dark_ir"
        r_fuse="max",                 # "max" | "avg"
        eta_init=0.03,
        raw_gamma_init=0.05,
        gate_floor=0.5,
        eps=1e-6,
    ):
        super().__init__()
        assert large_win >= small_win, "large_win should be >= small_win"
        assert score_mode in ["simple", "with_r", "with_r_consistency","with_r_dark_ir"]
        assert r_fuse in ["max", "avg"]

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.topk_ratio = topk_ratio
        self.score_lambda = score_lambda
        self.score_mode = score_mode
        self.r_fuse = r_fuse
        self.gate_floor = gate_floor
        self.eps = eps

        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        self.rgb_from_ir = _TSCIv2LocalAttention(embed_dim, num_heads)
        self.ir_from_rgb = _TSCIv2LocalAttention(embed_dim, num_heads)

        self.rgb_delta_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_delta_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.eta_r = nn.Parameter(torch.tensor(float(eta_init)))
        self.eta_i = nn.Parameter(torch.tensor(float(eta_init)))
        self.raw_gamma = nn.Parameter(torch.tensor(float(raw_gamma_init)))

        self.refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.beta = nn.Parameter(torch.zeros(1))
        self.enable_debug = False
        self.debug_info = {}
        self.dark_thr = 0.15
        self.dark_s = 0.04

    # -------------------------
    # helper
    # -------------------------
    def _norm01_window(self, x):
        """
        x: [B, L]
        对每张图内部的所有窗口做 0-1 归一化，用于 top-k 排序。
        """
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
        return (x - x_min) / (x_max - x_min + self.eps)

    def _pad_to_multiple(self, x, multiple):
        B, C, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (H, W)

    @staticmethod
    def _remove_pad(x, hw):
        H, W = hw
        return x[:, :, :H, :W]

    def _window_partition(self, x, win):
        B, C, H, W = x.shape
        assert H % win == 0 and W % win == 0, f"H,W must be divisible by win={win}, got {H},{W}"
        gh, gw = H // win, W // win

        x = x.view(B, C, gh, win, gw, win)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        x = x.view(B, gh * gw, win * win, C)
        return x, gh, gw

    def _window_reverse(self, x_win, gh, gw, win):
        B, L, N, C = x_win.shape
        x = x_win.view(B, gh, gw, win, win, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, gh * win, gw * win)
        return x

    def _window_scalar_to_map(self, s, gh, gw, win):
        """
        s: [B, L]，每个窗口一个标量
        返回: [B, 1, H, W]，把每个窗口分数铺回到该窗口覆盖区域
        """
        B, L = s.shape
        m = s.view(B, gh, gw, 1, 1, 1).expand(-1, -1, -1, win, win, 1)
        m = m.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, 1, gh * win, gw * win)
        return m
    
    def _extract_large_windows(self, x, small_win, large_win):
        B, C, H, W = x.shape

        total_pad = large_win - small_win
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        pad_top = total_pad // 2
        pad_bottom = total_pad - pad_top

        x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

        patches = F.unfold(
            x_pad,
            kernel_size=large_win,
            stride=small_win,
            padding=0,
        )  # [B, C*large*large, L]

        L = patches.shape[-1]
        patches = patches.transpose(1, 2).contiguous()
        patches = patches.view(B, L, C, large_win * large_win)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches

    def _grad_mag(self, x):
        dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])

        dx = F.pad(dx, (0, 1, 0, 0))
        dy = F.pad(dy, (0, 0, 0, 1))
        return dx + dy

    def _local_mean(self, x, k):
        kk = k if k % 2 == 1 else k + 1
        return F.avg_pool2d(x, kernel_size=kk, stride=1, padding=kk // 2)

    # -------------------------
    # score calculation
    # -------------------------
    def _feature_conflict_score(self, rgb, ir):
        """
        rgb, ir: [B,C,H,W]
        return:
            C_score: [B,L]
            Dn     : [B,L]
            In     : [B,L]
            gh, gw
        """
        rgb_win, gh, gw = self._window_partition(rgb, self.small_win)
        ir_win, _, _ = self._window_partition(ir, self.small_win)

        # D_i = mean(abs(Fr - Ft))
        D = torch.abs(rgb_win - ir_win).mean(dim=(2, 3))  # [B,L]

        # I_i = 1 - cosine(vec(Fr), vec(Ft))
        rgb_vec = rgb_win.flatten(2)
        ir_vec = ir_win.flatten(2)
        S = F.cosine_similarity(rgb_vec, ir_vec, dim=-1, eps=self.eps)
        I = 1.0 - S

        Dn = self._norm01_window(D)
        In = self._norm01_window(I)

        # C_score = 0.5 * Dn + 0.5 * In
        C_score =  Dn + In
        return C_score, Dn, In, gh, gw

    def _local_quality_score(self, cue_channel):
        """
        cue_channel: [B,1,H,W]
        return Q_i: [B,L]

        Q_i = 0.5 * Norm(local_std) + 0.5 * Norm(local_grad)
        """
        win, _, _ = self._window_partition(cue_channel, self.small_win)
        win = win.squeeze(-1)  # [B,L,N]

        local_std = win.std(dim=-1, unbiased=False)  # [B,L]

        grad_map = self._grad_mag(cue_channel)
        grad_win, _, _ = self._window_partition(grad_map, self.small_win)
        local_grad = grad_win.mean(dim=(2, 3))  # [B,L]

        std_n = self._norm01_window(local_std)
        grad_n = self._norm01_window(local_grad)

        # Q = 0.5 * std_n + 0.5 * grad_n
        Q =std_n + grad_n
        return Q.clamp(0.0, 1.0)

    def _feature_energy_score(self, feat):
        """
        feat: [B,C,H,W]
        return: [B,L]
        用特征能量作为窗口目标响应代理
        """
        win, _, _ = self._window_partition(feat, self.small_win)
        e = win.abs().mean(dim=(2, 3))  # [B,L]
        return self._norm01_window(e).clamp(0.0, 1.0)

    def _reliability_score(self, Qr, Qt, In=None):
        """
        共享基础量版 R_i
        - with_r:
            R_i = max(Qr, Qt) 或 avg(Qr, Qt)
        - with_r_consistency:
            R_i = base_R * (1 - In)
        """
        if self.r_fuse == "max":
            base_R = torch.maximum(Qr, Qt)
        else:
            base_R = 0.5 * (Qr + Qt)

        if self.score_mode == "with_r_consistency":
            assert In is not None, "with_r_consistency requires In"
            R = base_R * (1.0 - In)
        else:
            R = base_R

        return R.clamp(0.0, 1.0)

    def _compose_priority_score(self, C_score, N_score, Qr=None, Qt=None, In=None):
        """
        统一窗口优先级分数
        """
        lam = self.score_lambda
        base_score = lam * C_score + (1.0 - lam) * N_score

        if self.score_mode == "simple":
            P = base_score
        elif self.score_mode in ["with_r", "with_r_consistency","with_r_dark_ir"]:
            R_score = self._reliability_score(Qr, Qt, In=In)
            P = R_score * base_score
        else:
            raise ValueError(f"Unknown score_mode: {self.score_mode}")

        return P.clamp(0.0, 1.0)
    def _make_topk_mask(self, P):
        """
        P: [B, L]
        return:
            mask: [B, L, 1, 1]
        """
        B, L = P.shape

        if self.topk_ratio <= 0:
            return torch.zeros(B, L, 1, 1, device=P.device, dtype=P.dtype)

        k = max(1, int(math.ceil(L * self.topk_ratio)))
        idx = torch.topk(P, k=k, dim=1).indices

        mask = torch.zeros_like(P)
        mask.scatter_(1, idx, 1.0)
        return mask.unsqueeze(-1).unsqueeze(-1)

    # -------------------------
    # forward
    # -------------------------
    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            f"TSCIv2RawCueFusion expects [rgb_feat, ir_feat, raw_cue], got {type(x)}"

        rgb, ir, cue = x[0], x[1], x[2]

        assert rgb.shape == ir.shape, f"rgb and ir feature mismatch: {rgb.shape} vs {ir.shape}"
        assert cue.dim() == 4 and cue.shape[1] == 2, f"cue should be [B,2,H,W], got {cue.shape}"

        B, C, H0, W0 = rgb.shape
        assert C == self.c, f"TSCIv2RawCueFusion channel mismatch: module c={self.c}, input C={C}"

        rgb_pad, pad_hw = self._pad_to_multiple(rgb, self.small_win)
        ir_pad, _ = self._pad_to_multiple(ir, self.small_win)

        _, _, H, W = rgb_pad.shape

        # raw cue resize 到当前特征尺度
        cue = F.interpolate(cue, size=(H, W), mode="bilinear", align_corners=False)
        y = cue[:, 0:1].clamp(0.0, 1.0)
        t = cue[:, 1:2].clamp(0.0, 1.0)

        # 1) feature conflict C_i
        C_score, Dn, In, gh, gw = self._feature_conflict_score(rgb_pad, ir_pad)

        # 2) local quality Qr_i, Qt_i
        Qr = self._local_quality_score(y)
        Qt = self._local_quality_score(t)

        # 3) complementary need N_i
        N_r_from_t = (1.0 - Qr) * Qt
        N_t_from_r = (1.0 - Qt) * Qr
        N_score = 0.5 * (N_r_from_t + N_t_from_r)

        # 4) final interaction priority P_i
        P_normal = self._compose_priority_score(
            C_score=C_score,
            N_score=N_score,
            Qr=Qr,
            Qt=Qt,
            In=In,
        )
        if self.score_mode == "with_r_dark_ir":
            # 1) 整图低光程度，越暗越接近 1
            y_mean = y.mean(dim=(2, 3))  # [B,1]
            dark = torch.sigmoid((self.dark_thr - y_mean) / self.dark_s)  # [B,1]

            # 2) 极暗时用 IR feature energy 选窗
            P_dark = self._feature_energy_score(ir_pad)  # [B,L]

            # 3) 混合：正常光照用原公式，极暗用 IR 主导
            # P = (1.0 - dark) * P_normal + dark * P_dark
            P = P_normal + 0.5 * dark * (P_dark - P_normal) #soft 版本
        else:
            P = P_normal

        # 5) top-k mask
        topk_mask = self._make_topk_mask(P)

        # direction gates: 保留方向信息，但不让 gate 过度压残差
        gate_r = self.gate_floor + (1.0 - self.gate_floor) * N_r_from_t
        gate_i = self.gate_floor + (1.0 - self.gate_floor) * N_t_from_r

        gate_r = gate_r.unsqueeze(-1).unsqueeze(-1)
        gate_i = gate_i.unsqueeze(-1).unsqueeze(-1)

        # 6) asymmetric window attention residual
        rgb_e = self.rgb_in(rgb_pad)
        ir_e = self.ir_in(ir_pad)

        rgb_small, gh, gw = self._window_partition(rgb_e, self.small_win)
        ir_small, _, _ = self._window_partition(ir_e, self.small_win)

        rgb_large = self._extract_large_windows(rgb_e, self.small_win, self.large_win)
        ir_large = self._extract_large_windows(ir_e, self.small_win, self.large_win)

        rgb_small = self.rgb_norm(rgb_small)
        ir_small = self.ir_norm(ir_small)
        rgb_large = self.rgb_norm(rgb_large)
        ir_large = self.ir_norm(ir_large)

        rgb_delta = self.rgb_from_ir(rgb_small, ir_large)   # [B,L,Ns,E]
        ir_delta = self.ir_from_rgb(ir_small, rgb_large)    # [B,L,Ns,E]

        rgb_delta = topk_mask * gate_r * rgb_delta
        ir_delta = topk_mask * gate_i * ir_delta

        rgb_delta = self._window_reverse(rgb_delta, gh, gw, self.small_win)
        ir_delta = self._window_reverse(ir_delta, gh, gw, self.small_win)

        # rgb_delta = self._remove_pad(rgb_delta, pad_hw)
        # ir_delta = self._remove_pad(ir_delta, pad_hw)

        rgb_delta = self.rgb_delta_out(rgb_delta)
        ir_delta = self.ir_delta_out(ir_delta)

        rgb_enh = rgb_pad + self.eta_r * rgb_delta
        ir_enh = ir_pad + self.eta_i * ir_delta

        # 7) RawCueAddC final fusion
        Qr_map = self._local_mean(y, self.small_win).clamp(0.0, 1.0)
        Qt_map = self._local_mean(t, self.small_win).clamp(0.0, 1.0)

        fadd = rgb_enh + ir_enh
        fcue = Qr_map * rgb_enh + Qt_map * ir_enh

        out = fadd + self.raw_gamma * fcue

        # refine: 和 RawCueAddC 保持一致
        out = out + self.beta * self.refine(out)

        out = self._remove_pad(out, pad_hw)
        if getattr(self, "enable_debug", False):
            rgb_energy = rgb_pad.abs().mean(dim=1, keepdim=True)
            ir_energy = ir_pad.abs().mean(dim=1, keepdim=True)

            C_map = self._window_scalar_to_map(C_score, gh, gw, self.small_win)
            N_map = self._window_scalar_to_map(N_score, gh, gw, self.small_win)
            P_map = self._window_scalar_to_map(P, gh, gw, self.small_win)

            topk_scalar = topk_mask.squeeze(-1).squeeze(-1)   # [B, L]
            topk_map = self._window_scalar_to_map(topk_scalar, gh, gw, self.small_win)

            rgb_residual = (self.eta_r * rgb_delta).abs().mean(dim=1, keepdim=True)
            ir_residual = (self.eta_i * ir_delta).abs().mean(dim=1, keepdim=True)

            fused_energy = out.abs().mean(dim=1, keepdim=True)

            if self.score_mode == "with_r_dark_ir":
                self.debug_info["P_normal_map"] = self._remove_pad(
                    self._window_scalar_to_map(P_normal, gh, gw, self.small_win), pad_hw
                ).detach().cpu()

                self.debug_info["P_dark_map"] = self._remove_pad(
                    self._window_scalar_to_map(P_dark, gh, gw, self.small_win), pad_hw
                ).detach().cpu()

                self.debug_info["dark_value"] = dark.detach().cpu()

            self.debug_info = {
                "rgb_energy": self._remove_pad(rgb_energy, pad_hw).detach().cpu(),
                "ir_energy": self._remove_pad(ir_energy, pad_hw).detach().cpu(),
                "C_map": self._remove_pad(C_map, pad_hw).detach().cpu(),
                "N_map": self._remove_pad(N_map, pad_hw).detach().cpu(),
                "P_map": self._remove_pad(P_map, pad_hw).detach().cpu(),
                "topk_map": self._remove_pad(topk_map, pad_hw).detach().cpu(),
                "Qr_map": self._remove_pad(Qr_map, pad_hw).detach().cpu(),
                "Qt_map": self._remove_pad(Qt_map, pad_hw).detach().cpu(),
                "rgb_residual": self._remove_pad(rgb_residual, pad_hw).detach().cpu(),
                "ir_residual": self._remove_pad(ir_residual, pad_hw).detach().cpu(),
                "fused_energy": self._remove_pad(fused_energy, pad_hw).detach().cpu(),
            }
        return out

# ============================================================
# TSCI-v3: Top-k 前置窗口交互 + 证据注入式 SS2D 上下文传播
# ============================================================

class _TSCIv3MatchedWindowAttention(nn.Module):
    """
    Local asymmetric cross-modal attention for selected windows.

    输入:
        src_small: [B, K, Ns, C]
        ref_large: [B, K, Nl, C]

    输出:
        matched: [B, K, Ns, C]

    注意:
        这里返回的是 matched cross evidence，不再做 matched - src_small。
        因此它表示“另一模态补充过来的匹配特征”，不是 residual delta。
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, src_small, ref_large):
        B, K, Ns, C = src_small.shape
        Nl = ref_large.shape[2]

        q = self.q_proj(src_small).view(B, K, Ns, self.num_heads, self.head_dim)
        k = self.k_proj(ref_large).view(B, K, Nl, self.num_heads, self.head_dim)
        v = self.v_proj(ref_large).view(B, K, Nl, self.num_heads, self.head_dim)

        q = q.permute(0, 1, 3, 2, 4)  # [B,K,H,Ns,D]
        k = k.permute(0, 1, 3, 2, 4)  # [B,K,H,Nl,D]
        v = v.permute(0, 1, 3, 2, 4)  # [B,K,H,Nl,D]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        matched = attn @ v
        matched = matched.permute(0, 1, 3, 2, 4).contiguous().view(B, K, Ns, C)
        return self.out_proj(matched)


class TSCISS2DContext(nn.Module):
    """
    Evidence-injected SS2D context propagation for TSCI.

    输入:
        rgb_loc  : [B,C,H,W]  局部交互后 RGB 特征
        ir_loc   : [B,C,H,W]  局部交互后 IR 特征
        rgb_cross: [B,C,H,W]  IR -> RGB 的 Top-k 跨模态补充证据
        ir_cross : [B,C,H,W]  RGB -> IR 的 Top-k 跨模态补充证据

    输出:
        rgb_out, ir_out, ctx

    设计逻辑:
        1) 原特征提供稠密语义载体；
        2) cross evidence 提供 Top-k 局部交互证据；
        3) SS2D 四方向 Mamba 传播上下文；
        4) 将上下文分别回写到 RGB / IR 双流。
    """
    def __init__(
        self,
        c,
        d_state=16,
        d_conv=4,
        expand=2,
        gamma_init=0.10,
    ):
        super().__init__()
        from mamba_ssm import Mamba

        self.c = c

        def proj_block():
            return nn.Sequential(
                nn.Conv2d(c, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(inplace=True),
            )

        # 稠密语义载体
        self.rgb_proj = proj_block()
        self.ir_proj = proj_block()

        # Top-k cross evidence 注入项
        self.rgb_cross_proj = proj_block()
        self.ir_cross_proj = proj_block()
        self.lambda_rgb_cross = nn.Parameter(torch.tensor(1.0))
        self.lambda_ir_cross = nn.Parameter(torch.tensor(1.0))

        # SS2D 输入投影，分成 content 和 gate
        self.in_proj = nn.Conv2d(c, 2 * c, 1, bias=True)
        self.dwconv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        # 四方向 Mamba。这里直接使用官方 Mamba block 做方向扫描，
        # 避免手写 selective_scan 的低层参数细节，同时保留 SS2D 的四方向传播思想。
        self.m_lr = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_rl = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_tb = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        self.m_bt = Mamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm = nn.LayerNorm(c)

        self.out_proj = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
        )

        self.to_rgb = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.to_ir = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ctx_gate = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1, bias=True),
            nn.Sigmoid()
        )

        self.gamma_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_ir = nn.Parameter(torch.tensor(float(gamma_init)))

    @staticmethod
    def _map_to_seq(x):
        # [B,C,H,W] -> [B,H*W,C]
        return x.flatten(2).transpose(1, 2).contiguous()

    @staticmethod
    def _seq_to_map(seq, h, w):
        # [B,H*W,C] -> [B,C,H,W]
        b, _, c = seq.shape
        return seq.transpose(1, 2).contiguous().view(b, c, h, w)

    def _scan_lr(self, x):
        b, c, h, w = x.shape
        seq = self._map_to_seq(x)
        seq = self.m_lr(self.norm(seq))
        return self._seq_to_map(seq, h, w)

    def _scan_rl(self, x):
        b, c, h, w = x.shape
        xf = x.flip(-1).contiguous()
        seq = self._map_to_seq(xf)
        seq = self.m_rl(self.norm(seq))
        y = self._seq_to_map(seq, h, w)
        return y.flip(-1).contiguous()

    def _scan_tb(self, x):
        # transpose 后按行扫描，相当于原图从上到下的列方向扫描
        b, c, h, w = x.shape
        xt = x.transpose(2, 3).contiguous()  # [B,C,W,H]
        seq = self._map_to_seq(xt)
        seq = self.m_tb(self.norm(seq))
        y = self._seq_to_map(seq, w, h)
        return y.transpose(2, 3).contiguous()

    def _scan_bt(self, x):
        # 先上下翻转，再做 top-bottom 扫描，最后还原
        b, c, h, w = x.shape
        xf = x.flip(-2).contiguous()
        xt = xf.transpose(2, 3).contiguous()  # [B,C,W,H]
        seq = self._map_to_seq(xt)
        seq = self.m_bt(self.norm(seq))
        y = self._seq_to_map(seq, w, h).transpose(2, 3).contiguous()
        return y.flip(-2).contiguous()
        

    def forward(self, rgb_loc, ir_loc, rgb_cross, ir_cross):
        # -------------------------------------------------
        # CPU fallback:
        # YOLO 在 model 初始化 / info 阶段会用 CPU dummy input
        # 但 mamba_ssm 的 Mamba 依赖 CUDA kernel，CPU 上会报错。
        # 这里直接跳过 SS2D，只保证输出形状正确。
        # 真正训练/推理时输入在 CUDA 上，会正常进入 Mamba。
        # -------------------------------------------------
        if not rgb_loc.is_cuda:
            ctx = torch.zeros_like(rgb_loc)
            return rgb_loc, ir_loc, ctx
        # 证据注入式稠密上下文输入
        h = (
            self.rgb_proj(rgb_loc)
            + self.ir_proj(ir_loc)
            + self.lambda_rgb_cross * self.rgb_cross_proj(rgb_cross)
            + self.lambda_ir_cross * self.ir_cross_proj(ir_cross)
        )

        x, z = self.in_proj(h).chunk(2, dim=1)
        x = self.dwconv(x)

        y = (
            self._scan_lr(x)
            + self._scan_rl(x)
            + self._scan_tb(x)
            + self._scan_bt(x)
        ) * 0.25

        # gate 控制上下文输出
        ctx = self.out_proj(y * F.silu(z))
        # v1 
        # rgb_out = rgb_loc + self.gamma_rgb * self.to_rgb(ctx)
        # ir_out = ir_loc + self.gamma_ir * self.to_ir(ctx)

        # v2 只有 cross evidence 强的位置，才允许 SS2D 强回写；
        e = torch.cat([
            rgb_cross.abs().mean(1, keepdim=True),
            ir_cross.abs().mean(1, keepdim=True),
        ], dim=1)
        g = self.ctx_gate(e)
        rgb_out = rgb_loc + self.gamma_rgb * g * self.to_rgb(ctx)
        ir_out  = ir_loc  + self.gamma_ir  * g * self.to_ir(ctx)

        return rgb_out, ir_out, ctx


class TSCIv3RawCueSS2DFusion(TSCIv2RawCueFusion):
    """
    TSCI-v3 + RawCue + Top-k 前置窗口交互 + SS2D Mamba 上下文传播。

    YAML args 推荐:
        [c, embed_dim, small_win, large_win, num_heads, topk_ratio,
         score_lambda, score_mode, r_fuse, eta_init, raw_gamma_init,
         gate_floor, eps, d_state, d_conv, expand, ctx_gamma_init]

    与 TSCIv2 的关键区别:
        1) Top-k 前置：只对 Top-k 窗口计算局部跨模态 attention；
        2) attention 输出 matched cross evidence，不再 matched - source；
        3) 先用 cross evidence 局部增强双流；
        4) 再用证据注入式 SS2D 对增强后的双流做上下文传播；
        5) 最后仍使用 RawCueAddC 风格融合。
    """
    def __init__(
        self,
        c,
        embed_dim=64,
        small_win=4,
        large_win=6,
        num_heads=4,
        topk_ratio=0.25,
        score_lambda=0.5,
        score_mode="with_r",
        r_fuse="max",
        eta_init=0.03,
        raw_gamma_init=0.05,
        gate_floor=0.5,
        eps=1e-6,
        d_state=16,
        d_conv=4,
        expand=2,
        # ctx_gamma_init=0.10, # v1 版本，SS2D 回写强度较大
        ctx_gamma_init=0.02, # v2 版本，现在可视化已经说明 上下文传播太强。我们要让模型自己学会什么时候开传播，而不是开局就让 SS2D 改写特征。
    ):
        super().__init__(
            c=c,
            embed_dim=embed_dim,
            small_win=small_win,
            large_win=large_win,
            num_heads=num_heads,
            topk_ratio=topk_ratio,
            score_lambda=score_lambda,
            score_mode=score_mode,
            r_fuse=r_fuse,
            eta_init=eta_init,
            raw_gamma_init=raw_gamma_init,
            gate_floor=gate_floor,
            eps=eps,
        )

        # matched cross evidence attention：不再输出 delta
        self.rgb_from_ir = _TSCIv3MatchedWindowAttention(embed_dim, num_heads)
        self.ir_from_rgb = _TSCIv3MatchedWindowAttention(embed_dim, num_heads)

        # 名字沿用旧变量以减少其他代码改动，但语义变为 cross evidence projection
        self.rgb_delta_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_delta_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.ss2d_context = TSCISS2DContext(
            c=c,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            gamma_init=ctx_gamma_init,
        )

    def _get_topk_indices(self, P):
        """
        P: [B,L]
        return idx: [B,K]
        """
        B, L = P.shape
        if self.topk_ratio <= 0:
            return None
        k = max(1, int(math.ceil(L * self.topk_ratio)))
        k = min(k, L)
        return torch.topk(P, k=k, dim=1).indices

    @staticmethod
    def _gather_window_tokens(x, idx):
        """
        x  : [B,L,N,C]
        idx: [B,K]
        out: [B,K,N,C]
        """
        B, L, N, C = x.shape
        K = idx.shape[1]
        index = idx[:, :, None, None].expand(B, K, N, C)
        return x.gather(dim=1, index=index)

    @staticmethod
    def _scatter_window_tokens(src, idx, out_shape):
        """
        src: [B,K,N,C]
        idx: [B,K]
        out_shape: shape of full window tensor [B,L,N,C]
        """
        B, L, N, C = out_shape
        K = idx.shape[1]
        out = src.new_zeros(out_shape)
        index = idx[:, :, None, None].expand(B, K, N, C)
        out.scatter_(dim=1, index=index, src=src)
        return out

    @staticmethod
    def _gather_window_gate(g, idx):
        """
        g: [B,L,1,1]
        idx: [B,K]
        return: [B,K,1,1]
        """
        B, L, _, _ = g.shape
        K = idx.shape[1]
        index = idx[:, :, None, None].expand(B, K, 1, 1)
        return g.gather(dim=1, index=index)

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, \
            f"TSCIv3RawCueSS2DFusion expects [rgb_feat, ir_feat, raw_cue], got {type(x)}"

        rgb, ir, cue = x[0], x[1], x[2]
        assert rgb.shape == ir.shape, f"rgb and ir feature mismatch: {rgb.shape} vs {ir.shape}"
        assert cue.dim() == 4 and cue.shape[1] == 2, f"cue should be [B,2,H,W], got {cue.shape}"

        B, C, H0, W0 = rgb.shape
        assert C == self.c, f"TSCIv3RawCueSS2DFusion channel mismatch: module c={self.c}, input C={C}"

        rgb_pad, pad_hw = self._pad_to_multiple(rgb, self.small_win)
        ir_pad, _ = self._pad_to_multiple(ir, self.small_win)
        _, _, H, W = rgb_pad.shape

        # raw cue resize 到当前特征尺度
        cue = F.interpolate(cue, size=(H, W), mode="bilinear", align_corners=False)
        y = cue[:, 0:1].clamp(0.0, 1.0)
        t = cue[:, 1:2].clamp(0.0, 1.0)

        # 1) 窗口评分
        C_score, Dn, In, gh, gw = self._feature_conflict_score(rgb_pad, ir_pad)
        Qr = self._local_quality_score(y)
        Qt = self._local_quality_score(t)

        N_r_from_t = (1.0 - Qr) * Qt
        N_t_from_r = (1.0 - Qt) * Qr
        N_score = 0.5 * (N_r_from_t + N_t_from_r)

        P_normal = self._compose_priority_score(
            C_score=C_score,
            N_score=N_score,
            Qr=Qr,
            Qt=Qt,
            In=In,
        )

        if self.score_mode == "with_r_dark_ir":
            y_mean = y.mean(dim=(2, 3))
            dark = torch.sigmoid((self.dark_thr - y_mean) / self.dark_s)
            P_dark = self._feature_energy_score(ir_pad)
            P = P_normal + 0.5 * dark * (P_dark - P_normal)
        else:
            P = P_normal
            P_dark = None
            dark = None

        idx = self._get_topk_indices(P)

        # 方向门控
        gate_r = self.gate_floor + (1.0 - self.gate_floor) * N_r_from_t
        gate_i = self.gate_floor + (1.0 - self.gate_floor) * N_t_from_r
        gate_r = gate_r.unsqueeze(-1).unsqueeze(-1)  # [B,L,1,1]
        gate_i = gate_i.unsqueeze(-1).unsqueeze(-1)

        # 2) 投影到 attention embedding 空间
        rgb_e = self.rgb_in(rgb_pad)
        ir_e = self.ir_in(ir_pad)

        rgb_small, gh, gw = self._window_partition(rgb_e, self.small_win)
        ir_small, _, _ = self._window_partition(ir_e, self.small_win)
        rgb_large = self._extract_large_windows(rgb_e, self.small_win, self.large_win)
        ir_large = self._extract_large_windows(ir_e, self.small_win, self.large_win)

        rgb_small = self.rgb_norm(rgb_small)
        ir_small = self.ir_norm(ir_small)
        rgb_large = self.rgb_norm(rgb_large)
        ir_large = self.ir_norm(ir_large)

        # 3) Top-k 前置：只对选中窗口做局部跨模态 attention
        if idx is None:
            rgb_cross_win = rgb_small.new_zeros(rgb_small.shape)
            ir_cross_win = ir_small.new_zeros(ir_small.shape)
        else:
            rgb_small_sel = self._gather_window_tokens(rgb_small, idx)
            ir_small_sel = self._gather_window_tokens(ir_small, idx)
            rgb_large_sel = self._gather_window_tokens(rgb_large, idx)
            ir_large_sel = self._gather_window_tokens(ir_large, idx)

            gate_r_sel = self._gather_window_gate(gate_r, idx)
            gate_i_sel = self._gather_window_gate(gate_i, idx)

            # matched cross evidence，不做 matched - source
            rgb_cross_sel = self.rgb_from_ir(rgb_small_sel, ir_large_sel) * gate_r_sel
            ir_cross_sel = self.ir_from_rgb(ir_small_sel, rgb_large_sel) * gate_i_sel

            rgb_cross_win = self._scatter_window_tokens(rgb_cross_sel, idx, rgb_small.shape)
            ir_cross_win = self._scatter_window_tokens(ir_cross_sel, idx, ir_small.shape)

        # 4) 窗口证据还原到二维特征图
        rgb_cross_e = self._window_reverse(rgb_cross_win, gh, gw, self.small_win)
        ir_cross_e = self._window_reverse(ir_cross_win, gh, gw, self.small_win)

        rgb_cross = self.rgb_delta_out(rgb_cross_e)
        ir_cross = self.ir_delta_out(ir_cross_e)

        # 5) 第一次局部增强：Top-k attention 的局部补充
        rgb_loc = rgb_pad + self.eta_r * rgb_cross
        ir_loc = ir_pad + self.eta_i * ir_cross

        # 6) SS2D Mamba 上下文传播：输入不是稀疏图，而是“局部增强特征 + cross evidence”
        rgb_ctx, ir_ctx, ctx = self.ss2d_context(rgb_loc, ir_loc, rgb_cross, ir_cross)

        # 7) RawCueAddC 风格最终融合
        Qr_map = self._local_mean(y, self.small_win).clamp(0.0, 1.0)
        Qt_map = self._local_mean(t, self.small_win).clamp(0.0, 1.0)

        fadd = rgb_ctx + ir_ctx
        fcue = Qr_map * rgb_ctx + Qt_map * ir_ctx
        out = fadd + self.raw_gamma * fcue
        out = out + self.beta * self.refine(out)
        out = self._remove_pad(out, pad_hw)

        if getattr(self, "enable_debug", False):
            topk_scalar = P.new_zeros(P.shape)
            if idx is not None:
                topk_scalar.scatter_(1, idx, 1.0)

            self.debug_info = {
                "C_map": self._remove_pad(self._window_scalar_to_map(C_score, gh, gw, self.small_win), pad_hw).detach().cpu(),
                "N_map": self._remove_pad(self._window_scalar_to_map(N_score, gh, gw, self.small_win), pad_hw).detach().cpu(),
                "P_map": self._remove_pad(self._window_scalar_to_map(P, gh, gw, self.small_win), pad_hw).detach().cpu(),
                "topk_map": self._remove_pad(self._window_scalar_to_map(topk_scalar, gh, gw, self.small_win), pad_hw).detach().cpu(),
                "Qr_map": self._remove_pad(Qr_map, pad_hw).detach().cpu(),
                "Qt_map": self._remove_pad(Qt_map, pad_hw).detach().cpu(),
                "rgb_cross": self._remove_pad(rgb_cross.abs().mean(dim=1, keepdim=True), pad_hw).detach().cpu(),
                "ir_cross": self._remove_pad(ir_cross.abs().mean(dim=1, keepdim=True), pad_hw).detach().cpu(),
                "ss2d_ctx": self._remove_pad(ctx.abs().mean(dim=1, keepdim=True), pad_hw).detach().cpu(),
                "out_energy": out.abs().mean(dim=1, keepdim=True).detach().cpu(),
            }
            if self.score_mode == "with_r_dark_ir":
                self.debug_info["P_normal_map"] = self._remove_pad(
                    self._window_scalar_to_map(P_normal, gh, gw, self.small_win), pad_hw
                ).detach().cpu()
                self.debug_info["P_dark_map"] = self._remove_pad(
                    self._window_scalar_to_map(P_dark, gh, gw, self.small_win), pad_hw
                ).detach().cpu()
                self.debug_info["dark_value"] = dark.detach().cpu()

        return out

"""
TSCI-v4: Shared Hidden Window Interaction + Reliability-Guided Window Mamba Propagation
=====================================================================================

This file is self-contained and can be copied into `ultralytics/nn/modules/block.py`,
or imported from there. It only depends on PyTorch. If `mamba_ssm` is available and the
input is on CUDA, the window-level SS2D branch uses Mamba; otherwise it falls back to a
lightweight convolutional propagation branch so that Ultralytics CPU dummy forward/model
summary will not crash.

Main module:
    TSCIv4SharedWindowMambaFusion

Expected input in YAML:
    x = [rgb_feat, ir_feat, raw_cue]  # raw_cue can be omitted; [rgb_feat, ir_feat] also works

Output:
    fused feature = rgb_out + ir_out

Suggested YAML args:
    [c, embed_dim, small_win, large_win, num_heads, topk_ratio,
     soft_train, eta_init, d_state, d_conv, expand, gamma_init, gate_bias_init, cpu_fast_init]

Example:
    - [[12, 13, 1], 1, TSCIv4SharedWindowMambaFusion,
       [128, 64, 4, 6, 4, 0.15, True, 0.03, 16, 4, 2, 0.0, -1.5]]
"""

# from __future__ import annotations



from mamba_ssm import Mamba
_HAS_MAMBA = True
# try:
#     from mamba_ssm import Mamba  # type: ignore
#     _HAS_MAMBA = True
# except Exception:  # pragma: no cover - mamba_ssm may be unavailable in CPU/model-info env
#     Mamba = None
#     _HAS_MAMBA = False


# -----------------------------------------------------------------------------
# Basic blocks
# -----------------------------------------------------------------------------


class ConvBNAct(nn.Module):
    """Minimal Conv-BN-SiLU block; independent of Ultralytics Conv."""

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1, p: Optional[int] = None,
                 groups: int = 1, act: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# -----------------------------------------------------------------------------
# Window utilities
# -----------------------------------------------------------------------------


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pad H/W to a multiple of `multiple` using replication padding."""
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


def _remove_pad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def _window_partition(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, int, int]:
    """
    [B,C,H,W] -> [B,L,win*win,C], where L=(H/win)*(W/win).
    H and W must be divisible by win.
    """
    b, c, h, w = x.shape
    assert h % win == 0 and w % win == 0, f"H={h}, W={w} must be divisible by win={win}"
    gh, gw = h // win, w // win
    x = x.view(b, c, gh, win, gw, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, gh * gw, win * win, c), gh, gw


def _window_reverse(windows: torch.Tensor, gh: int, gw: int, win: int) -> torch.Tensor:
    """[B,L,win*win,C] -> [B,C,H,W]."""
    b, l, tokens, c = windows.shape
    assert l == gh * gw and tokens == win * win
    x = windows.view(b, gh, gw, win, win, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, gh * win, gw * win)


def _extract_large_windows(x: torch.Tensor, small_win: int, large_win: int) -> torch.Tensor:
    """
    Extract one large search region for each non-overlapping small window.

    Input:
        x: [B,C,H,W]
    Output:
        patches: [B,L,large_win*large_win,C]
    """
    assert large_win >= small_win, f"large_win={large_win} should be >= small_win={small_win}"
    half = (large_win - small_win) // 2
    extra = (large_win - small_win) - half
    x_pad = F.pad(x, (half, extra, half, extra), mode="replicate")
    patches = F.unfold(x_pad, kernel_size=large_win, stride=small_win)  # [B,C*Lw*Lw,L]
    b, _, l = patches.shape
    c = x.shape[1]
    patches = patches.transpose(1, 2).contiguous().view(b, l, c, large_win * large_win)
    return patches.permute(0, 1, 3, 2).contiguous()  # [B,L,N,C]


def _window_avg_pool_tokens(x: torch.Tensor, win: int) -> Tuple[torch.Tensor, int, int]:
    """[B,C,H,W] -> [B,L,C] by non-overlapping window average pooling."""
    w, gh, gw = _window_partition(x, win)
    return w.mean(dim=2), gh, gw


def _gather_windows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather window tensor x=[B,L,N,C] with idx=[B,K] -> [B,K,N,C]."""
    b, l, n, c = x.shape
    k = idx.shape[1]
    index = idx[:, :, None, None].expand(b, k, n, c)
    return x.gather(dim=1, index=index)

def _scatter_windows(src: torch.Tensor, idx: torch.Tensor, out_shape: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
# def _scatter_windows(src: torch.Tensor, idx: torch.Tensor, out_shape: torch.Size | Tuple[int, ...]) -> torch.Tensor:
    """Scatter selected windows back into a zero tensor of shape [B,L,N,C]."""
    b, l, n, c = out_shape
    k = idx.shape[1]
    out = src.new_zeros((b, l, n, c))
    index = idx[:, :, None, None].expand(b, k, n, c)
    out.scatter_(dim=1, index=index, src=src)
    return out


def _scatter_window_scalar(src: torch.Tensor, idx: torch.Tensor, length: int) -> torch.Tensor:
    """Scatter [B,K,1] scalar tokens back to [B,L,1]."""
    b, k, one = src.shape
    out = src.new_zeros((b, length, one))
    index = idx[:, :, None].expand(b, k, one)
    out.scatter_(dim=1, index=index, src=src)
    return out


def _l2_normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + eps)


# -----------------------------------------------------------------------------
# Shared hidden comparison scorer
# -----------------------------------------------------------------------------


class SharedHiddenWindowScorer(nn.Module):
    """
    Estimate window-level cross-modal interaction demand in a shared comparison space.

    Key design:
      1) RGB/IR have separate modality adapters.
      2) A shared projector encodes both adapted features with the same parameters.
      3) Window relations are computed only after this shared projection.
    """

    def __init__(self, c: int, hidden_dim: int = 128, win: int = 4, eps: float = 1e-6):
        super().__init__()
        self.win = win
        self.eps = eps

        self.rgb_adapter = nn.Sequential(
            ConvBNAct(c, hidden_dim, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, groups=max(hidden_dim // 16, 1)),
        )
        self.ir_adapter = nn.Sequential(
            ConvBNAct(c, hidden_dim, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, groups=max(hidden_dim // 16, 1)),
        )

        # Shared comparison projector: same weights for RGB and IR.
        self.shared_projector = nn.Sequential(
            ConvBNAct(hidden_dim, hidden_dim, 1),
            ConvBNAct(hidden_dim, hidden_dim, 3, groups=hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
        )

        # score input: diff, inconsistency, energy_gap, object_proxy, raw_rgb_quality, raw_ir_quality
        self.score_mlp = nn.Sequential(
            nn.Linear(6, 24),
            nn.SiLU(inplace=True),
            nn.Linear(24, 1),
        )

    @staticmethod
    def _resize_raw_cue(raw_cue: Optional[torch.Tensor], size: Tuple[int, int], device: torch.device,
                        dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return resized RGB brightness and IR intensity maps, or neutral 0.5 maps if cue is absent."""
        h, w = size
        if raw_cue is None:
            q = torch.full((1, 1, h, w), 0.5, device=device, dtype=dtype)
            return q, q
        raw_cue = F.interpolate(raw_cue, size=size, mode="bilinear", align_corners=False)
        y = raw_cue[:, 0:1].clamp(0.0, 1.0)
        t = raw_cue[:, 1:2].clamp(0.0, 1.0)
        return y, t

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor, raw_cue: Optional[torch.Tensor] = None):
        # Modality-specific adaptation
        a_rgb = self.rgb_adapter(rgb)
        a_ir = self.ir_adapter(ir)

        # Shared comparison projection
        z_rgb = self.shared_projector(a_rgb)
        z_ir = self.shared_projector(a_ir)
        z_rgb = _l2_normalize_map(z_rgb, self.eps)
        z_ir = _l2_normalize_map(z_ir, self.eps)

        # Window tokens in shared space
        t_rgb, gh, gw = _window_avg_pool_tokens(z_rgb, self.win)  # [B,L,D]
        t_ir, _, _ = _window_avg_pool_tokens(z_ir, self.win)

        diff = (t_rgb - t_ir).abs().mean(dim=-1, keepdim=True)  # [B,L,1]
        cos = F.cosine_similarity(t_rgb, t_ir, dim=-1, eps=self.eps).unsqueeze(-1)
        incons = (1.0 - cos).clamp(0.0, 2.0)

        rgb_energy = t_rgb.abs().mean(dim=-1, keepdim=True)
        ir_energy = t_ir.abs().mean(dim=-1, keepdim=True)
        energy_gap = (rgb_energy - ir_energy).abs()
        obj_proxy = torch.maximum(rgb_energy, ir_energy)

        # Raw cue window quality as additional low-level prior, not as final decision.
        y, t = self._resize_raw_cue(raw_cue, rgb.shape[-2:], rgb.device, rgb.dtype)
        # If raw_cue absent, y/t have batch=1 and will broadcast after pooling.
        if y.shape[0] == 1 and rgb.shape[0] > 1:
            y = y.expand(rgb.shape[0], -1, -1, -1)
            t = t.expand(rgb.shape[0], -1, -1, -1)
        y_win, _, _ = _window_avg_pool_tokens(y, self.win)  # [B,L,1]
        t_win, _, _ = _window_avg_pool_tokens(t, self.win)

        score_in = torch.cat([diff, incons, energy_gap, obj_proxy, y_win, t_win], dim=-1)
        p_win = torch.sigmoid(self.score_mlp(score_in)).squeeze(-1)  # [B,L]
        return p_win, z_rgb, z_ir, gh, gw


# -----------------------------------------------------------------------------
# Local asymmetric window attention
# -----------------------------------------------------------------------------


class MatchedWindowAttentionWithReliability(nn.Module):
    """
    Cross-modal local attention for selected/all windows.

    Returns:
        matched: [B,K,Ns,C]
        rel    : [B,K,1], attention reliability estimated by normalized entropy.
    """

    def __init__(self, dim: int, num_heads: int = 4, eps: float = 1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.eps = eps

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, src_small: torch.Tensor, ref_large: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, k, ns, c = src_small.shape
        nl = ref_large.shape[2]

        q = self.q_proj(src_small).view(b, k, ns, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        kk = self.k_proj(ref_large).view(b, k, nl, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(ref_large).view(b, k, nl, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = (q @ kk.transpose(-2, -1)) * self.scale  # [B,K,H,Ns,Nl]
        attn = attn.softmax(dim=-1)
        matched = attn @ v
        matched = matched.permute(0, 1, 3, 2, 4).contiguous().view(b, k, ns, c)

        # Reliability: 1 - normalized entropy. High when attention is sharp.
        entropy = -(attn.clamp_min(self.eps) * attn.clamp_min(self.eps).log()).sum(dim=-1)  # [B,K,H,Ns]
        rel = 1.0 - entropy / math.log(max(nl, 2))
        rel = rel.mean(dim=(2, 3)).clamp(0.0, 1.0).unsqueeze(-1)  # [B,K,1]
        return self.o_proj(matched), rel


# -----------------------------------------------------------------------------
# Window-level SS2D/Mamba propagation
# -----------------------------------------------------------------------------


class WindowSS2D(nn.Module):
    """
    Four-direction state propagation over the window grid.

    If mamba_ssm is unavailable or input is on CPU, a depthwise-conv fallback is used.
    This avoids Ultralytics CPU dummy-forward crashes.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.has_mamba = _HAS_MAMBA

        if self.has_mamba:
            self.m_lr = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.m_rl = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.m_tb = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.m_bt = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            self.norm = nn.LayerNorm(d_model)
        else:
            self.m_lr = self.m_rl = self.m_tb = self.m_bt = None
            self.norm = nn.LayerNorm(d_model)

        self.fallback = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model, bias=False),
            nn.BatchNorm2d(d_model),
            nn.SiLU(inplace=True),
            nn.Conv2d(d_model, d_model, 1, bias=False),
            nn.BatchNorm2d(d_model),
        )
        self.out = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def _mamba_lr(self, x_map: torch.Tensor) -> torch.Tensor:
        # x_map: [B,D,GH,GW]
        b, d, gh, gw = x_map.shape
        seq = x_map.permute(0, 2, 3, 1).contiguous().view(b * gh, gw, d)
        seq = self.m_lr(self.norm(seq))
        return seq.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()

    def _mamba_rl(self, x_map: torch.Tensor) -> torch.Tensor:
        b, d, gh, gw = x_map.shape
        xf = x_map.flip(-1).contiguous()
        seq = xf.permute(0, 2, 3, 1).contiguous().view(b * gh, gw, d)
        seq = self.m_rl(self.norm(seq))
        y = seq.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()
        return y.flip(-1).contiguous()

    def _mamba_tb(self, x_map: torch.Tensor) -> torch.Tensor:
        b, d, gh, gw = x_map.shape
        seq = x_map.permute(0, 3, 2, 1).contiguous().view(b * gw, gh, d)
        seq = self.m_tb(self.norm(seq))
        return seq.view(b, gw, gh, d).permute(0, 3, 2, 1).contiguous()

    def _mamba_bt(self, x_map: torch.Tensor) -> torch.Tensor:
        b, d, gh, gw = x_map.shape
        xf = x_map.flip(-2).contiguous()
        seq = xf.permute(0, 3, 2, 1).contiguous().view(b * gw, gh, d)
        seq = self.m_bt(self.norm(seq))
        y = seq.view(b, gw, gh, d).permute(0, 3, 2, 1).contiguous()
        return y.flip(-2).contiguous()

    def forward(self, x: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        # x: [B,L,D]
        b, l, d = x.shape
        assert l == gh * gw
        x_map = x.view(b, gh, gw, d).permute(0, 3, 1, 2).contiguous()

        # CPU/model-info fallback. mamba_ssm Mamba normally depends on CUDA kernels.
        if (not self.has_mamba) or (not x.is_cuda):
            # print("[TSCIv4] WindowSS2D uses fallback. has_mamba =", self.has_mamba, "is_cuda =", x.is_cuda)
            y_map = self.fallback(x_map)
        else:
            # print("[TSCIv4] WindowSS2D uses REAL Mamba.")
            y_map = (self._mamba_lr(x_map) + self._mamba_rl(x_map) + self._mamba_tb(x_map) + self._mamba_bt(x_map)) * 0.25

        y = y_map.permute(0, 2, 3, 1).contiguous().view(b, l, d)
        return self.out(y)

class WindowS6_2D(WindowSS2D):
    pass

#v4
class ReliabilityGuidedWindowStatePropagation(nn.Module):
    """
    Propagate sparse cross-modal evidence on the window grid, then write back safely.

    gamma controls the global strength of this branch.
    g_win controls where the propagated state is reliable enough to be written back.
    """

    def __init__(self, c: int, d_model: int = 128, win: int = 4, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, gamma_init: float = 0.0, gate_bias_init: float = -1.5):
        super().__init__()
        self.c = c
        self.win = win
        self.d_model = d_model

        # token input: T_rgb, T_ir, T_rgb_cross, T_ir_cross, p, reliability, evidence_strength = 4C + 3
        self.token_proj = nn.Sequential(
            nn.Linear(4 * c + 3, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.ss2d = WindowSS2D(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

        self.gate = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        # Make gate conservative at the beginning.
        with torch.no_grad():
            last = self.gate[-2]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.constant_(last.bias, gate_bias_init)

        # Token -> patch update, allowing intra-window spatial update rather than constant window update.
        self.to_rgb_patch = nn.Linear(d_model, c * win * win)
        self.to_ir_patch = nn.Linear(d_model, c * win * win)

        self.gamma_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_ir = nn.Parameter(torch.tensor(float(gamma_init)))

    def _token_to_map(self, patch_tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        # patch_tokens: [B,L,C*win*win] -> [B,C,H,W]
        b, l, _ = patch_tokens.shape
        patch = patch_tokens.view(b, l, self.win * self.win, self.c)
        return _window_reverse(patch, gh, gw, self.win)

    def forward(self, rgb_loc: torch.Tensor, ir_loc: torch.Tensor, rgb_cross: torch.Tensor, ir_cross: torch.Tensor,
                p_win: torch.Tensor, rel_win: torch.Tensor, gh: int, gw: int):
        # Window tokens
        t_rgb, _, _ = _window_avg_pool_tokens(rgb_loc, self.win)
        t_ir, _, _ = _window_avg_pool_tokens(ir_loc, self.win)
        t_cr, _, _ = _window_avg_pool_tokens(rgb_cross, self.win)
        t_ci, _, _ = _window_avg_pool_tokens(ir_cross, self.win)

        e_cr = t_cr.abs().mean(dim=-1, keepdim=True)
        e_ci = t_ci.abs().mean(dim=-1, keepdim=True)
        e_win = 0.5 * (e_cr + e_ci)  # [B,L,1]

        if rel_win.dim() == 2:
            rel_win = rel_win.unsqueeze(-1)
        if p_win.dim() == 2:
            p_in = p_win.unsqueeze(-1)
        else:
            p_in = p_win

        gate_in = torch.cat([p_in, rel_win, e_win], dim=-1)
        g_win = self.gate(gate_in)  # [B,L,1]

        token_in = torch.cat([t_rgb, t_ir, t_cr, t_ci, p_in, rel_win, e_win], dim=-1)
        x = self.token_proj(token_in)

        # Reliability controls state input and write-back.
        # x = g_win * x
        # z = self.ss2d(x, gh, gw)
        # delta_rgb_token = g_win * self.to_rgb_patch(z)
        # delta_ir_token = g_win * self.to_ir_patch(z)
        # delta_rgb = self._token_to_map(delta_rgb_token, gh, gw)
        # delta_ir = self._token_to_map(delta_ir_token, gh, gw)

        # No reliability gate in the first diagnostic version.
        # Let Mamba see all window states, and use only gamma to control residual strength.
        z = self.ss2d(x, gh, gw)
        delta_rgb_token = self.to_rgb_patch(z)
        delta_ir_token = self.to_ir_patch(z)
        delta_rgb = self._token_to_map(delta_rgb_token, gh, gw)
        delta_ir = self._token_to_map(delta_ir_token, gh, gw)


        rgb_out = rgb_loc + self.gamma_rgb * delta_rgb
        ir_out = ir_loc + self.gamma_ir * delta_ir
        return rgb_out, ir_out, g_win

#v4.1
class DualInputWindowS6Fusion(nn.Module):
    """
    Dual-input window-level S6 fusion.

    Compared with the previous reliability-gated version:
      1. RGB and IR are encoded as two separate state inputs.
      2. Cross evidence is injected into the corresponding modality input.
      3. beta controls modality write-in strength.
      4. No external write-back gate is used.
      5. gamma only controls global residual strength.
    """

    def __init__(
        self,
        c: int,
        d_model: int = 128,
        win: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        gamma_init: float = 0.02,
    ):
        super().__init__()
        self.c = c
        self.win = win
        self.d_model = d_model

        # RGB input: RGB token + IR->RGB evidence + condition
        self.rgb_in = nn.Sequential(
            nn.Linear(2 * c + 3, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # IR input: IR token + RGB->IR evidence + condition
        self.ir_in = nn.Sequential(
            nn.Linear(2 * c + 3, d_model),
            nn.SiLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

        # beta controls modality write-in strength
        self.beta_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.SiLU(inplace=True),
            nn.Linear(16, 2),
        )

        self.s6_rgb = WindowSS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.s6_ir = WindowSS2D(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

        self.out_norm = nn.LayerNorm(d_model)

        self.to_rgb_patch = nn.Linear(d_model, c * win * win)
        self.to_ir_patch = nn.Linear(d_model, c * win * win)

        self.gamma_rgb = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_ir = nn.Parameter(torch.tensor(float(gamma_init)))

    def _token_to_map(self, patch_tokens: torch.Tensor, gh: int, gw: int) -> torch.Tensor:
        b, l, _ = patch_tokens.shape
        patch = patch_tokens.view(b, l, self.win * self.win, self.c)
        return _window_reverse(patch, gh, gw, self.win)

    def forward(
        self,
        rgb_loc: torch.Tensor,
        ir_loc: torch.Tensor,
        rgb_cross: torch.Tensor,
        ir_cross: torch.Tensor,
        p_win: torch.Tensor,
        rel_win: torch.Tensor,
        gh: int,
        gw: int,
    ):
        # 1. Window tokens
        t_rgb, _, _ = _window_avg_pool_tokens(rgb_loc, self.win)
        t_ir, _, _ = _window_avg_pool_tokens(ir_loc, self.win)
        t_cr, _, _ = _window_avg_pool_tokens(rgb_cross, self.win)
        t_ci, _, _ = _window_avg_pool_tokens(ir_cross, self.win)

        # 2. Evidence strength
        e_cr = t_cr.abs().mean(dim=-1, keepdim=True)
        e_ci = t_ci.abs().mean(dim=-1, keepdim=True)
        e_win = 0.5 * (e_cr + e_ci)

        if rel_win.dim() == 2:
            rel_win = rel_win.unsqueeze(-1)

        if p_win.dim() == 2:
            p_in = p_win.unsqueeze(-1)
        else:
            p_in = p_win

        # condition = interaction demand + matching reliability + evidence strength
        cond = torch.cat([p_in, rel_win, e_win], dim=-1)

        # 3. Dual input state tokens
        x_rgb = self.rgb_in(torch.cat([t_rgb, t_cr, cond], dim=-1))
        x_ir = self.ir_in(torch.cat([t_ir, t_ci, cond], dim=-1))

        # 4. Modality write-in weights
        beta = torch.softmax(self.beta_mlp(cond), dim=-1)
        beta_rgb = beta[..., 0:1]
        beta_ir = beta[..., 1:2]

        x_rgb = beta_rgb * x_rgb
        x_ir = beta_ir * x_ir

        # 5. Dual S6 state propagation
        z_rgb = self.s6_rgb(x_rgb, gh, gw)
        z_ir = self.s6_ir(x_ir, gh, gw)

        z = self.out_norm(z_rgb + z_ir)

        # 6. Project fusion state back to RGB / IR update
        delta_rgb_token = self.to_rgb_patch(z)
        delta_ir_token = self.to_ir_patch(z)

        delta_rgb = self._token_to_map(delta_rgb_token, gh, gw)
        delta_ir = self._token_to_map(delta_ir_token, gh, gw)

        # 7. Residual write-back
        rgb_out = rgb_loc + self.gamma_rgb * delta_rgb
        ir_out = ir_loc + self.gamma_ir * delta_ir

        debug = {
            "beta_rgb": beta_rgb,
            "beta_ir": beta_ir,
            "state_energy": z.abs().mean(dim=-1, keepdim=True),
        }

        return rgb_out, ir_out, debug


# -----------------------------------------------------------------------------
# Main V4 fusion module
# -----------------------------------------------------------------------------


class TSCIv4SharedWindowMambaFusion(nn.Module):
    """
    V4 module: problem-driven version.

    Stages:
      1) Shared hidden window scorer estimates interaction demand.
      2) Selected/soft windows perform asymmetric local RGB-IR matching.
      3) Sparse cross evidence locally updates both streams.
      4) Window-level reliability-guided Mamba propagates evidence states.
      5) Final fusion is direct Add: F_out = F_rgb_out + F_ir_out.

    Inputs:
      x = [rgb_feat, ir_feat, raw_cue] or [rgb_feat, ir_feat]

    Args for YAML:
      [c, embed_dim, small_win, large_win, num_heads, topk_ratio,
       soft_train, eta_init, d_state, d_conv, expand, gamma_init, gate_bias_init, cpu_fast_init]
    """

    def __init__(self, c: int, embed_dim: int = 128, small_win: int = 4, large_win: int = 6,
                 num_heads: int = 4, topk_ratio: float = 0.25, soft_train: bool = True,
                 eta_init: float = 0.03, d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 gamma_init: float = 0.0, gate_bias_init: float = -1.5, eps: float = 1e-6,
                 cpu_fast_init: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
        assert large_win >= small_win, f"large_win={large_win} should be >= small_win={small_win}"
        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.topk_ratio = float(topk_ratio)
        self.soft_train = bool(soft_train)
        self.soft_all = bool(soft_train)  # for clarity in code
        self.eps = eps
        # Skip heavy custom computation on CPU dummy forward/model summary.
        # Set cpu_fast_init=False if you really want to train/evaluate this module on CPU.
        self.cpu_fast_init = bool(cpu_fast_init)

        self.scorer = SharedHiddenWindowScorer(c=c, hidden_dim=embed_dim, win=small_win, eps=eps)
        self.rgb_from_ir = MatchedWindowAttentionWithReliability(embed_dim, num_heads, eps=eps)
        self.ir_from_rgb = MatchedWindowAttentionWithReliability(embed_dim, num_heads, eps=eps)

        self.rgb_cross_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_cross_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.eta_rgb = nn.Parameter(torch.tensor(float(eta_init)))
        self.eta_ir = nn.Parameter(torch.tensor(float(eta_init)))

        #v4
        # self.state_prop = ReliabilityGuidedWindowStatePropagation(
        #     c=c,
        #     d_model=embed_dim,
        #     win=small_win,
        #     d_state=d_state,
        #     d_conv=d_conv,
        #     expand=expand,
        #     gamma_init=gamma_init,
        #     gate_bias_init=gate_bias_init,
        # )

        #v4.1
        self.state_prop = DualInputWindowS6Fusion(
            c=c,
            d_model=embed_dim,
            win=small_win,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            gamma_init=gamma_init,
        )

        # Optional tiny post-add refinement. Kept closed by default to preserve "direct Add" interpretation.
        self.post_refine = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, groups=max(c // 16, 1), bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.post_beta = nn.Parameter(torch.zeros(1))

        self.debug_info = {}

    def _topk_indices(self, p_win: torch.Tensor) -> torch.Tensor:
        b, l = p_win.shape
        k = max(1, int(math.ceil(l * self.topk_ratio)))
        k = min(k, l)
        return torch.topk(p_win, k=k, dim=1).indices

    @staticmethod
    def _map_scalar_to_windows(x: torch.Tensor, idx: torch.Tensor, length: int) -> torch.Tensor:
        return _scatter_window_scalar(x, idx, length)

    def _cross_attention_hard(self, z_rgb: torch.Tensor, z_ir: torch.Tensor, p_win: torch.Tensor):
        """Hard Top-k attention path for inference or efficient training."""
        rgb_small, gh, gw = _window_partition(z_rgb, self.small_win)
        ir_small, _, _ = _window_partition(z_ir, self.small_win)
        rgb_large = _extract_large_windows(z_rgb, self.small_win, self.large_win)
        ir_large = _extract_large_windows(z_ir, self.small_win, self.large_win)

        b, l, ns, d = rgb_small.shape
        idx = self._topk_indices(p_win)

        rgb_small_sel = _gather_windows(rgb_small, idx)
        ir_small_sel = _gather_windows(ir_small, idx)
        rgb_large_sel = _gather_windows(rgb_large, idx)
        ir_large_sel = _gather_windows(ir_large, idx)

        p_sel = p_win.gather(1, idx).unsqueeze(-1).unsqueeze(-1)  # [B,K,1,1]

        rgb_cross_sel, rel_r_sel = self.rgb_from_ir(rgb_small_sel, ir_large_sel)
        ir_cross_sel, rel_i_sel = self.ir_from_rgb(ir_small_sel, rgb_large_sel)

        # p_win is a proposal strength, not a hard truth. It modulates evidence strength.
        rgb_cross_sel = p_sel * rgb_cross_sel
        ir_cross_sel = p_sel * ir_cross_sel

        rgb_cross_win = _scatter_windows(rgb_cross_sel, idx, rgb_small.shape)
        ir_cross_win = _scatter_windows(ir_cross_sel, idx, ir_small.shape)

        rel_sel = 0.5 * (rel_r_sel + rel_i_sel)  # [B,K,1]
        rel_win = _scatter_window_scalar(rel_sel, idx, l)  # [B,L,1]

        return rgb_cross_win, ir_cross_win, rel_win, gh, gw, idx

    def _cross_attention_soft_all(self, z_rgb: torch.Tensor, z_ir: torch.Tensor, p_win: torch.Tensor):
        """Soft all-window path. More expensive but gives gradients to all window scores during training."""
        rgb_small, gh, gw = _window_partition(z_rgb, self.small_win)
        ir_small, _, _ = _window_partition(z_ir, self.small_win)
        rgb_large = _extract_large_windows(z_rgb, self.small_win, self.large_win)
        ir_large = _extract_large_windows(z_ir, self.small_win, self.large_win)

        p = p_win.unsqueeze(-1).unsqueeze(-1)  # [B,L,1,1]
        rgb_cross_win, rel_r = self.rgb_from_ir(rgb_small, ir_large)
        ir_cross_win, rel_i = self.ir_from_rgb(ir_small, rgb_large)
        rgb_cross_win = p * rgb_cross_win
        ir_cross_win = p * ir_cross_win
        rel_win = 0.5 * (rel_r + rel_i)  # [B,L,1]
        idx = None
        return rgb_cross_win, ir_cross_win, rel_win, gh, gw, idx
    
    def _win_scalar_to_map(v, gh, gw, h_pad, w_pad):
        # v: [B, L, 1] or [B, L]
        if v.dim() == 2:
            v = v.unsqueeze(-1)
        b = v.shape[0]
        m = v.view(b, gh, gw, 1).permute(0, 3, 1, 2).contiguous()
        m = F.interpolate(m, size=(h_pad, w_pad), mode="nearest")
        return m

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"TSCIv4SharedWindowMambaFusion expects list/tuple input, got {type(x)}")
        if len(x) == 3:
            rgb, ir, raw_cue = x[0], x[1], x[2]
        elif len(x) == 2:
            rgb, ir = x[0], x[1]
            raw_cue = None
        else:
            raise ValueError(f"TSCIv4SharedWindowMambaFusion expects [rgb, ir, raw_cue] or [rgb, ir], got len={len(x)}")

        assert rgb.shape == ir.shape, f"rgb and ir feature mismatch: {rgb.shape} vs {ir.shape}"
        b, c, h0, w0 = rgb.shape
        assert c == self.c, f"channel mismatch: module c={self.c}, input C={c}"

        if self.cpu_fast_init and (not rgb.is_cuda):
            return rgb + ir

        rgb_pad, pad_hw = _pad_to_multiple(rgb, self.small_win)
        ir_pad, _ = _pad_to_multiple(ir, self.small_win)
        raw_cue_pad = None
        if raw_cue is not None:
            raw_cue_pad = F.interpolate(raw_cue, size=rgb_pad.shape[-2:], mode="bilinear", align_corners=False)

        # 1) Shared hidden space scorer
        p_win, z_rgb, z_ir, gh, gw = self.scorer(rgb_pad, ir_pad, raw_cue_pad)

        # 2) Local cross-modal window matching
        if self.soft_all:     
        # if self.training and self.soft_train:  
            rgb_cross_win, ir_cross_win, rel_win, gh, gw, idx = self._cross_attention_soft_all(z_rgb, z_ir, p_win)
        else:
            rgb_cross_win, ir_cross_win, rel_win, gh, gw, idx = self._cross_attention_hard(z_rgb, z_ir, p_win)

        rgb_cross_e = _window_reverse(rgb_cross_win, gh, gw, self.small_win)
        ir_cross_e = _window_reverse(ir_cross_win, gh, gw, self.small_win)
        rgb_cross = self.rgb_cross_out(rgb_cross_e)
        ir_cross = self.ir_cross_out(ir_cross_e)

        # 3) Local cross-evidence injection
        rgb_loc = rgb_pad + self.eta_rgb * rgb_cross
        ir_loc = ir_pad + self.eta_ir * ir_cross

        # 4) Reliability-guided window state propagation v4.0
        # rgb_out, ir_out, g_win = self.state_prop(
        #     rgb_loc=rgb_loc,
        #     ir_loc=ir_loc,
        #     rgb_cross=rgb_cross,
        #     ir_cross=ir_cross,
        #     p_win=p_win,
        #     rel_win=rel_win,
        #     gh=gh,
        #     gw=gw,
        # )

        #v4.1
        rgb_out, ir_out, state_debug = self.state_prop(
            rgb_loc=rgb_loc,
            ir_loc=ir_loc,
            rgb_cross=rgb_cross,
            ir_cross=ir_cross,
            p_win=p_win,
            rel_win=rel_win,
            gh=gh,
            gw=gw,
        )

        # 5) Direct Add before detection head
        fused = rgb_out + ir_out
        fused = fused + self.post_beta * self.post_refine(fused)
        fused = _remove_pad(fused, pad_hw)

        #v4.0 debug maps
        # if getattr(self, "enable_debug", False):
        #     with torch.no_grad():
        #         p_map = p_win.view(b, gh, gw).unsqueeze(1)
        #         p_map = F.interpolate(p_map, size=(rgb_pad.shape[-2], rgb_pad.shape[-1]), mode="nearest")
        #         g_map = g_win.view(b, gh, gw, 1).permute(0, 3, 1, 2).contiguous()
        #         g_map = F.interpolate(g_map, size=(rgb_pad.shape[-2], rgb_pad.shape[-1]), mode="nearest")
        #         rel_map = rel_win.view(b, gh, gw, 1).permute(0, 3, 1, 2).contiguous()
        #         rel_map = F.interpolate(rel_map, size=(rgb_pad.shape[-2], rgb_pad.shape[-1]), mode="nearest")

        #         self.debug_info = {
        #             "p_win_map": _remove_pad(p_map, pad_hw).detach().cpu(),
        #             "reliability_map": _remove_pad(rel_map, pad_hw).detach().cpu(),
        #             "write_gate_map": _remove_pad(g_map, pad_hw).detach().cpu(),
        #             "rgb_cross": _remove_pad(rgb_cross.abs().mean(1, keepdim=True), pad_hw).detach().cpu(),
        #             "ir_cross": _remove_pad(ir_cross.abs().mean(1, keepdim=True), pad_hw).detach().cpu(),
        #             "fused_energy": fused.abs().mean(1, keepdim=True).detach().cpu(),
        #         }
        #         if idx is not None:
        #             topk = torch.zeros_like(p_win)
        #             topk.scatter_(1, idx, 1.0)
        #             topk_map = topk.view(b, gh, gw).unsqueeze(1)
        #             topk_map = F.interpolate(topk_map, size=(rgb_pad.shape[-2], rgb_pad.shape[-1]), mode="nearest")
        #             self.debug_info["topk_map"] = _remove_pad(topk_map, pad_hw).detach().cpu()

        # Debug maps for v4.1
        if getattr(self, "enable_debug", False):
            # ------------------------------------------------------------
            # Debug maps for TSCIv4.1
            # Required variables from forward:
            #   p_win:        [B, L]
            #   rel_win:      [B, L] or [B, L, 1]
            #   state_debug:  dict with beta_rgb, beta_ir, state_energy
            #   rgb_cross:    [B, C, H_pad, W_pad]
            #   ir_cross:     [B, C, H_pad, W_pad]
            #   rgb_out:      [B, C, H_pad, W_pad]
            #   ir_out:       [B, C, H_pad, W_pad]
            #   fused:        [B, C, H_pad, W_pad]
            #   gh, gw:       window grid size
            #   pad_hw:       padding info used by _remove_pad()
            # ------------------------------------------------------------
            with torch.no_grad():
                def _win_scalar_to_map(v, gh, gw, h_pad, w_pad):
                    """
                    Convert window-level scalar tensor to spatial map.
                    Args:
                        v: [B, L] or [B, L, 1]
                        gh, gw: window grid height / width
                        h_pad, w_pad: padded feature map size
                    Returns:
                        map: [B, 1, H_pad, W_pad]
                    """
                    if v is None:
                        return None

                    if v.dim() == 2:
                        v = v.unsqueeze(-1)  # [B, L, 1]

                    b, l, c = v.shape
                    assert c == 1, f"Expected scalar window value with shape [B, L, 1], got {v.shape}"
                    assert l == gh * gw, f"Window number mismatch: L={l}, gh*gw={gh * gw}"

                    m = v.view(b, gh, gw, 1).permute(0, 3, 1, 2).contiguous()
                    m = F.interpolate(m, size=(h_pad, w_pad), mode="nearest")
                    return m

                h_pad, w_pad = rgb_pad.shape[-2], rgb_pad.shape[-1]
                # --------------------------------------------------------
                # 1. Window interaction demand score
                # --------------------------------------------------------
                p_win_map = _win_scalar_to_map(
                    p_win.detach(),
                    gh,
                    gw,
                    h_pad,
                    w_pad,
                )
                # --------------------------------------------------------
                # 2. Matching reliability map
                # --------------------------------------------------------
                rel_map = _win_scalar_to_map(
                    rel_win.detach(),
                    gh,
                    gw,
                    h_pad,
                    w_pad,
                )
                # --------------------------------------------------------
                # 3. Dual-input S6 state write-in weights
                # --------------------------------------------------------
                beta_rgb = state_debug.get("beta_rgb", None)
                beta_ir = state_debug.get("beta_ir", None)
                state_energy = state_debug.get("state_energy", None)
                beta_rgb_map = _win_scalar_to_map(
                    beta_rgb.detach() if beta_rgb is not None else None,
                    gh,
                    gw,
                    h_pad,
                    w_pad,
                )
                beta_ir_map = _win_scalar_to_map(
                    beta_ir.detach() if beta_ir is not None else None,
                    gh,
                    gw,
                    h_pad,
                    w_pad,
                )
                state_energy_map = _win_scalar_to_map(
                    state_energy.detach() if state_energy is not None else None,
                    gh,
                    gw,
                    h_pad,
                    w_pad,
                )
                # --------------------------------------------------------
                # 4. Optional visualization of high-score windows
                #    In soft_all mode, this is only for visualization.
                #    It does NOT mean hard Top-k is used in forward.
                # --------------------------------------------------------
                try:
                    k_vis = max(1, int(math.ceil(p_win.shape[1] * self.topk_ratio)))
                    idx_vis = torch.topk(p_win.detach(), k=k_vis, dim=1).indices

                    topk_vis = torch.zeros_like(p_win.detach())
                    topk_vis.scatter_(1, idx_vis, 1.0)

                    topk_vis_map = _win_scalar_to_map(
                        topk_vis,
                        gh,
                        gw,
                        h_pad,
                        w_pad,
                    )
                except Exception:
                    topk_vis_map = None
                # --------------------------------------------------------
                # 5. Feature energy maps
                # --------------------------------------------------------
                rgb_cross_energy = rgb_cross.detach().abs().mean(dim=1, keepdim=True)
                ir_cross_energy = ir_cross.detach().abs().mean(dim=1, keepdim=True)
                rgb_out_energy = rgb_out.detach().abs().mean(dim=1, keepdim=True)
                ir_out_energy = ir_out.detach().abs().mean(dim=1, keepdim=True)
                fused_energy = fused.detach().abs().mean(dim=1, keepdim=True)
                # --------------------------------------------------------
                # 6. Save debug information
                # --------------------------------------------------------
                debug_dict = {
                    "p_win_map": _remove_pad(p_win_map, pad_hw).detach().cpu(),
                    "reliability_map": _remove_pad(rel_map, pad_hw).detach().cpu(),

                    "rgb_cross_energy": _remove_pad(rgb_cross_energy, pad_hw).detach().cpu(),
                    "ir_cross_energy": _remove_pad(ir_cross_energy, pad_hw).detach().cpu(),

                    "rgb_out_energy": _remove_pad(rgb_out_energy, pad_hw).detach().cpu(),
                    "ir_out_energy": _remove_pad(ir_out_energy, pad_hw).detach().cpu(),
                    "fused_energy": _remove_pad(fused_energy, pad_hw).detach().cpu(),
                }
                if beta_rgb_map is not None:
                    debug_dict["beta_rgb_map"] = _remove_pad(beta_rgb_map, pad_hw).detach().cpu()
                if beta_ir_map is not None:
                    debug_dict["beta_ir_map"] = _remove_pad(beta_ir_map, pad_hw).detach().cpu()
                if state_energy_map is not None:
                    debug_dict["state_energy_map"] = _remove_pad(state_energy_map, pad_hw).detach().cpu()
                if topk_vis_map is not None:
                    debug_dict["topk_vis_map"] = _remove_pad(topk_vis_map, pad_hw).detach().cpu()
                self.debug_info = debug_dict

        return fused

# __all__ = [
#     "TSCIv4SharedWindowMambaFusion",
#     "SharedHiddenWindowScorer",
#     "MatchedWindowAttentionWithReliability",
#     "ReliabilityGuidedWindowStatePropagation",
#     "WindowSS2D",
# ]

class _ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSSF_SS2D(nn.Module):
    """
    Dual-input State Space Fusion with SS2D.

    输入:
        x = (rgb, ir)
        rgb: [B, C, H, W]
        ir : [B, C, H, W]

    输出:
        fused: [B, C_out, H, W]

    核心:
        1) RGB / IR 不 concat；
        2) 四方向扫描；
        3) RGB 和 IR 分别进入 selective scan；
        4) 两路 SSM 输出相加；
        5) out = RGB + IR + gamma * SS2D_residual。
    """

    def __init__(
        self,
        c1,
        c2=None,
        d_state=16,
        expand=1.0,
        dt_rank="auto",
        dropout=0.0,
        gamma_init=1e-3,
    ):
        super().__init__()

        if isinstance(c1, (list, tuple)):
            assert len(c1) == 2, f"DSSF_SS2D expects two inputs, got {c1}"
            assert c1[0] == c1[1], f"RGB/IR channels must match, got {c1}"
            c1 = c1[0]

        c2 = c1 if c2 is None else c2

        self.c1 = c1
        self.c2 = c2
        self.d_state = d_state
        self.d_inner = int(c2 * expand)
        self.K = 4

        if dt_rank == "auto":
            dt_rank = math.ceil(self.d_inner / 16)
        self.dt_rank = dt_rank

        if selective_scan_fn is None:
            raise ImportError(
                "DSSF_SS2D requires mamba_ssm. "
                "Please install mamba_ssm or use a manual scan version."
            )

        # shortcut: 保证初始接近 Add
        if c1 == c2:
            self.rgb_short = nn.Identity()
            self.ir_short = nn.Identity()
        else:
            self.rgb_short = _ConvBNAct(c1, c2, k=1)
            self.ir_short = _ConvBNAct(c1, c2, k=1)

        # 输入投影
        self.rgb_proj = _ConvBNAct(c1, self.d_inner, k=1)
        self.ir_proj = _ConvBNAct(c1, self.d_inner, k=1)

        # 对 RGB / IR 分别生成 dt, B, C
        proj_out_dim = dt_rank + d_state * 2

        self.x_proj_rgb_weight = nn.Parameter(
            torch.empty(self.K, proj_out_dim, self.d_inner)
        )
        self.x_proj_ir_weight = nn.Parameter(
            torch.empty(self.K, proj_out_dim, self.d_inner)
        )

        self.dt_proj_rgb_weight = nn.Parameter(
            torch.empty(self.K, self.d_inner, dt_rank)
        )
        self.dt_proj_ir_weight = nn.Parameter(
            torch.empty(self.K, self.d_inner, dt_rank)
        )

        self.dt_proj_rgb_bias = nn.Parameter(
            torch.empty(self.K, self.d_inner)
        )
        self.dt_proj_ir_bias = nn.Parameter(
            torch.empty(self.K, self.d_inner)
        )

        # shared A: 四方向共享同一类状态形式，但每个方向/通道有独立参数
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A = A.view(1, 1, d_state).repeat(self.K, self.d_inner, 1)
        self.A_logs = nn.Parameter(torch.log(A).reshape(self.K * self.d_inner, d_state))

        # modality-specific D
        self.Ds_rgb = nn.Parameter(torch.ones(self.K * self.d_inner))
        self.Ds_ir = nn.Parameter(torch.ones(self.K * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = _ConvBNAct(self.d_inner, c2, k=1)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # residual scale
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.x_proj_rgb_weight)
        nn.init.xavier_uniform_(self.x_proj_ir_weight)

        nn.init.xavier_uniform_(self.dt_proj_rgb_weight)
        nn.init.xavier_uniform_(self.dt_proj_ir_weight)

        # dt bias 设小一点，避免初始扫描过激
        nn.init.constant_(self.dt_proj_rgb_bias, -2.0)
        nn.init.constant_(self.dt_proj_ir_bias, -2.0)

    @staticmethod
    def _cross_scan(x):
        """
        x: [B, C, H, W]
        return: [B, 4, C, L]
        directions:
            0: left -> right
            1: right -> left
            2: top -> bottom
            3: bottom -> top
        """
        B, C, H, W = x.shape

        x_lr = x.flatten(2)  # [B, C, H*W]

        x_rl = torch.flip(x, dims=[3]).flatten(2)

        x_tb = x.transpose(2, 3).contiguous().flatten(2)

        x_bt = torch.flip(x, dims=[2]).transpose(2, 3).contiguous().flatten(2)

        xs = torch.stack([x_lr, x_rl, x_tb, x_bt], dim=1)
        return xs

    @staticmethod
    def _cross_merge(ys, H, W):
        """
        ys: [B, 4, C, L]
        return: [B, C, H, W]
        """
        B, K, C, L = ys.shape
        assert K == 4

        y_lr = ys[:, 0].reshape(B, C, H, W)

        y_rl = ys[:, 1].reshape(B, C, H, W)
        y_rl = torch.flip(y_rl, dims=[3])

        y_tb = ys[:, 2].reshape(B, C, W, H)
        y_tb = y_tb.transpose(2, 3).contiguous()

        y_bt = ys[:, 3].reshape(B, C, W, H)
        y_bt = y_bt.transpose(2, 3).contiguous()
        y_bt = torch.flip(y_bt, dims=[2])

        y = y_lr + y_rl + y_tb + y_bt
        y = y * 0.25
        return y

    def _ss2d_one_modality(
        self,
        xs,
        x_proj_weight,
        dt_proj_weight,
        dt_proj_bias,
        A_logs,
        Ds,
    ):
        """
        xs: [B, 4, C_inner, L]
        return: [B, 4, C_inner, L]
        """
        B, K, D, L = xs.shape
        assert K == self.K
        assert D == self.d_inner

        # [B, K, proj_out_dim, L]
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, x_proj_weight)

        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2
        )

        # dts: [B, K, D, L]
        dts = torch.einsum("bkrl,kdr->bkdl", dts, dt_proj_weight)
        dts = dts + dt_proj_bias.view(1, K, D, 1)

        # reshape for selective_scan_fn
        xs_scan = xs.reshape(B, K * D, L)
        dts_scan = dts.reshape(B, K * D, L)

        # Bs/Cs: [B, K, d_state, L]
        Bs = Bs.contiguous()
        Cs = Cs.contiguous()

        As = -torch.exp(A_logs.float())  # [K*D, d_state]
        Ds = Ds.float()

        y = selective_scan_fn(
            xs_scan.float(),
            dts_scan.float(),
            As,
            Bs.float(),
            Cs.float(),
            Ds,
            z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )

        y = y.reshape(B, K, D, L)
        return y.to(xs.dtype)

    def forward(self, x):
        rgb, ir = x[0], x[1]
        B, C, H, W = rgb.shape

        base = self.rgb_short(rgb) + self.ir_short(ir)
        # YOLO model.info()/parse/build 阶段可能用 CPU dummy tensor
        # selective_scan_fn 要求 CUDA，所以 CPU 时直接退化为 Add
        if not rgb.is_cuda:
            return base

        xr = self.rgb_proj(rgb)
        xi = self.ir_proj(ir)

        xs_r = self._cross_scan(xr)
        xs_i = self._cross_scan(xi)

        ys_r = self._ss2d_one_modality(
            xs_r,
            self.x_proj_rgb_weight,
            self.dt_proj_rgb_weight,
            self.dt_proj_rgb_bias,
            self.A_logs,
            self.Ds_rgb,
        )

        ys_i = self._ss2d_one_modality(
            xs_i,
            self.x_proj_ir_weight,
            self.dt_proj_ir_weight,
            self.dt_proj_ir_bias,
            self.A_logs,
            self.Ds_ir,
        )

        # F2SSM-like: RGB SSM + IR SSM
        ys = ys_r + ys_i

        y = self._cross_merge(ys, H, W)

        # LayerNorm on channel-last
        y = y.permute(0, 2, 3, 1).contiguous()
        y = self.out_norm(y)
        y = y.permute(0, 3, 1, 2).contiguous()

        y = self.out_proj(y)
        y = self.dropout(y)

        out = base + self.gamma * y
        return out




# ============================================================
# Helper functions
# ============================================================

def _cmss_autopad_to_multiple(x, multiple: int):
    """
    x: [B, C, H, W]
    pad H/W to be divisible by multiple
    """
    B, C, H, W = x.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (H, W)


def _cmss_remove_pad(x, hw):
    H, W = hw
    return x[:, :, :H, :W].contiguous()


def _cmss_window_partition(x, win: int):
    """
    x: [B, C, H, W]
    return: [B*num_windows, win*win, C]
    """
    B, C, H, W = x.shape
    assert H % win == 0 and W % win == 0, \
        f"H={H}, W={W} must be divisible by win={win}"

    x = x.view(B, C, H // win, win, W // win, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    x = x.view(B * (H // win) * (W // win), win * win, C)
    return x


def _cmss_window_reverse(windows, win: int, H: int, W: int):
    """
    windows: [B*num_windows, win*win, C]
    return: [B, C, H, W]
    """
    num_win = (H // win) * (W // win)
    B = windows.shape[0] // num_win
    C = windows.shape[-1]

    x = windows.view(B, H // win, W // win, win, win, C)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, C, H, W)
    return x


def _cmss_extract_large_windows(x, small_win: int, large_win: int):
    """
    x: [B, C, H, W]
    return: [B*num_small_windows, large_win*large_win, C]

    large window is centered around the corresponding small window.
    Requirement: large_win - small_win should be even.
    Example: small_win=4, large_win=6 => radius=1
    """
    assert large_win >= small_win
    assert (large_win - small_win) % 2 == 0, \
        "For stable center alignment, require large_win = small_win + 2 * radius, e.g. 4/6 or 2/4."

    B, C, H, W = x.shape
    radius = (large_win - small_win) // 2

    # pad by radius, then unfold large window with stride = small_win
    x_pad = F.pad(x, (radius, radius, radius, radius), mode="replicate")
    patches = F.unfold(
        x_pad,
        kernel_size=large_win,
        stride=small_win,
        padding=0
    )  # [B, C*large_win*large_win, num_windows]

    patches = patches.transpose(1, 2).contiguous()
    patches = patches.view(B * patches.shape[1], C, large_win * large_win)
    patches = patches.transpose(1, 2).contiguous()  # [B*Nw, N*N, C]
    return patches


def _norm01_per_sample(x, eps=1e-6):
    """
    x: [B, Nw]
    normalize each image independently to [0, 1]
    """
    x_min = x.min(dim=1, keepdim=True)[0]
    x_max = x.max(dim=1, keepdim=True)[0]
    return (x - x_min) / (x_max - x_min + eps)


class _ConvBNAct(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, groups=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.block = nn.Sequential(
            nn.Conv2d(c1, c2, k, s, p, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# Branches
# ============================================================

class _CrossWindowAttention(nn.Module):
    """
    small-window query attends to large-window key/value.
    q_win:  [B*Nw, M*M, C]
    kv_win: [B*Nw, N*N, C]
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q_win, kv_win):
        Bn, Nq, C = q_win.shape
        Nk = kv_win.shape[1]
        h = self.num_heads
        d = self.head_dim

        q = self.q_proj(q_win).view(Bn, Nq, h, d).transpose(1, 2)
        k = self.k_proj(kv_win).view(Bn, Nk, h, d).transpose(1, 2)
        v = self.v_proj(kv_win).view(Bn, Nk, h, d).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(Bn, Nq, C)
        out = self.o_proj(out)
        return out


class _LightSameWindowFusion(nn.Module):
    """
    同位置轻交互分支。
    不做 large-window search，只在对应小窗口内做轻量 RGB-T 互补。
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim * 3, dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim * 2)

        # 让 light 分支初始接近 identity，减少训练初期破坏
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, rgb_win, ir_win):
        h = torch.cat([rgb_win, ir_win, torch.abs(rgb_win - ir_win)], dim=-1)
        h = self.fc2(self.act(self.fc1(h)))
        dr, di = torch.chunk(h, 2, dim=-1)

        rgb_light = rgb_win + dr
        ir_light = ir_win + di
        return rgb_light, ir_light


# ============================================================
# Main module
# ============================================================

class CMSSMahalanobisWindowInteraction(nn.Module):
    """
    CMSS + 低维对角马氏距离 引导的差异化窗口交互模块

    输入:
        x = [rgb, ir]
        rgb, ir: [B, C, H, W]

    输出:
        (rgb_out, ir_out)

    三条窗口路径:
        strong: 非对称跨模态窗口注意力，小窗口 query，大窗口 search
        light : 同位置轻量 RGB-T 交互
        keep  : 单模态局部保留/增强

    路由依据:
        1) CMSS: 估计窗口跨模态信息密度
        2) diagonal Mahalanobis distance: 估计当前窗口跨模态差异是否偏离训练集常见模式
        3) structural energy: 防止低结构噪声被误判为 strong
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        stat_dim=32,
        small_win=4,
        large_win=6,
        num_heads=4,
        route_temp=1.0,
        ema_momentum=0.99,
        eps=1e-6,
        outer_residual=False,
        debug=False,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0
        assert large_win >= small_win
        assert (large_win - small_win) % 2 == 0, \
            "建议使用 4/6 或 2/4；large_win-small_win 需要为偶数，方便中心对齐。"

        self.c = c
        self.embed_dim = embed_dim
        self.stat_dim = stat_dim
        self.small_win = small_win
        self.large_win = large_win
        self.num_heads = num_heads
        self.route_temp = route_temp
        self.ema_momentum = ema_momentum
        self.eps = eps
        self.outer_residual = outer_residual
        self.debug = debug
        self.last_debug = None

        # -----------------------------
        # 1. 输入投影到统一交互空间
        # -----------------------------
        self.rgb_in = _ConvBNAct(c, embed_dim, k=1)
        self.ir_in = _ConvBNAct(c, embed_dim, k=1)

        # -----------------------------
        # 2. 低维共享统计投影
        #    用于 CMSS / Mahalanobis，不用于主 attention
        # -----------------------------
        self.stat_proj = nn.Conv2d(embed_dim, stat_dim, kernel_size=1, bias=False)
        self.stat_norm = nn.LayerNorm(stat_dim)

        # EMA 统计量：训练集级别的 RGB-T 差异均值和方差
        self.register_buffer("mah_mean", torch.zeros(stat_dim))
        self.register_buffer("mah_var", torch.ones(stat_dim))
        self.register_buffer("mah_initialized", torch.tensor(0.0))

        # -----------------------------
        # 3. 三条路径
        # -----------------------------
        self.rgb_from_ir = _CrossWindowAttention(embed_dim, num_heads)
        self.ir_from_rgb = _CrossWindowAttention(embed_dim, num_heads)

        self.light_fuse = _LightSameWindowFusion(embed_dim)

        self.local_rgb = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.local_ir = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1, groups=embed_dim, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        self.rgb_ln = nn.LayerNorm(embed_dim)
        self.ir_ln = nn.LayerNorm(embed_dim)

        # -----------------------------
        # 4. 输出回原通道
        # -----------------------------
        self.rgb_out = _ConvBNAct(embed_dim, c, k=1)
        self.ir_out = _ConvBNAct(embed_dim, c, k=1)

        # 为了方便插入现有 YOLO，默认开外层残差。
        # 如果你想验证“非残差输出”，初始化时 outer_residual=False。
        self.gamma_rgb = nn.Parameter(torch.zeros(1))
        self.gamma_ir = nn.Parameter(torch.zeros(1))

    @torch.no_grad()
    def _update_mahalanobis_stats(self, delta):
        """
        delta: [B*Nw, stat_dim]
        """
        delta = delta.detach()
        batch_mean = delta.mean(dim=0)
        batch_var = delta.var(dim=0, unbiased=False).clamp_min(self.eps)

        if self.mah_initialized.item() < 0.5:
            self.mah_mean.copy_(batch_mean)
            self.mah_var.copy_(batch_var)
            self.mah_initialized.fill_(1.0)
        else:
            m = self.ema_momentum
            self.mah_mean.mul_(m).add_(batch_mean, alpha=1.0 - m)
            self.mah_var.mul_(m).add_(batch_var, alpha=1.0 - m)

    def _compute_route(self, rgb_feat, ir_feat, H, W):
        """
        rgb_feat, ir_feat: [B, embed_dim, H, W], padded
        return:
            route: [B, 3, Nw]
            debug_maps: dict
        """
        B = rgb_feat.shape[0]
        M = self.small_win
        Nw = (H // M) * (W // M)

        rgb_stat = self.stat_proj(rgb_feat)
        ir_stat = self.stat_proj(ir_feat)

        rgb_win = _cmss_window_partition(rgb_stat, M)  # [B*Nw, M*M, d]
        ir_win = _cmss_window_partition(ir_stat, M)

        rgb_win = self.stat_norm(rgb_win)
        ir_win = self.stat_norm(ir_win)

        # window vector: [B*Nw, d]
        z_rgb = rgb_win.mean(dim=1)
        z_ir = ir_win.mean(dim=1)

        # --------------------------------------------------
        # 1) CMSS
        # CMSS = (1 + cosine) / (2 * var_rgb * var_ir + eps)
        # low CMSS -> high information density
        # --------------------------------------------------
        cos = F.cosine_similarity(z_rgb, z_ir, dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)

        var_rgb = z_rgb.var(dim=-1, unbiased=False).clamp_min(self.eps)
        var_ir = z_ir.var(dim=-1, unbiased=False).clamp_min(self.eps)

        cmss_raw = (1.0 + cos) / (2.0 * var_rgb * var_ir + self.eps)
        cmss = cmss_raw.view(B, Nw)
        cmss = _norm01_per_sample(cmss, self.eps)

        info = 1.0 - cmss  # [B, Nw]

        # --------------------------------------------------
        # 2) Low-dimensional diagonal Mahalanobis distance
        # --------------------------------------------------
        delta = z_rgb - z_ir  # [B*Nw, d]

        if self.training:
            self._update_mahalanobis_stats(delta)

        dmah_raw = ((delta - self.mah_mean) ** 2 / (self.mah_var + self.eps)).mean(dim=-1)
        dmah = dmah_raw.view(B, Nw)
        dmah = _norm01_per_sample(dmah, self.eps)

        # --------------------------------------------------
        # 3) Structural energy
        # window 内 token 方差，判断差异是否有结构支撑
        # --------------------------------------------------
        e_rgb = rgb_win.var(dim=1, unbiased=False).mean(dim=-1)
        e_ir = ir_win.var(dim=1, unbiased=False).mean(dim=-1)
        energy = 0.5 * (e_rgb + e_ir)
        energy = energy.view(B, Nw)
        energy = _norm01_per_sample(energy, self.eps)

        # --------------------------------------------------
        # 4) Three-way routing prior
        #
        # strong: 高信息 + 差异异常 + 有结构
        # light : 差异不异常 + 有结构
        # keep  : 低结构，或低结构异常
        # --------------------------------------------------
        strong_logit = info + dmah + energy
        light_logit = (1.0 - dmah) + energy + 0.5 * info
        keep_logit = (1.0 - energy) + dmah * (1.0 - energy) + cmss * (1.0 - energy)

        logits = torch.stack([strong_logit, light_logit, keep_logit], dim=1)  # [B, 3, Nw]
        route = torch.softmax(logits / max(self.route_temp, self.eps), dim=1)

        debug_maps = None
        if self.debug:
            debug_maps = {
                "cmss": cmss.detach(),        # [B, Nw], high -> low info density
                "info": info.detach(),        # [B, Nw], high -> high info density
                "dmah": dmah.detach(),        # [B, Nw], high -> abnormal RGB-T difference
                "energy": energy.detach(),    # [B, Nw], high -> structured window
                "route": route.detach(),      # [B, 3, Nw]
            }

        return route, debug_maps

    def forward(self, x):
        if not (isinstance(x, (list, tuple)) and len(x) == 2):
            raise ValueError(
                f"CMSSMahalanobisWindowInteraction expects [rgb, ir], got {type(x)} with len={len(x) if isinstance(x, (list, tuple)) else 'NA'}"
            )

        rgb, ir = x[0], x[1]
        rgb_res, ir_res = rgb, ir

        # 1. project
        rgb = self.rgb_in(rgb)
        ir = self.ir_in(ir)

        # 2. pad
        rgb, ori_hw = _cmss_autopad_to_multiple(rgb, self.small_win)
        ir, _ = _cmss_autopad_to_multiple(ir, self.small_win)

        B, C, H, W = rgb.shape
        M = self.small_win

        # 3. window routing
        route, debug_maps = self._compute_route(rgb, ir, H, W)
        Nw = (H // M) * (W // M)

        r_strong = route[:, 0].reshape(B * Nw, 1, 1)
        r_light = route[:, 1].reshape(B * Nw, 1, 1)
        r_keep = route[:, 2].reshape(B * Nw, 1, 1)

        # 4. small and large windows
        rgb_small = _cmss_window_partition(rgb, self.small_win)
        ir_small = _cmss_window_partition(ir, self.small_win)

        rgb_large = _cmss_extract_large_windows(rgb, self.small_win, self.large_win)
        ir_large = _cmss_extract_large_windows(ir, self.small_win, self.large_win)

        rgb_small_n = self.rgb_ln(rgb_small)
        ir_small_n = self.ir_ln(ir_small)
        rgb_large_n = self.rgb_ln(rgb_large)
        ir_large_n = self.ir_ln(ir_large)

        # ==================================================
        # strong branch: bidirectional asymmetric attention
        # ==================================================
        rgb_cross = rgb_small + self.rgb_from_ir(rgb_small_n, ir_large_n)
        ir_cross = ir_small + self.ir_from_rgb(ir_small_n, rgb_large_n)

        # ==================================================
        # light branch: same-window light fusion
        # ==================================================
        rgb_light, ir_light = self.light_fuse(rgb_small, ir_small)

        # ==================================================
        # keep branch: local single-modal enhancement
        # ==================================================
        rgb_keep_map = self.local_rgb(rgb)
        ir_keep_map = self.local_ir(ir)

        rgb_keep = _cmss_window_partition(rgb_keep_map, self.small_win)
        ir_keep = _cmss_window_partition(ir_keep_map, self.small_win)

        # ==================================================
        # route mixture
        # 注意：route 作用在三条路径输出上，不作用在 attention 矩阵上
        # ==================================================
        rgb_mix = r_strong * rgb_cross + r_light * rgb_light + r_keep * rgb_keep
        ir_mix = r_strong * ir_cross + r_light * ir_light + r_keep * ir_keep

        # 5. reverse windows
        rgb_mix = _cmss_window_reverse(rgb_mix, self.small_win, H, W)
        ir_mix = _cmss_window_reverse(ir_mix, self.small_win, H, W)

        # 6. remove padding
        rgb_mix = _cmss_remove_pad(rgb_mix, ori_hw)
        ir_mix = _cmss_remove_pad(ir_mix, ori_hw)

        # 7. output projection
        rgb_delta = self.rgb_out(rgb_mix)
        ir_delta = self.ir_out(ir_mix)

        if self.outer_residual:
            rgb_out = rgb_res + self.gamma_rgb * rgb_delta
            ir_out = ir_res + self.gamma_ir * ir_delta
        else:
            rgb_out = rgb_delta
            ir_out = ir_delta

        if self.debug:
            h_win = H // self.small_win
            w_win = W // self.small_win
            if debug_maps is not None:
                self.last_debug = {
                    k: (v.view(B, -1, h_win, w_win) if k == "route" else v.view(B, 1, h_win, w_win))
                    for k, v in debug_maps.items()
                }

        # return (rgb_out, ir_out)
        return rgb_out + ir_out

# ============================================================
# LA-SCI: Low-light Aware Soft Cross-modal Interaction
# 低光感知软选择跨模态交互模块
#
# 设计目标：
#   1) 用 RGB 原图亮度计算局部低光退化项，而不是用特征弱响应冒充低光；
#   2) 用“特征差异 + 余弦不一致 + 局部低光”评估窗口交互需求；
#   3) 不使用 hard Top-K / 硬阈值，而使用 soft budget gate 生成连续交互强度；
#   4) 当前模态 small window 作为 Query，另一模态 large window 作为 Key/Value，
#      在局部邻域内做软对齐式跨模态补偿；
#   5) 低光方向调节是局部窗口级的：同一张图中亮区和暗区会得到不同方向权重；
#   6) 输出双路特征 (rgb_out, ir_out)，不在模块内部做最终融合；
#   7) 不包含 Mamba / SS2D，便于直接插入 block.py 后面进行验证。
#
# 推荐 YAML 形式示例：
#   - [[rgb_layer, ir_layer, raw_rgb_layer], 1, LASCIModule, [256, 128, 4, 6, 4, 0.35]]
#
# 如果暂时无法传 raw_rgb，也可以输入 [rgb_feat, ir_feat]，此时低光项自动置零，
# 模块退化为“特征差异 + 余弦不一致”的软选择跨模态交互。
# ============================================================

# 让 from block import * 时可以导出该模块。
# 如果 block.py 顶部已有 __all__，追加即可；如果没有，则新建一个。



def _lasci_pad_to_multiple(x, multiple):
    """
    将特征图 H/W pad 到 multiple 的整数倍，方便非重叠窗口划分。

    Args:
        x: [B, C, H, W]
        multiple: 窗口大小，例如 small_win=4

    Returns:
        x_pad: pad 后的特征
        pad_hw: (pad_h, pad_w)，用于后续去 pad
    """
    b, c, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        # replicate 比 zero pad 更稳，避免边界生成异常暗/异常强区域
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


def _lasci_remove_pad(x, pad_hw):
    """去掉 _lasci_pad_to_multiple 增加的 padding。"""
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def _lasci_window_partition(x, win):
    """
    将 [B,C,H,W] 划分为非重叠小窗口。

    Args:
        x: [B, C, H, W]
        win: small window size

    Returns:
        windows: [B, L, win*win, C]
        gh, gw : 窗口网格数量，L = gh * gw
    """
    b, c, h, w = x.shape
    assert h % win == 0 and w % win == 0, f"H={h}, W={w} must be divisible by win={win}"
    gh, gw = h // win, w // win
    x = x.view(b, c, gh, win, gw, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = x.view(b, gh * gw, win * win, c)
    return windows, gh, gw


def _lasci_window_reverse(windows, gh, gw, win):
    """
    将窗口序列恢复为二维特征图。

    Args:
        windows: [B, L, win*win, C]
        gh, gw : 窗口网格数量
        win    : small window size

    Returns:
        x: [B, C, gh*win, gw*win]
    """
    b, l, n, c = windows.shape
    assert l == gh * gw, f"L={l} does not match gh*gw={gh*gw}"
    assert n == win * win, f"N={n} does not match win*win={win*win}"
    x = windows.view(b, gh, gw, win, win, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, gh * win, gw * win)


def _lasci_extract_large_windows(x, small_win, large_win):
    """
    以 small_win 为步长，为每个 small window 提取另一模态 large window 搜索邻域。

    这里 large_win 可以是偶数，例如 small_win=4, large_win=6。
    它不是传统“以单像素为中心”的奇数卷积核，而是围绕当前 small window
    扩展出的更大局部区域，用于容忍 RGB-IR 轻中度错位。

    Args:
        x: [B, C, H, W]
        small_win: 当前模态 query 窗口大小
        large_win: 另一模态 search 窗口大小，应 >= small_win

    Returns:
        patches: [B, L, large_win*large_win, C]
    """
    b, c, h, w = x.shape
    assert large_win >= small_win, f"large_win={large_win} should be >= small_win={small_win}"

    # 为了保证 unfold 后窗口数量仍为 H/small_win * W/small_win，
    # 需要总 padding = large_win - small_win。
    pad_total = large_win - small_win
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top

    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    patches = F.unfold(
        x_pad,
        kernel_size=large_win,
        stride=small_win,
        padding=0,
    )  # [B, C*large_win*large_win, L]

    patches = patches.transpose(1, 2).contiguous()  # [B, L, C*K]
    patches = patches.view(b, patches.shape[1], c, large_win * large_win)
    patches = patches.permute(0, 1, 3, 2).contiguous()  # [B, L, K, C]
    return patches


def _lasci_norm_01(x, eps=1e-6):
    """
    对每个 batch 样本单独做 0~1 归一化。
    支持：
        x: [B, L]
        x: [B, 1, H, W]
    """
    if x.ndim == 2:
        x_min = x.amin(dim=1, keepdim=True)
        x_max = x.amax(dim=1, keepdim=True)
    elif x.ndim == 4:
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
    else:
        raise ValueError(f"Unsupported tensor ndim={x.ndim}")
    return (x - x_min) / (x_max - x_min + eps)


class _LASCILocalCrossAttention(nn.Module):
    """
    局部跨模态注意力：
        当前模态 small window 作为 Query；
        另一模态 large window 作为 Key/Value。

    作用：
        不强制同位置融合，而是在另一模态局部大窗口内寻找相关内容，
        从而在有限邻域内缓解 RGB-IR 弱错位造成的特征不对应。
    """

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, q_win, kv_win):
        """
        Args:
            q_win : [N, Nq, C]，当前模态 selected/all small windows
            kv_win: [N, Nk, C]，另一模态对应 large windows

        Returns:
            out: [N, Nq, C]
        """
        n, nq, c = q_win.shape
        nk = kv_win.shape[1]

        q = self.q_proj(q_win)
        k = self.k_proj(kv_win)
        v = self.v_proj(kv_win)

        # [N, heads, tokens, head_dim]
        q = q.view(n, nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(n, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(n, nk, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.permute(0, 2, 1, 3).contiguous().view(n, nq, c)
        return self.o_proj(out)


class LASCIModule(nn.Module):
    """
    LA-SCI: Low-light Aware Soft Cross-modal Interaction
    低光感知软选择跨模态交互模块。

    输入：
        1) [rgb_feat, ir_feat, rgb_raw]
           rgb_feat: [B,C,H,W]
           ir_feat : [B,C,H,W]
           rgb_raw : [B,3,H0,W0]，RGB 原图，建议范围为 [0,1]
                     如果是 [0,255]，将 rgb_raw_range=255.0。

        2) [rgb_feat, ir_feat]
           不提供 RGB 原图时，低光项自动置 0，模块退化为仅根据特征关系评估交互。

    输出：
        (rgb_out, ir_out): 双路增强后的特征。

    核心公式：
        D_w: 窗口级特征差异
        U_w: 窗口级余弦不一致
        L_w: 从 RGB 原图亮度得到的局部低光退化项

        s_w = sigmoid(MLP([D_w, U_w, L_w]))
        g_w = 1 - exp(- budget * softmax(s_w / tau))

        g_{i->r} = g_w * (1 + lambda_low * L_w)
        g_{r->i} = g_w * (1 - lambda_low * L_w)

        rgb_out = rgb + gamma_r * g_{i->r} * Delta_{i->r}
        ir_out  = ir  + gamma_i * g_{r->i} * Delta_{r->i}
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=6,
        num_heads=4,
        budget_ratio=0.35,
        softmax_tau=0.25,
        lambda_low=0.5,
        low_thr=100.0 / 255.0,
        low_smooth=0.08,
        rgb_raw_range=1.0,
        chunk_windows=2048,
        init_gamma=0.05,
        return_pair=False,
        debug = False,
        last_debug = None,
    ):
        """
        Args:
            c: 输入 RGB/IR 特征通道数。
            embed_dim: 局部相关性计算空间维度。
            small_win: 当前模态 Query 小窗口大小。
            large_win: 另一模态 Key/Value 搜索大窗口大小。
            num_heads: 局部跨模态注意力头数。
            budget_ratio: 软交互预算比例；不是 hard top-k 比例。
                          例如 0.35 表示全图窗口共享约 35% 的交互预算。
            softmax_tau: soft budget 的温度；越小越集中，越大越平均。
            lambda_low: 低光方向调节强度；低光越强，越增强 IR->RGB，越抑制 RGB->IR。
            low_thr: RGB 亮度低光阈值，默认约 100/255。
            low_smooth: 低光 soft threshold 的平滑系数。
            rgb_raw_range: rgb_raw 的取值范围。YOLO 中通常为 1.0；如果输入 0~255，设为 255.0。
            chunk_windows: 所有窗口做 attention 时的分块大小，用于控制显存。
            init_gamma: 残差回写强度初值。设小一点可避免训练初期破坏 baseline。
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert large_win >= small_win, "large_win should be >= small_win"

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.budget_ratio = float(budget_ratio)
        self.softmax_tau = float(softmax_tau)
        self.lambda_low = float(lambda_low)
        self.low_thr = float(low_thr)
        self.low_smooth = float(low_smooth)
        self.rgb_raw_range = float(rgb_raw_range)
        self.chunk_windows = int(chunk_windows)
        self.return_pair = return_pair
        self.debug = False
        self.last_debug = None

        # 将 RGB/IR 特征投影到跨模态相关性计算空间。
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        # token 级归一化，稳定窗口内 attention 与分数计算。
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        # # 三项评分 MLP：输入 [D_w, U_w, L_w]，输出窗口交互需求 s_w。
        # # 不人为固定三项之间的乘法或加法关系，而是交给 MLP 在检测监督下学习。
        # self.score_mlp = nn.Sequential(
        #     nn.Linear(3, 16, bias=True),
        #     nn.SiLU(inplace=True),
        #     nn.Linear(16, 1, bias=True),
        # )

        # 轻量窗口方向 Router：
        # 输入当前窗口 RGB/IR 的局部特征摘要，输出两个方向的残差门控：
        # gate_i2r: IR -> RGB
        # gate_r2i: RGB -> IR
        router_hidden = max(embed_dim // 4, 16)
        self.router_mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, router_hidden, bias=True),
            nn.SiLU(inplace=True),
            nn.Linear(router_hidden, 2, bias=True),
        )
        # 保守初始化：初始 gate = sigmoid(0) = 0.5
        # 由于后面还有 gamma_r/gamma_i，实际残差强度仍然较小。
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.zeros_(self.router_mlp[-1].bias)

        # 双向局部跨模态交互。
        self.rgb_from_ir = _LASCILocalCrossAttention(embed_dim, num_heads)
        self.ir_from_rgb = _LASCILocalCrossAttention(embed_dim, num_heads)

        # 将 embed_dim 空间的更新量投影回原通道数 c。
        self.rgb_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # 残差强度。给一个较小初值，既允许分支获得梯度，又不至于一开始破坏原特征。
        self.gamma_r = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gamma_i = nn.Parameter(torch.tensor(float(init_gamma)))

        # 输出后轻量稳定层。不是融合，只是对双路各自做局部整理。
        self.rgb_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.act = nn.SiLU(inplace=True)

    # --------------------------------------------------------
    # 亮度与低光项
    # --------------------------------------------------------
    def _build_lowlight_map(self, rgb_raw, target_hw, ref_tensor=None):
        """
        从 RGB 原图构造低光退化图，而不是从特征响应估计低光。

        Args:
            rgb_raw: [B,3,H0,W0]，RGB 原图，建议范围 [0,1]
            target_hw: 当前特征层大小 (H,W)
            ref_tensor: 用于对齐 device / dtype 的参考张量，一般传 rgb_e

        Returns:
            low_map: [B,1,H,W]，局部越暗越接近 1
        """
        if rgb_raw is None:
            return None

        # 关键：不要 rgb_raw.float()
        # 验证时模型可能是 half，必须让 low_map 和特征 dtype 一致
        if ref_tensor is not None:
            rgb = rgb_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            rgb = rgb_raw

        # 如果输入是 0~255，用户应设置 rgb_raw_range=255.0
        if self.rgb_raw_range != 1.0:
            rgb = rgb / self.rgb_raw_range

        rgb = rgb.clamp(0.0, 1.0)

        # RGB -> luma，使用同 dtype 计算
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        # resize 到当前特征层大小
        y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)

        # soft low-light
        low_thr = y.new_tensor(self.low_thr)
        low_smooth = max(self.low_smooth, 1e-6)
        low_map = torch.sigmoid((low_thr - y) / low_smooth)

        return low_map

    @staticmethod
    def _window_mean_map(x, small_win):
        """
        将 [B,1,H,W] 的 map 池化成窗口级 [B,L]。
        H/W 必须能被 small_win 整除。
        """
        return F.avg_pool2d(x, kernel_size=small_win, stride=small_win).flatten(1)

    # --------------------------------------------------------
    # 评分与软预算门控
    # --------------------------------------------------------
    # def _compute_window_score(self, rgb_small_n, ir_small_n, low_win):
    #     """
    #     计算窗口级交互需求分数。

    #     Args:
    #         rgb_small_n: [B,L,N,E]，归一化后的 RGB small windows
    #         ir_small_n : [B,L,N,E]，归一化后的 IR small windows
    #         low_win    : [B,L]，从 RGB 原图亮度得到的局部低光项

    #     Returns:
    #         score: [B,L]，窗口交互需求，范围 [0,1]
    #         diff_score: [B,L]，便于后续可视化/调试
    #         incons_score: [B,L]
    #     """
    #     # 1) 特征差异项 D_w：窗口内 L1 差异。
    #     diff_score = torch.abs(rgb_small_n - ir_small_n).mean(dim=(2, 3))  # [B,L]
    #     diff_score = _lasci_norm_01(diff_score)

    #     # 2) 余弦不一致项 U_w：窗口均值向量的方向差异。
    #     rgb_vec = rgb_small_n.mean(dim=2)  # [B,L,E]
    #     ir_vec = ir_small_n.mean(dim=2)    # [B,L,E]
    #     cos = F.cosine_similarity(rgb_vec, ir_vec, dim=-1, eps=1e-6)  # [-1,1]
    #     sim01 = 0.5 * (cos + 1.0)
    #     incons_score = 1.0 - sim01.clamp(0.0, 1.0)  # [B,L]

    #     # ------------------------------------------------------------
    #     # 3) 三项分数拼接
    #     # diff_score   : [B, L]
    #     # incons_score : [B, L]
    #     # low_win      : [B, L]
    #     # ------------------------------------------------------------

    #     # 先保证 low_win 和特征统计量 dtype/device 一致
    #     low_win = low_win.to(device=diff_score.device, dtype=diff_score.dtype)

    #     score_in = torch.stack(
    #         [diff_score, incons_score, low_win],
    #         dim=-1
    #     )  # [B, L, 3]

    #     # ------------------------------------------------------------
    #     # 关键修复：
    #     # Linear 要求输入 mat1 和权重 mat2 dtype 一致。
    #     # 验证阶段可能 model.half() 或输入 half，必须显式对齐到 MLP 权重 dtype。
    #     # ------------------------------------------------------------
    #     mlp_dtype = self.score_mlp[0].weight.dtype
    #     mlp_device = self.score_mlp[0].weight.device

    #     score_in = score_in.to(device=mlp_device, dtype=mlp_dtype)

    #     score = torch.sigmoid(self.score_mlp(score_in)).squeeze(-1)  # [B, L]

    #     # score 后面要和特征/gate 相乘，所以再转回特征 dtype
    #     score = score.to(device=diff_score.device, dtype=diff_score.dtype)

    #     return score, diff_score, incons_score

    # def _soft_budget_gate(self, score, return_prob=False):
    #     """
    #     Soft Budget Gate：用连续门控替代 hard Top-K。

    #     Args:
    #         score: [B,L]，窗口交互需求分数

    #     Returns:
    #         gate: [B,L]，连续交互强度，范围 (0,1)

    #     解释：
    #         budget_ratio 不再表示硬选多少窗口，而表示全图窗口共享多少交互预算。
    #         高分窗口获得更大 gate，低分窗口获得较小 gate，但不会被硬置零。
    #     """
    #     b, l = score.shape
    #     # if self.budget_ratio <= 0:
    #     #     return torch.zeros_like(score)
    #     # if self.budget_ratio <= 1.0:
    #     #     budget = self.budget_ratio * float(l)
    #     # else:
    #     #     budget = min(float(self.budget_ratio), float(l))
    #     if self.budget_ratio <= 0:
    #         gate = torch.zeros_like(score)
    #         prob = torch.zeros_like(score)
    #         return (gate, prob) if return_prob else gate
    #     if self.budget_ratio <= 1.0:
    #         budget = self.budget_ratio * float(l)
    #     else:
    #         budget = min(float(self.budget_ratio), float(l))

    #     tau = max(self.softmax_tau, 1e-6)
    #     prob = torch.softmax(score / tau, dim=1)  # [B,L]
    #     gate = 1.0 - torch.exp(-budget * prob)    # [B,L]
    #     # return gate
    #     return (gate, prob) if return_prob else gate
    def _compute_router_gate(self, rgb_small_n, ir_small_n):
        """
        根据当前 small window 的 RGB/IR 特征生成两个方向的窗口级 gate。

        Args:
            rgb_small_n: [B,L,N,E]
            ir_small_n : [B,L,N,E]

        Returns:
            gate_i2r: [B,L]，IR -> RGB 方向残差门控
            gate_r2i: [B,L]，RGB -> IR 方向残差门控
            router_logits: [B,L,2]，调试用
        """
        # 1) GAP over window tokens
        rgb_desc = rgb_small_n.mean(dim=2)  # [B,L,E]
        ir_desc = ir_small_n.mean(dim=2)    # [B,L,E]

        # 2) concat 当前窗口两种模态的局部描述
        router_in = torch.cat([rgb_desc, ir_desc], dim=-1)  # [B,L,2E]

        # 3) dtype/device 对齐，避免 half 验证时报错
        mlp_dtype = self.router_mlp[0].weight.dtype
        mlp_device = self.router_mlp[0].weight.device

        router_in = router_in.to(device=mlp_device, dtype=mlp_dtype)
        router_logits = self.router_mlp(router_in)  # [B,L,2]

        # 4) sigmoid：两个方向独立开关，不做 softmax 竞争
        gates = torch.sigmoid(router_logits)

        gates = gates.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)
        router_logits = router_logits.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)

        gate_i2r = gates[..., 0]  # [B,L]
        gate_r2i = gates[..., 1]  # [B,L]

        return gate_i2r, gate_r2i, router_logits

    def _cross_attention_all_windows(self, q_win, kv_win, attn_module):
        """
        所有窗口都做局部跨模态 attention，但用 chunk 控制显存。

        Args:
            q_win: [B,L,Nq,E]
            kv_win: [B,L,Nk,E]
            attn_module: _LASCILocalCrossAttention

        Returns:
            out: [B,L,Nq,E]
        """
        b, l, nq, e = q_win.shape
        nk = kv_win.shape[2]

        q_flat = q_win.reshape(b * l, nq, e)
        kv_flat = kv_win.reshape(b * l, nk, e)

        outs = []
        total = q_flat.shape[0]
        chunk = max(1, self.chunk_windows)
        for st in range(0, total, chunk):
            ed = min(st + chunk, total)
            outs.append(attn_module(q_flat[st:ed], kv_flat[st:ed]))

        out = torch.cat(outs, dim=0)
        return out.view(b, l, nq, e)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat, rgb_raw] 或 [rgb_feat, ir_feat]

        Returns:
            (rgb_out, ir_out)
        """
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"LASCIModule expects list/tuple input, got {type(x)}")

        if len(x) == 3:
            rgb, ir, rgb_raw = x[0], x[1], x[2]
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)
            rgb_raw = rgb_raw.to(device=rgb.device, dtype=rgb.dtype)
        elif len(x) == 2:
            rgb, ir = x[0], x[1]
            rgb_raw = None
        else:
            raise ValueError(f"LASCIModule expects [rgb, ir] or [rgb, ir, rgb_raw], got len={len(x)}")

        assert rgb.shape == ir.shape, f"rgb and ir feature shapes must match, got {rgb.shape} vs {ir.shape}"

        rgb_res, ir_res = rgb, ir
        b, c, h0, w0 = rgb.shape

        # 1) 投影到相关性空间。
        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)

        # 2) pad 到 small_win 的整数倍。
        rgb_e, pad_hw = _lasci_pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = _lasci_pad_to_multiple(ir_e, self.small_win)

        _, _, h, w = rgb_e.shape

        # 3) 从 RGB 原图构造局部低光图，注意它是局部的，不是整图单一标量。
        # low_map = self._build_lowlight_map(rgb_raw, target_hw=(h, w))
        # low_map = self._build_lowlight_map(rgb_raw, target_hw=(h, w), ref_tensor=rgb_e)
        # if low_map is not None:
        #     low_map, _ = _lasci_pad_to_multiple(low_map, self.small_win)
        #     low_win = self._window_mean_map(low_map, self.small_win)  # [B,L]
        #     low_win = low_win.to(device=rgb_e.device, dtype=rgb_e.dtype)
        # else:
        #     low_win = None

        # 4) 构建 small query windows 和 large search windows。
        rgb_small, gh, gw = _lasci_window_partition(rgb_e, self.small_win)  # [B,L,Ns,E]
        ir_small, _, _ = _lasci_window_partition(ir_e, self.small_win)

        rgb_large = _lasci_extract_large_windows(rgb_e, self.small_win, self.large_win)  # [B,L,Nl,E]
        ir_large = _lasci_extract_large_windows(ir_e, self.small_win, self.large_win)

        # 5) LayerNorm 后计算分数和 attention。
        rgb_small_n = self.rgb_norm(rgb_small)
        ir_small_n = self.ir_norm(ir_small)
        rgb_large_n = self.rgb_norm(rgb_large)
        ir_large_n = self.ir_norm(ir_large)

        # # 6) 三项评分：特征差异、余弦不一致、RGB 原图局部低光。
        # # score, _, _ = self._compute_window_score(rgb_small_n, ir_small_n, low_win)
        # score, diff_score, incons_score = self._compute_window_score(rgb_small_n, ir_small_n, low_win)

        # # 7) 软预算门控：所有窗口都有连续交互强度，避免 hard top-k。
        # # gate = self._soft_budget_gate(score)  # [B,L]
        # # gate = gate.to(device=rgb_e.device, dtype=rgb_e.dtype)
        # gate, prob = self._soft_budget_gate(score, return_prob=True)
        # gate = gate.to(device=rgb_e.device, dtype=rgb_e.dtype)
        # prob = prob.to(device=rgb_e.device, dtype=rgb_e.dtype)
        # # 如果没有 low_win，则方向调节为 0。
        # if low_win is None:
        #     low_win = torch.zeros_like(gate)
        # else:
        #     low_win = low_win.to(device=gate.device, dtype=gate.dtype)

        # # 6) 轻量窗口方向 router：
        # #    不再使用 D/U/L，也不再做全图 soft budget 分配。
        # gate_i2r, gate_r2i, router_logits = self._compute_router_gate(
        #     rgb_small_n,
        #     ir_small_n
        # )
        # gate_i2r = gate_i2r.to(device=rgb_e.device, dtype=rgb_e.dtype)
        # gate_r2i = gate_r2i.to(device=rgb_e.device, dtype=rgb_e.dtype)

        # 8) 双向局部跨模态交互：
        #    RGB small -> IR large: 用 IR 补 RGB
        #    IR  small -> RGB large: 用 RGB 补 IR

        delta_i2r = self._cross_attention_all_windows(rgb_small_n, ir_large_n, self.rgb_from_ir)

        delta_r2i = self._cross_attention_all_windows(ir_small_n, rgb_large_n, self.ir_from_rgb)

        # 9) 低光方向调节是局部窗口级的：
        #    一张图可以局部亮、局部暗；每个窗口都有自己的 low_win。
        #    low_win 越大，越增强 IR->RGB，越抑制 RGB->IR。
        # gate_i2r = gate * (1.0 + self.lambda_low * low_win)
        # gate_r2i = gate * (1.0 - self.lambda_low * low_win).clamp(min=0.0)

        # if getattr(self, "debug", False):
        #     self.last_debug = {
        #         # "D": diff_score.detach().float().cpu(),
        #         # "U": incons_score.detach().float().cpu(),
        #         # "L": low_win.detach().float().cpu(),
        #         # "score": score.detach().float().cpu(),
        #         # "prob": prob.detach().float().cpu(),
        #         # "gate": gate.detach().float().cpu(),
        #         # "gate_i2r": gate_i2r.detach().float().cpu(),
        #         # "gate_r2i": gate_r2i.detach().float().cpu(),
        #         "gate_i2r": gate_i2r.detach().float().cpu(),
        #         "gate_r2i": gate_r2i.detach().float().cpu(),
        #         "logit_i2r": router_logits[..., 0].detach().float().cpu(),
        #         "logit_r2i": router_logits[..., 1].detach().float().cpu(),
        #         "gh": gh,
        #         "gw": gw,
        #         "h": h,
        #         "w": w,
        #     }

        # gate_i2r = gate_i2r.unsqueeze(-1).unsqueeze(-1).to(
        #     device=delta_i2r.device, dtype=delta_i2r.dtype
        # )
        # gate_r2i = gate_r2i.unsqueeze(-1).unsqueeze(-1).to(
        #     device=delta_r2i.device, dtype=delta_r2i.dtype
        # )
        # delta_i2r = gate_i2r * delta_i2r
        # delta_r2i = gate_r2i * delta_r2i

        # 10) 窗口还原回特征图。

        delta_i2r = _lasci_window_reverse(delta_i2r, gh, gw, self.small_win)

        delta_r2i = _lasci_window_reverse(delta_r2i, gh, gw, self.small_win)


        # 11) 去 pad。
        delta_i2r = _lasci_remove_pad(delta_i2r, pad_hw)
        delta_r2i = _lasci_remove_pad(delta_r2i, pad_hw)

        # 12) 投影回原通道，并以残差形式回写。
        # 保险：Conv/BN 权重可能是 half，输入必须对齐
        delta_i2r = delta_i2r.to(
            device=self.rgb_out[0].weight.device,
            dtype=self.rgb_out[0].weight.dtype
        )
        delta_r2i = delta_r2i.to(
            device=self.ir_out[0].weight.device,
            dtype=self.ir_out[0].weight.dtype
        )
        delta_i2r = self.rgb_out(delta_i2r)
        delta_r2i = self.ir_out(delta_r2i)

        delta_i2r = delta_i2r.to(device=rgb_res.device, dtype=rgb_res.dtype)
        delta_r2i = delta_r2i.to(device=ir_res.device, dtype=ir_res.dtype)

        rgb_out = rgb_res + self.gamma_r.to(dtype=rgb_res.dtype) * delta_i2r
        ir_out = ir_res + self.gamma_i.to(dtype=ir_res.dtype) * delta_r2i

        # 13) 轻量 FFN 整理；仍然是双路输出，不做最终融合。
        rgb_out = self.act(rgb_out + self.rgb_ffn(rgb_out))
        # rgb_out = rgb_res
        ir_out = self.act(ir_out + self.ir_ffn(ir_out))

        # if self.return_pair:
        #     return rgb_out, ir_out
        if getattr(self, "return_pair", False):
            return rgb_out, ir_out
        return rgb_out + ir_out

class LASCIDSSF(nn.Module):
    """
    LA-SCI + DSSF_SS2D 包装模块。

    作用：
        1. 先用 LASCIModule 做光照感知的局部跨模态交互；
        2. 得到增强后的 rgb_out / ir_out 两路特征；
        3. 再用 DSSF_SS2D 做状态空间融合；
        4. 最终输出单路 fused feature，直接送入 YOLO Neck。

    这样 YAML 中不需要写：
        - [-1, 1, DSSF_SS2D, [c]]

    而是直接写：
        - [[rgb_idx, ir_idx, raw_rgb_idx], 1, LASCIDSSF, [...]]
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=6,
        num_heads=4,
        budget_ratio=0.35,
        softmax_tau=0.25,
        lambda_low=0.5,
    ):
        super().__init__()

        # LA-SCI 负责局部交互，必须返回双路增强特征
        self.lasci = LASCIModule(
            c=c,
            embed_dim=embed_dim,
            small_win=small_win,
            large_win=large_win,
            num_heads=num_heads,
            budget_ratio=budget_ratio,
            softmax_tau=softmax_tau,
            lambda_low=lambda_low,
            return_pair=True,
        )

        # DSSF_SS2D 负责状态空间融合
        # 保持你原来的简写逻辑，本质等价于 YAML 里的 DSSF_SS2D [c]
        self.dssf = DSSF_SS2D(c, c)

    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat, rgb_raw] 或 [rgb_feat, ir_feat]

        Returns:
            fused: [B, C, H, W]
        """
        rgb_out, ir_out = self.lasci(x)
        fused = self.dssf([rgb_out, ir_out])
        return fused

class CECC(nn.Module):
    """
    CECC: Cross-modal Evidence Consistency Calibration
    跨模态交互证据一致性校准模块

    设计目标：
    1) 不重新做 RGB/IR 融合；
    2) 不做 Mamba / SS2D / 全局状态传播；
    3) 只校准 LA-SCI 产生的跨模态补偿强度；
    4) 在小窗口网格上建模“交互证据是否被邻域支持”；
    5) 抑制孤立的错误交互，补全目标区域内断裂的交互响应。

    输入:
        rgb     : LA-SCI 输入 RGB 特征, [B,C,H,W]
        ir      : LA-SCI 输入 IR 特征,  [B,C,H,W]
        rgb_out : LA-SCI 输出 RGB 增强特征, [B,C,H,W]
        ir_out  : LA-SCI 输出 IR 增强特征,  [B,C,H,W]

    输出:
        rgb_cal : CECC 校准后的 RGB 特征
        ir_cal  : CECC 校准后的 IR 特征
    """

    def __init__(
        self,
        c,
        small_win=4,
        neigh_ks=3,
        hidden=16,
        eps=1e-6,
    ):
        super().__init__()
        assert neigh_ks % 2 == 1, "neigh_ks should be odd, e.g. 3 or 5"

        self.c = c
        self.small_win = small_win
        self.neigh_ks = neigh_ks
        self.eps = eps

        hidden = max(hidden, 8)

        # 目标响应保护图 O:
        # 用 fused feature 估计当前位置是否像目标区域，避免小目标孤立高交互被误杀。
        self.obj_head = nn.Sequential(
            nn.Conv2d(c, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 一致性权重 C:
        # 输入为 [G, G_bar, O] 三个低维证据图。
        self.consistency = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 校准强度，限制为 [0, max_alpha]，避免像 DSSF gamma 那样无界变负。
        self.alpha_raw = nn.Parameter(torch.tensor(-6.0))
        self.max_alpha = 0.2

    @staticmethod
    def _pad_to_multiple(x, multiple):
        b, c, h, w = x.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x, (pad_h, pad_w)

    @staticmethod
    def _remove_pad(x, pad_hw):
        pad_h, pad_w = pad_hw
        if pad_h > 0:
            x = x[:, :, :-pad_h, :]
        if pad_w > 0:
            x = x[:, :, :, :-pad_w]
        return x

    def _window_mean(self, x, win):
        """
        x: [B,1,H,W], H/W should be divisible by win
        return: [B,1,H//win,W//win]
        """
        return F.avg_pool2d(x, kernel_size=win, stride=win)

    def _expand_window_map(self, x_w, target_hw):
        """
        x_w: [B,1,Hw,Ww]
        return: [B,1,H,W]
        """
        x = x_w.repeat_interleave(self.small_win, dim=2)
        x = x.repeat_interleave(self.small_win, dim=3)
        h, w = target_hw
        return x[:, :, :h, :w]

    def _valid_avg_pool(self, x, k):
        """
        对窗口证据图做 valid-neighbor normalized average。
        避免边界 padding 0 被错误计入平均值。

        x: [B,1,Hw,Ww]
        """
        pad = k // 2
        return F.avg_pool2d(
            x,
            kernel_size=k,
            stride=1,
            padding=pad,
            count_include_pad=False
        )

    def forward(self, rgb, ir, rgb_out, ir_out):
        b, c, h, w = rgb.shape

        # -------------------------------------------------
        # 1) 得到 LA-SCI 的补偿残差
        # -------------------------------------------------
        delta_r = rgb_out - rgb
        delta_i = ir_out - ir

        # -------------------------------------------------
        # 2) 从补偿残差中估计原始交互证据 G
        #    残差越大，说明 LA-SCI 对该区域的跨模态补偿越强。
        # -------------------------------------------------
        evidence_map = 0.5 * (
            delta_r.abs().mean(dim=1, keepdim=True) +
            delta_i.abs().mean(dim=1, keepdim=True)
        )  # [B,1,H,W]

        # 归一化到 0~1，避免不同层/不同 batch 响应尺度差异过大。
        e_min = evidence_map.amin(dim=(2, 3), keepdim=True)
        e_max = evidence_map.amax(dim=(2, 3), keepdim=True)
        evidence_map = (evidence_map - e_min) / (e_max - e_min + self.eps)

        # -------------------------------------------------
        # 3) 在 small-window 网格上计算 G_w
        # -------------------------------------------------
        evidence_pad, pad_hw = self._pad_to_multiple(evidence_map, self.small_win)
        G_w = self._window_mean(evidence_pad, self.small_win)  # [B,1,Hw,Ww]

        # -------------------------------------------------
        # 4) 邻域证据 G_bar
        #    K=3 表示最多看周围 8 个窗口 + 当前窗口。
        # -------------------------------------------------
        G_bar = self._valid_avg_pool(G_w, self.neigh_ks)

        # -------------------------------------------------
        # 5) 目标响应保护图 O_w
        #    用 fused feature 生成目标响应，保护小目标孤立高交互。
        # -------------------------------------------------
        fused = rgb_out + ir_out
        O = self.obj_head(fused)  # [B,1,H,W]
        O_pad, _ = self._pad_to_multiple(O, self.small_win)
        O_w = self._window_mean(O_pad, self.small_win)  # [B,1,Hw,Ww]

        # -------------------------------------------------
        # 6) 一致性权重 C_w
        #    C 高：相信原始 LA-SCI 证据 G_w；
        #    C 低：更多采用邻域共识 G_bar。
        # -------------------------------------------------
        C_w = self.consistency(torch.cat([G_w, G_bar, O_w], dim=1))

        # -------------------------------------------------
        # 7) 校准后的交互证据 G_hat_w
        # -------------------------------------------------
        G_hat_w = C_w * G_w + (1.0 - C_w) * G_bar

        # 限制校准强度，避免彻底推翻 LA-SCI 原始证据。
        alpha = self.max_alpha * torch.sigmoid(self.alpha_raw)

        # 最终证据 = 原证据 + 小幅一致性修正
        G_cal_w = (1.0 - alpha) * G_w + alpha * G_hat_w

        # -------------------------------------------------
        # 8) 扩展回特征图大小
        # -------------------------------------------------
        G_cal = self._expand_window_map(G_cal_w, target_hw=(evidence_pad.shape[2], evidence_pad.shape[3]))
        G_cal = self._remove_pad(G_cal, pad_hw)
        G_cal = G_cal.to(device=rgb.device, dtype=rgb.dtype)

        # -------------------------------------------------
        # 9) 用校准证据重新控制 LA-SCI 补偿残差
        #    不直接改 fused feature，只改 LA-SCI 的 delta 强度。
        # -------------------------------------------------
        corr_w = torch.tanh(G_hat_w - G_w)
        corr = self._expand_window_map(corr_w, target_hw=(evidence_pad.shape[2], evidence_pad.shape[3]))
        corr = self._remove_pad(corr, pad_hw)

        alpha = self.max_alpha * torch.sigmoid(self.alpha_raw)

        rgb_cal = rgb_out + alpha * corr * delta_r
        ir_cal  = ir_out  + alpha * corr * delta_i

        return rgb_cal, ir_cal

class LASCICECC(nn.Module):
    """
    LA-SCI + CECC 包装模块。

    作用：
        1) 先用 LASCIModule 做低光感知局部跨模态交互；
        2) 得到 rgb_out / ir_out；
        3) 用 CECC 校准 LA-SCI 产生的交互补偿证据；
        4) 输出 rgb_cal + ir_cal，直接送入 YOLO Neck。

    与 LASCIDSSF 的区别：
        - LASCIDSSF: LA-SCI 后再用 DSSF_SS2D 重写融合特征；
        - LASCICECC: LA-SCI 后只校准其补偿证据，不做全局特征重建。
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=6,
        num_heads=4,
        budget_ratio=0.35,
        softmax_tau=0.25,
        lambda_low=0.5,
        neigh_ks=3,
    ):
        super().__init__()

        self.lasci = LASCIModule(
            c=c,
            embed_dim=embed_dim,
            small_win=small_win,
            large_win=large_win,
            num_heads=num_heads,
            budget_ratio=budget_ratio,
            softmax_tau=softmax_tau,
            lambda_low=lambda_low,
            return_pair=True,
        )

        self.cecc = CECC(
            c=c,
            small_win=small_win,
            neigh_ks=neigh_ks,
        )

    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat, rgb_raw] 或 [rgb_feat, ir_feat]

        Returns:
            fused: [B,C,H,W]
        """
        rgb, ir = x[0], x[1]

        rgb_out, ir_out = self.lasci(x)

        rgb_cal, ir_cal = self.cecc(
            rgb=rgb,
            ir=ir,
            rgb_out=rgb_out,
            ir_out=ir_out
        )

        fused = rgb_cal + ir_cal
        return fused

class LASCIModulev2(nn.Module):
    """
    mask guidance
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=4,
        large_win=6,
        num_heads=4,
        budget_ratio=0.35,
        softmax_tau=0.25,
        lambda_low=0.5,
        low_thr=100.0 / 255.0,
        low_smooth=0.08,
        rgb_raw_range=1.0,
        chunk_windows=2048,
        init_gamma=0.05,
        return_pair=False,
        debug = False,
        last_debug = None,
        use_mask_guidance =True,
    ):
        """
        Args:
            c: 输入 RGB/IR 特征通道数。
            embed_dim: 局部相关性计算空间维度。
            small_win: 当前模态 Query 小窗口大小。
            large_win: 另一模态 Key/Value 搜索大窗口大小。
            num_heads: 局部跨模态注意力头数。
            budget_ratio: 软交互预算比例；不是 hard top-k 比例。
                          例如 0.35 表示全图窗口共享约 35% 的交互预算。
            softmax_tau: soft budget 的温度；越小越集中，越大越平均。
            lambda_low: 低光方向调节强度；低光越强，越增强 IR->RGB，越抑制 RGB->IR。
            low_thr: RGB 亮度低光阈值，默认约 100/255。
            low_smooth: 低光 soft threshold 的平滑系数。
            rgb_raw_range: rgb_raw 的取值范围。YOLO 中通常为 1.0；如果输入 0~255，设为 255.0。
            chunk_windows: 所有窗口做 attention 时的分块大小，用于控制显存。
            init_gamma: 残差回写强度初值。设小一点可避免训练初期破坏 baseline。
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert large_win >= small_win, "large_win should be >= small_win"

        self.c = c
        self.embed_dim = embed_dim
        self.small_win = small_win
        self.large_win = large_win
        self.budget_ratio = float(budget_ratio)
        self.softmax_tau = float(softmax_tau)
        self.lambda_low = float(lambda_low)
        self.low_thr = float(low_thr)
        self.low_smooth = float(low_smooth)
        self.rgb_raw_range = float(rgb_raw_range)
        self.chunk_windows = int(chunk_windows)
        self.return_pair = return_pair
        self.debug = bool(debug)
        self.last_debug = None

        #mask guidance
        self.use_mask_guidance = bool(use_mask_guidance)
        self.cache_aux_mask_logits = False
        self.last_mask_logits = None
        mask_hidden = max(embed_dim // 4, 16)
        self.mask_head = nn.Sequential(
            nn.Conv2d(embed_dim * 2, mask_hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(mask_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(mask_hidden, mask_hidden, kernel_size=3, padding=1, groups=mask_hidden, bias=False),
            nn.BatchNorm2d(mask_hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(mask_hidden, 1, kernel_size=1, bias=True),
        )
        # query modulation strength，初始非常小，避免一开始破坏原 LA-SCI
        self.mask_alpha_raw = nn.Parameter(torch.tensor(-6.0))
        self.mask_alpha_max = 0.5
        # loss.py 会读取这个
        self.last_mask_logits = None

        # 将 RGB/IR 特征投影到跨模态相关性计算空间。
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )

        # token 级归一化，稳定窗口内 attention 与分数计算。
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        # 双向局部跨模态交互。
        self.rgb_from_ir = _LASCILocalCrossAttention(embed_dim, num_heads)
        self.ir_from_rgb = _LASCILocalCrossAttention(embed_dim, num_heads)

        # 将 embed_dim 空间的更新量投影回原通道数 c。
        self.rgb_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # 残差强度。给一个较小初值，既允许分支获得梯度，又不至于一开始破坏原特征。
        self.gamma_r = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gamma_i = nn.Parameter(torch.tensor(float(init_gamma)))

        # 输出后轻量稳定层。不是融合，只是对双路各自做局部整理。
        self.rgb_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.act = nn.SiLU(inplace=True)

    # --------------------------------------------------------
    # 亮度与低光项
    # --------------------------------------------------------
    def _build_lowlight_map(self, rgb_raw, target_hw, ref_tensor=None):
        """
        从 RGB 原图构造低光退化图，而不是从特征响应估计低光。

        Args:
            rgb_raw: [B,3,H0,W0]，RGB 原图，建议范围 [0,1]
            target_hw: 当前特征层大小 (H,W)
            ref_tensor: 用于对齐 device / dtype 的参考张量，一般传 rgb_e

        Returns:
            low_map: [B,1,H,W]，局部越暗越接近 1
        """
        if rgb_raw is None:
            return None

        # 关键：不要 rgb_raw.float()
        # 验证时模型可能是 half，必须让 low_map 和特征 dtype 一致
        if ref_tensor is not None:
            rgb = rgb_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        else:
            rgb = rgb_raw

        # 如果输入是 0~255，用户应设置 rgb_raw_range=255.0
        if self.rgb_raw_range != 1.0:
            rgb = rgb / self.rgb_raw_range

        rgb = rgb.clamp(0.0, 1.0)

        # RGB -> luma，使用同 dtype 计算
        y = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]

        # resize 到当前特征层大小
        y = F.interpolate(y, size=target_hw, mode="bilinear", align_corners=False)

        # soft low-light
        low_thr = y.new_tensor(self.low_thr)
        low_smooth = max(self.low_smooth, 1e-6)
        low_map = torch.sigmoid((low_thr - y) / low_smooth)

        return low_map

    @staticmethod
    def _window_mean_map(x, small_win):
        """
        将 [B,1,H,W] 的 map 池化成窗口级 [B,L]。
        H/W 必须能被 small_win 整除。
        """
        return F.avg_pool2d(x, kernel_size=small_win, stride=small_win).flatten(1)

    def _compute_router_gate(self, rgb_small_n, ir_small_n):
        """
        根据当前 small window 的 RGB/IR 特征生成两个方向的窗口级 gate。

        Args:
            rgb_small_n: [B,L,N,E]
            ir_small_n : [B,L,N,E]

        Returns:
            gate_i2r: [B,L]，IR -> RGB 方向残差门控
            gate_r2i: [B,L]，RGB -> IR 方向残差门控
            router_logits: [B,L,2]，调试用
        """
        # 1) GAP over window tokens
        rgb_desc = rgb_small_n.mean(dim=2)  # [B,L,E]
        ir_desc = ir_small_n.mean(dim=2)    # [B,L,E]

        # 2) concat 当前窗口两种模态的局部描述
        router_in = torch.cat([rgb_desc, ir_desc], dim=-1)  # [B,L,2E]

        # 3) dtype/device 对齐，避免 half 验证时报错
        mlp_dtype = self.router_mlp[0].weight.dtype
        mlp_device = self.router_mlp[0].weight.device

        router_in = router_in.to(device=mlp_device, dtype=mlp_dtype)
        router_logits = self.router_mlp(router_in)  # [B,L,2]

        # 4) sigmoid：两个方向独立开关，不做 softmax 竞争
        gates = torch.sigmoid(router_logits)

        gates = gates.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)
        router_logits = router_logits.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)

        gate_i2r = gates[..., 0]  # [B,L]
        gate_r2i = gates[..., 1]  # [B,L]

        return gate_i2r, gate_r2i, router_logits

    def _cross_attention_all_windows(self, q_win, kv_win, attn_module):
        """
        所有窗口都做局部跨模态 attention，但用 chunk 控制显存。

        Args:
            q_win: [B,L,Nq,E]
            kv_win: [B,L,Nk,E]
            attn_module: _LASCILocalCrossAttention

        Returns:
            out: [B,L,Nq,E]
        """
        b, l, nq, e = q_win.shape
        nk = kv_win.shape[2]

        q_flat = q_win.reshape(b * l, nq, e)
        kv_flat = kv_win.reshape(b * l, nk, e)

        outs = []
        total = q_flat.shape[0]
        chunk = max(1, self.chunk_windows)
        for st in range(0, total, chunk):
            ed = min(st + chunk, total)
            outs.append(attn_module(q_flat[st:ed], kv_flat[st:ed]))

        out = torch.cat(outs, dim=0)
        return out.view(b, l, nq, e)

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: [rgb_feat, ir_feat, rgb_raw] 或 [rgb_feat, ir_feat]

        Returns:
            (rgb_out, ir_out)
        """
        if not isinstance(x, (list, tuple)):
            raise TypeError(f"LASCIModule expects list/tuple input, got {type(x)}")

        if len(x) == 3:
            rgb, ir, rgb_raw = x[0], x[1], x[2]
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)
            rgb_raw = rgb_raw.to(device=rgb.device, dtype=rgb.dtype)
        elif len(x) == 2:
            rgb, ir = x[0], x[1]
            rgb_raw = None
        else:
            raise ValueError(f"LASCIModule expects [rgb, ir] or [rgb, ir, rgb_raw], got len={len(x)}")

        assert rgb.shape == ir.shape, f"rgb and ir feature shapes must match, got {rgb.shape} vs {ir.shape}"

        rgb_res, ir_res = rgb, ir
        b, c, h0, w0 = rgb.shape

        # 1) 投影到相关性空间。
        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)

        if self.use_mask_guidance:
            # 第一版建议 detach 输入，避免 mask loss 直接扰动主干/embedding
            mask_in = torch.cat([rgb_e.detach(), ir_e.detach()], dim=1)

            mask_in = mask_in.to(
                device=self.mask_head[0].weight.device,
                dtype=self.mask_head[0].weight.dtype
            )

            mask_logits = self.mask_head(mask_in)

            #默认不缓存，避免 model.info / EMA deepcopy / validation 阶段残留计算图
            self.last_mask_logits = None
            if (
                self.cache_aux_mask_logits
                and self.training
                and torch.is_grad_enabled()
            ):
                self.last_mask_logits = mask_logits

            # 用于 query modulation
            mask_prob = torch.sigmoid(mask_logits)
            mask_prob = mask_prob.to(device=rgb_e.device, dtype=rgb_e.dtype)
        else:
            self.last_mask_logits = None
            mask_prob = None

        # 2) pad 到 small_win 的整数倍。
        rgb_e, pad_hw = _lasci_pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = _lasci_pad_to_multiple(ir_e, self.small_win)
        if mask_prob is not None:
            mask_prob, _ = _lasci_pad_to_multiple(mask_prob, self.small_win)

        _, _, h, w = rgb_e.shape

        # 3) 从 RGB 原图构造局部低光图，注意它是局部的，不是整图单一标量。

        # 4) 构建 small query windows 和 large search windows。
        rgb_small, gh, gw = _lasci_window_partition(rgb_e, self.small_win)  # [B,L,Ns,E]
        ir_small, _, _ = _lasci_window_partition(ir_e, self.small_win)
        if mask_prob is not None:
            mask_small, _, _ = _lasci_window_partition(mask_prob, self.small_win)
            # mask_small: [B, L, small_win*small_win, 1]
        else:
            mask_small = None

        rgb_large = _lasci_extract_large_windows(rgb_e, self.small_win, self.large_win)  # [B,L,Nl,E]
        ir_large = _lasci_extract_large_windows(ir_e, self.small_win, self.large_win)

        # 5) LayerNorm 后计算分数和 attention。
        rgb_small_n = self.rgb_norm(rgb_small)
        ir_small_n = self.ir_norm(ir_small)
        rgb_large_n = self.rgb_norm(rgb_large)
        ir_large_n = self.ir_norm(ir_large)

        # # 6) 三项评分：特征差异、余弦不一致、RGB 原图局部低光。

        # 8) 双向局部跨模态交互：
        #    RGB small -> IR large: 用 IR 补 RGB
        #    IR  small -> RGB large: 用 RGB 补 IR
        rgb_q = rgb_small_n
        ir_q = ir_small_n
        if mask_small is not None:
            alpha = self.mask_alpha_max * torch.sigmoid(self.mask_alpha_raw)
            alpha = alpha.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)
            mask_small = mask_small.to(device=rgb_small_n.device, dtype=rgb_small_n.dtype)
            # residual-style query modulation
            rgb_q = rgb_small_n * (1.0 + alpha * mask_small)
            ir_q = ir_small_n * (1.0 + alpha * mask_small)
        delta_i2r = self._cross_attention_all_windows(
            rgb_q,
            ir_large_n,
            self.rgb_from_ir
        )
        delta_r2i = self._cross_attention_all_windows(
            ir_q,
            rgb_large_n,
            self.ir_from_rgb
        )

        # 9) 低光方向调节是局部窗口级的：

        # 10) 窗口还原回特征图。
        delta_i2r = _lasci_window_reverse(delta_i2r, gh, gw, self.small_win)
        delta_r2i = _lasci_window_reverse(delta_r2i, gh, gw, self.small_win)

        # 11) 去 pad。
        delta_i2r = _lasci_remove_pad(delta_i2r, pad_hw)
        delta_r2i = _lasci_remove_pad(delta_r2i, pad_hw)

        # 12) 投影回原通道，并以残差形式回写。
        # 保险：Conv/BN 权重可能是 half，输入必须对齐
        delta_i2r = delta_i2r.to(
            device=self.rgb_out[0].weight.device,
            dtype=self.rgb_out[0].weight.dtype
        )
        delta_r2i = delta_r2i.to(
            device=self.ir_out[0].weight.device,
            dtype=self.ir_out[0].weight.dtype
        )
        delta_i2r = self.rgb_out(delta_i2r)
        delta_r2i = self.ir_out(delta_r2i)

        delta_i2r = delta_i2r.to(device=rgb_res.device, dtype=rgb_res.dtype)
        delta_r2i = delta_r2i.to(device=ir_res.device, dtype=ir_res.dtype)

        rgb_out = rgb_res + self.gamma_r.to(dtype=rgb_res.dtype) * delta_i2r
        ir_out = ir_res + self.gamma_i.to(dtype=ir_res.dtype) * delta_r2i

        # 13) 轻量 FFN 整理；仍然是双路输出，不做最终融合。
        rgb_out = self.act(rgb_out + self.rgb_ffn(rgb_out))
        ir_out = self.act(ir_out + self.ir_ffn(ir_out))

        # if self.return_pair:
        #     return rgb_out, ir_out
        if getattr(self, "return_pair", False):
            return rgb_out, ir_out
        return rgb_out + ir_out


# ============================================================
# Reconstructed LA-SCI
#
# Core behavior:
#   1) IR 3*3 center window is the spatial reference.
#   2) A paired RGB 5*5 region is unfolded into nine overlapping
#      3*3 candidates: center plus eight one-token shifts.
#   3) The routing score is
#          best non-center cosine - center cosine.
#   4) A threshold selects expanded 5*5 interaction. The selected
#      ratio is bounded per image and per feature scale.
#   5) Unselected windows still use center 3*3 interaction.
#   6) One shared spatial mask controls both directions, while the
#      two cross-attention modules keep independent Q/K/V weights.
#   7) RGB luminance only modulates the two residual directions.
# ============================================================


def _lasci_extract_shifted_candidates(large_windows, small_win, large_win):
    """Split each large window into center + eight shifted small candidates.

    Args:
        large_windows: [B, L, large_win*large_win, C]
        small_win: candidate side length
        large_win: search-region side length

    Returns:
        candidates: [B, L, 9, small_win*small_win, C]

    For the reconstructed first version, large_win must equal
    small_win + 2. With small_win=3 and large_win=5, a stride-1
    unfold produces the nine offsets in row-major order:
        0 1 2
        3 4 5
        6 7 8
    Candidate 4 is the center candidate.
    """
    if large_win != small_win + 2:
        raise ValueError(
            f"nine-candidate routing requires large_win=small_win+2, "
            f"got small_win={small_win}, large_win={large_win}"
        )

    b, l, n, c = large_windows.shape
    if n != large_win * large_win:
        raise ValueError(
            f"large window token count {n} does not match {large_win}*{large_win}"
        )

    grid = large_windows.view(b, l, large_win, large_win, c)
    grid = grid.permute(0, 1, 4, 2, 3).contiguous()
    grid = grid.view(b * l, c, large_win, large_win)

    patches = F.unfold(grid, kernel_size=small_win, stride=1)
    num_candidates = patches.shape[-1]
    if num_candidates != 9:
        raise RuntimeError(f"expected 9 shifted candidates, got {num_candidates}")

    patches = patches.transpose(1, 2).contiguous()
    patches = patches.view(
        b * l,
        num_candidates,
        c,
        small_win * small_win,
    )
    patches = patches.permute(0, 1, 3, 2).contiguous()
    return patches.view(
        b,
        l,
        num_candidates,
        small_win * small_win,
        c,
    )


def _lasci_candidate_valid_mask(gh, gw, device):
    """Return [L,9] validity for one-token shifted candidates.

    Outward shifts at the outermost feature boundary are invalid and
    must not win because of replicated padding.
    """
    offsets = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),  (0, 0),  (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )
    rows = torch.arange(gh, device=device).view(gh, 1).expand(gh, gw)
    cols = torch.arange(gw, device=device).view(1, gw).expand(gh, gw)
    valid = []
    for dy, dx in offsets:
        valid.append(
            (rows + dy >= 0)
            & (rows + dy < gh)
            & (cols + dx >= 0)
            & (cols + dx < gw)
        )
    return torch.stack(valid, dim=-1).view(gh * gw, 9)

class LASCIModulev3(nn.Module):
    """IR-referenced adaptive local cross-modal interaction.

    The range decision is asymmetric: an IR center window compares
    against nine RGB candidates. The evidence transfer remains
    bidirectional, with independent attention parameters.
    """

    def __init__(
        self,
        c,
        embed_dim=128,
        small_win=3,
        large_win=5,
        num_heads=4,
        sim_threshold=0.05,
        min_ratio=0.0,
        max_ratio=0.35,
        lambda_low=0.5,
        low_thr=100.0 / 255.0,
        low_smooth=0.08,
        rgb_raw_range=1.0,
        chunk_windows=2048,
        init_gamma=0.05,
        return_pair=False,
        debug=False,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        if large_win != small_win + 2:
            raise ValueError(
                "the nine-candidate version requires large_win=small_win+2"
            )
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError(f"min_ratio must be in [0,1], got {min_ratio}")
        if not 0.0 <= max_ratio <= 1.0:
            raise ValueError(f"max_ratio must be in [0,1], got {max_ratio}")
        if min_ratio > max_ratio:
            raise ValueError(
                f"min_ratio={min_ratio} cannot exceed max_ratio={max_ratio}"
            )

        self.c = int(c)
        self.embed_dim = int(embed_dim)
        self.small_win = int(small_win)
        self.large_win = int(large_win)
        self.num_heads = int(num_heads)
        self.sim_threshold = float(sim_threshold)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.lambda_low = float(lambda_low)
        self.low_thr = float(low_thr)
        self.low_smooth = float(low_smooth)
        self.rgb_raw_range = float(rgb_raw_range)
        self.chunk_windows = int(chunk_windows)
        self.return_pair = bool(return_pair)
        self.debug = bool(debug)
        self.last_debug = None

        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(inplace=True),
        )
        self.rgb_norm = nn.LayerNorm(embed_dim)
        self.ir_norm = nn.LayerNorm(embed_dim)

        # rgb_from_ir: RGB query, IR key/value
        # ir_from_rgb: IR query, RGB key/value
        self.rgb_from_ir = _LASCILocalCrossAttention(embed_dim, num_heads)
        self.ir_from_rgb = _LASCILocalCrossAttention(embed_dim, num_heads)

        self.rgb_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        self.gamma_r = nn.Parameter(torch.tensor(float(init_gamma)))
        self.gamma_i = nn.Parameter(torch.tensor(float(init_gamma)))

        self.rgb_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_ffn = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = nn.SiLU(inplace=True)

    def _build_lowlight_map(self, rgb_raw, target_hw, ref_tensor):
        """Build local RGB low-light intensity in [0,1]."""
        if rgb_raw is None:
            return None

        rgb = rgb_raw.to(device=ref_tensor.device, dtype=ref_tensor.dtype)
        if rgb.shape[1] < 3:
            raise ValueError(
                f"rgb_raw must contain at least 3 channels, got {rgb.shape}"
            )
        rgb = rgb[:, :3]
        if self.rgb_raw_range != 1.0:
            rgb = rgb / self.rgb_raw_range
        rgb = rgb.clamp(0.0, 1.0)

        # Keep the coefficients and parameters of the validated legacy branch.
        y = (
            0.299 * rgb[:, 0:1]
            + 0.587 * rgb[:, 1:2]
            + 0.114 * rgb[:, 2:3]
        )
        y = F.interpolate(
            y,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        low_smooth = max(self.low_smooth, 1e-6)
        return torch.sigmoid((self.low_thr - y) / low_smooth)

    def _compute_shared_route(
        self,
        ir_small_n,
        rgb_candidates_n,
        gh,
        gw,
    ):
        """Compute IR-reference routing from raw cosine margins.

        Similarity is measured in the learned Q/K space of the
        IR-query/RGB-key attention direction. This avoids comparing
        two independently projected modality vectors directly.
        """
        with torch.no_grad():
            ir_proto = ir_small_n.mean(dim=2)
            rgb_proto = rgb_candidates_n.mean(dim=3)

            q = self.ir_from_rgb.q_proj(ir_proto)
            k = self.ir_from_rgb.k_proj(rgb_proto)
            q = F.normalize(q, p=2, dim=-1, eps=1e-6)
            k = F.normalize(k, p=2, dim=-1, eps=1e-6)
            similarities = (q.unsqueeze(2) * k).sum(dim=-1)

            valid = _lasci_candidate_valid_mask(
                gh,
                gw,
                device=similarities.device,
            )
            similarities = similarities.masked_fill(
                ~valid.unsqueeze(0),
                float("-inf"),
            )

            center_similarity = similarities[:, :, 4]
            noncenter_indices = similarities.new_tensor(
                [0, 1, 2, 3, 5, 6, 7, 8],
                dtype=torch.long,
            )
            noncenter = similarities.index_select(2, noncenter_indices)
            best_noncenter, best_local_index = noncenter.max(dim=2)
            best_candidate_index = noncenter_indices[best_local_index]
            score = best_noncenter - center_similarity
            route_mask = self._bounded_threshold_mask(score)

        return (
            route_mask,
            score,
            center_similarity,
            best_noncenter,
            best_candidate_index,
        )

    def _bounded_threshold_mask(self, score):
        """Threshold then enforce per-image lower/upper ratio bounds."""
        b, l = score.shape
        masks = []
        max_count = int(math.floor(self.max_ratio * l))
        min_count = int(math.ceil(self.min_ratio * l))
        max_count = max(0, min(l, max_count))
        min_count = max(0, min(max_count, min_count))

        for batch_index in range(b):
            score_b = score[batch_index]
            finite = torch.isfinite(score_b)
            mask_b = finite & (score_b > self.sim_threshold)
            count = int(mask_b.sum().item())

            order = torch.argsort(score_b, descending=True)
            order = order[finite[order]]

            if count > max_count:
                mask_b = torch.zeros_like(mask_b)
                if max_count > 0:
                    mask_b[order[:max_count]] = True
            elif count < min_count:
                mask_b = torch.zeros_like(mask_b)
                if min_count > 0:
                    mask_b[order[:min_count]] = True

            masks.append(mask_b)

        return torch.stack(masks, dim=0)

    def _attention_in_chunks(self, q_flat, kv_flat, attn_module):
        if q_flat.shape[0] == 0:
            return q_flat
        outputs = []
        chunk = max(1, self.chunk_windows)
        for start in range(0, q_flat.shape[0], chunk):
            end = min(start + chunk, q_flat.shape[0])
            outputs.append(attn_module(q_flat[start:end], kv_flat[start:end]))
        return torch.cat(outputs, dim=0)

    def _routed_cross_attention(
        self,
        q_windows,
        center_kv_windows,
        large_kv_windows,
        route_mask,
        attn_module,
    ):
        """Use 3*3 K/V for unselected and 5*5 K/V for selected windows."""
        b, l, nq, e = q_windows.shape
        q_flat = q_windows.reshape(b * l, nq, e)
        center_flat = center_kv_windows.reshape(
            b * l,
            center_kv_windows.shape[2],
            e,
        )
        large_flat = large_kv_windows.reshape(
            b * l,
            large_kv_windows.shape[2],
            e,
        )
        mask_flat = route_mask.reshape(b * l)

        # Under AMP, LayerNorm/query tensors may remain float32 while
        # Linear/attention outputs are autocast to float16. Index writes
        # require an exact dtype match, so every routed branch is cast
        # back to the query/output dtype before index_copy.
        output_flat = torch.zeros_like(q_flat)
        center_indices = torch.nonzero(~mask_flat, as_tuple=False).flatten()
        large_indices = torch.nonzero(mask_flat, as_tuple=False).flatten()

        if center_indices.numel() > 0:
            center_output = self._attention_in_chunks(
                q_flat.index_select(0, center_indices),
                center_flat.index_select(0, center_indices),
                attn_module,
            )
            center_output = center_output.to(
                device=output_flat.device,
                dtype=output_flat.dtype,
            )
            output_flat = output_flat.index_copy(
                0,
                center_indices,
                center_output,
            )
        if large_indices.numel() > 0:
            large_output = self._attention_in_chunks(
                q_flat.index_select(0, large_indices),
                large_flat.index_select(0, large_indices),
                attn_module,
            )
            large_output = large_output.to(
                device=output_flat.device,
                dtype=output_flat.dtype,
            )
            output_flat = output_flat.index_copy(
                0,
                large_indices,
                large_output,
            )
        return output_flat.view(b, l, nq, e)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            raise TypeError(
                f"LASCIModule expects list/tuple input, got {type(x)}"
            )
        if len(x) == 3:
            rgb, ir, rgb_raw = x
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)
            rgb_raw = rgb_raw.to(device=rgb.device, dtype=rgb.dtype)
        elif len(x) == 2:
            rgb, ir = x
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)
            rgb_raw = None
        else:
            raise ValueError(
                "LASCIModule expects [rgb, ir] or [rgb, ir, rgb_raw]"
            )
        if rgb.shape != ir.shape:
            raise ValueError(
                f"rgb and ir feature shapes must match, got "
                f"{rgb.shape} vs {ir.shape}"
            )

        rgb_res, ir_res = rgb, ir
        _, _, h0, w0 = rgb.shape

        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)
        rgb_e, pad_hw = _lasci_pad_to_multiple(rgb_e, self.small_win)
        ir_e, _ = _lasci_pad_to_multiple(ir_e, self.small_win)

        rgb_small, gh, gw = _lasci_window_partition(
            rgb_e,
            self.small_win,
        )
        ir_small, _, _ = _lasci_window_partition(
            ir_e,
            self.small_win,
        )
        rgb_large = _lasci_extract_large_windows(
            rgb_e,
            self.small_win,
            self.large_win,
        )
        ir_large = _lasci_extract_large_windows(
            ir_e,
            self.small_win,
            self.large_win,
        )

        rgb_small_n = self.rgb_norm(rgb_small)
        ir_small_n = self.ir_norm(ir_small)
        rgb_large_n = self.rgb_norm(rgb_large)
        ir_large_n = self.ir_norm(ir_large)

        rgb_candidates_n = _lasci_extract_shifted_candidates(
            rgb_large_n,
            self.small_win,
            self.large_win,
        )
        (
            route_mask,
            route_score,
            center_similarity,
            best_noncenter,
            best_candidate_index,
        ) = self._compute_shared_route(
            ir_small_n,
            rgb_candidates_n,
            gh,
            gw,
        )

        # IR -> RGB: RGB queries IR center/large evidence.
        delta_i2r = self._routed_cross_attention(
            rgb_small_n,
            ir_small_n,
            ir_large_n,
            route_mask,
            self.rgb_from_ir,
        )
        # RGB -> IR: IR queries RGB center/large evidence.
        delta_r2i = self._routed_cross_attention(
            ir_small_n,
            rgb_small_n,
            rgb_large_n,
            route_mask,
            self.ir_from_rgb,
        )

        # Low light controls residual directions only, not range selection.
        low_map = self._build_lowlight_map(
            rgb_raw,
            target_hw=(h0, w0),
            ref_tensor=rgb_e,
        )
        if low_map is None:
            low_win = rgb_small_n.new_zeros(
                rgb_small_n.shape[0],
                rgb_small_n.shape[1],
            )
        else:
            low_map, _ = _lasci_pad_to_multiple(
                low_map,
                self.small_win,
            )
            low_win = F.avg_pool2d(
                low_map,
                kernel_size=self.small_win,
                stride=self.small_win,
            ).flatten(1)
            low_win = low_win.to(
                device=delta_i2r.device,
                dtype=delta_i2r.dtype,
            )

        gate_i2r = 1.0 + self.lambda_low * low_win
        gate_r2i = (1.0 - self.lambda_low * low_win).clamp(min=0.0)
        delta_i2r = delta_i2r * gate_i2r.unsqueeze(-1).unsqueeze(-1)
        delta_r2i = delta_r2i * gate_r2i.unsqueeze(-1).unsqueeze(-1)

        if self.debug and not self.training:
            self.last_debug = {
                "route_score": route_score.detach().float().cpu().numpy(),
                "route_mask": route_mask.detach().cpu().numpy(),
                "center_similarity": (
                    center_similarity.detach().float().cpu().numpy()
                ),
                "best_noncenter": (
                    best_noncenter.detach().float().cpu().numpy()
                ),
                "best_noncenter_index": (
                    best_candidate_index.detach().cpu().numpy()
                ),
                "low_win": low_win.detach().float().cpu().numpy(),
                "selected_ratio": (
                    route_mask.float().mean(dim=1).cpu().numpy()
                ),
                "gh": gh,
                "gw": gw,
            }

        delta_i2r = _lasci_window_reverse(
            delta_i2r,
            gh,
            gw,
            self.small_win,
        )
        delta_r2i = _lasci_window_reverse(
            delta_r2i,
            gh,
            gw,
            self.small_win,
        )
        delta_i2r = _lasci_remove_pad(delta_i2r, pad_hw)
        delta_r2i = _lasci_remove_pad(delta_r2i, pad_hw)

        delta_i2r = self.rgb_out(delta_i2r)
        delta_r2i = self.ir_out(delta_r2i)
        delta_i2r = delta_i2r.to(
            device=rgb_res.device,
            dtype=rgb_res.dtype,
        )
        delta_r2i = delta_r2i.to(
            device=ir_res.device,
            dtype=ir_res.dtype,
        )

        rgb_out = (
            rgb_res
            + self.gamma_r.to(dtype=rgb_res.dtype) * delta_i2r
        )
        ir_out = (
            ir_res
            + self.gamma_i.to(dtype=ir_res.dtype) * delta_r2i
        )
        rgb_out = self.act(rgb_out + self.rgb_ffn(rgb_out))
        ir_out = self.act(ir_out + self.ir_ffn(ir_out))

        if self.return_pair:
            return rgb_out, ir_out
        return rgb_out + ir_out


"""IR-guided confidence-rejected local RGB offset sampling.

This file is self-contained so the class can first be tested outside Ultralytics
and then copied into ``ultralytics/nn/modules/block.py``.

Direction convention:
    RGB->IR means that IR is the spatial reference.  At an IR position p, the
    module searches RGB positions p + delta and writes the sampled RGB feature
    into the IR-referenced fused representation.

The initial research version intentionally contains no cross-attention,
bidirectional update, low-light branch, or P4/P5 alignment.
"""

class IRGuidedSelectiveOffset(nn.Module):
    """Selectively sample RGB into the IR coordinate frame at P3.

    Args:
        c: RGB/IR input channel count. Both inputs must have this channel count.
        embed_dim: Common-space projection dimension. The supplied projection
            probe checkpoint currently uses 128.
        search_radius: Candidate-center radius in feature cells. Radius 2 gives
            a 5*5 candidate set.
        patch_radius: Local descriptor radius. Radius 1 gives a 3*3 patch.
        margin_threshold: The best-noncenter minus center score threshold.
        entropy_threshold: Maximum normalized candidate entropy for a reliable
            shift.
        softmax_tau: Temperature used for candidate probabilities and soft
            offset expectation.
        route_mode: One of ``center``, ``all``, or ``reliable``.
            center: always use zero offset (Add baseline behavior).
            all: apply the soft offset at every fully valid search position.
            reliable: apply it only where margin and entropy pass the route.
        freeze_projection: Freeze RGB/IR projection weights. This should be True
            for the current RGB-box-supervised upper-bound experiment.
        projection_path: Optional ``projection_probe.pt`` path. The checkpoint
            must contain ``matcher``, ``input_dim``, and ``embed_dim``.
        require_full_search: If True, only cells for which all candidates and
            all patch tokens are in bounds may move. Other cells keep zero
            offset.
        debug: Store detached CPU diagnostic maps in ``last_debug``. Do not
            enable this during normal training because it synchronizes devices.

    Input:
        ``[rgb, ir]``, both B*C*H*W tensors with identical shapes.

    Output:
        ``ir + routed_rgb``. At inactive locations, ``routed_rgb == rgb``, so
        the exact zero-offset behavior is the ordinary Add fusion.
    """

    ROUTE_INVALID = 0
    ROUTE_ALIGNED = 1
    ROUTE_RELIABLE = 2
    ROUTE_UNCERTAIN = 3
    VALID_ROUTE_MODES = {"center", "all", "reliable"}

    def __init__(
        self,
        c: int,
        embed_dim: int = 64,
        search_radius: int = 2,
        patch_radius: int = 1,
        margin_threshold: float = 0.10,
        entropy_threshold: float = 0.68,
        softmax_tau: float = 0.10,
        route_mode: str = "reliable",
        freeze_projection: bool = True,
        projection_path: str = "",
        require_full_search: bool = True,
        debug: bool = False,
    ) -> None:
        super().__init__()
        if c <= 0 or embed_dim <= 0:
            raise ValueError("c and embed_dim must be positive.")
        if search_radius < 0 or patch_radius < 0:
            raise ValueError("search_radius and patch_radius must be non-negative.")
        if softmax_tau <= 0:
            raise ValueError("softmax_tau must be positive.")
        if not 0.0 <= entropy_threshold <= 1.0:
            raise ValueError("entropy_threshold must be in [0, 1].")
        route_mode = str(route_mode).lower().strip()
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(
                "route_mode must be one of {}, got {!r}.".format(
                    sorted(self.VALID_ROUTE_MODES), route_mode
                )
            )

        self.c = int(c)
        self.input_dim = int(c)
        self.embed_dim = int(embed_dim)
        self.search_radius = int(search_radius)
        self.patch_radius = int(patch_radius)
        self.margin_threshold = float(margin_threshold)
        self.entropy_threshold = float(entropy_threshold)
        self.softmax_tau = float(softmax_tau)
        self.route_mode = route_mode
        self.freeze_projection = bool(freeze_projection)
        self.projection_path = str(projection_path or "")
        self.require_full_search = bool(require_full_search)
        self.debug = bool(debug)
        self.last_debug: Optional[Dict[str, torch.Tensor]] = None

        # Keep nn.Linear rather than Conv2d so projection_probe.pt loads exactly.
        self.rgb_proj = nn.Linear(self.c, self.embed_dim, bias=True)
        self.ir_proj = nn.Linear(self.c, self.embed_dim, bias=True)
        nn.init.orthogonal_(self.rgb_proj.weight)
        nn.init.orthogonal_(self.ir_proj.weight)
        nn.init.zeros_(self.rgb_proj.bias)
        nn.init.zeros_(self.ir_proj.bias)

        candidates = [
            (dx, dy)
            for dy in range(-self.search_radius, self.search_radius + 1)
            for dx in range(-self.search_radius, self.search_radius + 1)
        ]
        self.candidate_list: List[Tuple[int, int]] = candidates
        self.center_index = candidates.index((0, 0))
        self.register_buffer(
            "candidate_offsets",
            torch.tensor(candidates, dtype=torch.float32),
            persistent=False,
        )

        if self.projection_path:
            self.load_projection_checkpoint(self.projection_path)
        self.set_projection_trainable(not self.freeze_projection)

    def extra_repr(self) -> str:
        return (
            "c={}, embed_dim={}, search_radius={}, patch_radius={}, "
            "margin_threshold={}, entropy_threshold={}, softmax_tau={}, "
            "route_mode={!r}, freeze_projection={}, require_full_search={}".format(
                self.c,
                self.embed_dim,
                self.search_radius,
                self.patch_radius,
                self.margin_threshold,
                self.entropy_threshold,
                self.softmax_tau,
                self.route_mode,
                self.freeze_projection,
                self.require_full_search,
            )
        )

    def set_route_mode(self, route_mode: str) -> None:
        """Change the ablation mode without rebuilding the model."""
        route_mode = str(route_mode).lower().strip()
        if route_mode not in self.VALID_ROUTE_MODES:
            raise ValueError(
                "route_mode must be one of {}, got {!r}.".format(
                    sorted(self.VALID_ROUTE_MODES), route_mode
                )
            )
        self.route_mode = route_mode

    def set_projection_trainable(self, trainable: bool) -> None:
        for parameter in self.rgb_proj.parameters():
            parameter.requires_grad_(bool(trainable))
        for parameter in self.ir_proj.parameters():
            parameter.requires_grad_(bool(trainable))
        self.freeze_projection = not bool(trainable)

    def load_projection_checkpoint(self, path: str) -> None:
        """Load the RGB-box-supervised common-space probe checkpoint."""
        checkpoint_path = Path(path).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Projection checkpoint does not exist: {}".format(checkpoint_path)
            )
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state = checkpoint.get("matcher", checkpoint)

        checkpoint_input = checkpoint.get("input_dim")
        checkpoint_embed = checkpoint.get("embed_dim")
        if checkpoint_input is not None and int(checkpoint_input) != self.c:
            raise ValueError(
                "Projection input mismatch: module c={} but checkpoint input_dim={}."
                .format(self.c, checkpoint_input)
            )
        if checkpoint_embed is not None and int(checkpoint_embed) != self.embed_dim:
            raise ValueError(
                "Projection embed mismatch: module embed_dim={} but checkpoint "
                "embed_dim={}.".format(self.embed_dim, checkpoint_embed)
            )

        required = {
            "rgb_proj.weight",
            "rgb_proj.bias",
            "ir_proj.weight",
            "ir_proj.bias",
        }
        missing = sorted(required.difference(state.keys()))
        if missing:
            raise ValueError(
                "Projection checkpoint is missing keys: {}".format(missing)
            )
        self.rgb_proj.load_state_dict(
            {"weight": state["rgb_proj.weight"], "bias": state["rgb_proj.bias"]},
            strict=True,
        )
        self.ir_proj.load_state_dict(
            {"weight": state["ir_proj.weight"], "bias": state["ir_proj.bias"]},
            strict=True,
        )
        self.projection_path = str(checkpoint_path.resolve())

    @staticmethod
    def _shift_source_to_destination(
        tensor: torch.Tensor, dx: int, dy: int
    ) -> torch.Tensor:
        """Return out[..., y, x] = tensor[..., y + dy, x + dx]."""
        if tensor.ndim < 2:
            raise ValueError("Shifted tensor must have at least two dimensions.")
        height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
        output = torch.zeros_like(tensor)

        dst_x0 = max(0, -dx)
        dst_x1 = min(width, width - dx)
        dst_y0 = max(0, -dy)
        dst_y1 = min(height, height - dy)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return output
        src_x0, src_x1 = dst_x0 + dx, dst_x1 + dx
        src_y0, src_y1 = dst_y0 + dy, dst_y1 + dy
        output[..., dst_y0:dst_y1, dst_x0:dst_x1] = tensor[
            ..., src_y0:src_y1, src_x0:src_x1
        ]
        return output

    def _project(self, feature: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        # Linear consumes the last dimension and therefore matches the probe.
        projected = projection(feature.permute(0, 2, 3, 1))
        projected = F.normalize(projected, dim=-1, eps=1e-6)
        return projected.permute(0, 3, 1, 2).contiguous()

    def _dense_candidate_scores(
        self, rgb: torch.Tensor, ir: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return float32 score maps and validity masks, both B*K*H*W."""
        rgb_projected = self._project(rgb, self.rgb_proj)
        ir_projected = self._project(ir, self.ir_proj)
        batch, _, height, width = rgb_projected.shape
        kernel = 2 * self.patch_radius + 1

        # float32 score computation keeps entropy/margin stable under AMP.
        ir_score = ir_projected.float()
        base_valid = torch.ones(
            (batch, 1, height, width),
            device=rgb.device,
            dtype=torch.float32,
        )
        score_maps: List[torch.Tensor] = []
        valid_maps: List[torch.Tensor] = []
        for dx, dy in self.candidate_list:
            shifted_rgb = self._shift_source_to_destination(rgb_projected, dx, dy)
            shifted_valid = self._shift_source_to_destination(base_valid, dx, dy)
            point_score = (ir_score * shifted_rgb.float()).sum(dim=1, keepdim=True)
            point_score = point_score * shifted_valid
            patch_score = F.avg_pool2d(
                point_score,
                kernel_size=kernel,
                stride=1,
                padding=self.patch_radius,
                count_include_pad=True,
            )
            patch_valid_fraction = F.avg_pool2d(
                shifted_valid,
                kernel_size=kernel,
                stride=1,
                padding=self.patch_radius,
                count_include_pad=True,
            )
            candidate_valid = patch_valid_fraction >= (1.0 - 1e-6)
            score_maps.append(patch_score[:, 0])
            valid_maps.append(candidate_valid[:, 0])

        scores = torch.stack(score_maps, dim=1)
        valid = torch.stack(valid_maps, dim=1)
        return scores, valid

    def _route_from_scores(
        self, scores: torch.Tensor, valid: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Build probability, offset, confidence, and three-state route maps."""
        if scores.ndim != 4 or valid.shape != scores.shape:
            raise ValueError("scores and valid must both have shape B*K*H*W.")
        candidate_count = int(scores.shape[1])
        if candidate_count != len(self.candidate_list):
            raise ValueError("Candidate count does not match module geometry.")

        center_valid = valid[:, self.center_index]
        full_search_valid = valid.all(dim=1)
        any_search_valid = valid.any(dim=1)
        usable = center_valid & any_search_valid
        if self.require_full_search:
            usable = usable & full_search_valid

        negative = torch.finfo(scores.dtype).min
        masked_scores = scores.masked_fill(~valid, negative)
        maximum = masked_scores.max(dim=1).values
        maximum = torch.where(usable, maximum, torch.zeros_like(maximum))
        logits = (masked_scores - maximum[:, None]) / self.softmax_tau
        exp_scores = torch.where(valid, torch.exp(logits), torch.zeros_like(logits))
        denominator = exp_scores.sum(dim=1).clamp_min(1e-12)
        probabilities = exp_scores / denominator[:, None]
        probabilities = torch.where(
            usable[:, None], probabilities, torch.zeros_like(probabilities)
        )

        offsets = self.candidate_offsets.to(
            device=scores.device, dtype=probabilities.dtype
        )
        pred_dx = (probabilities * offsets[:, 0][None, :, None, None]).sum(dim=1)
        pred_dy = (probabilities * offsets[:, 1][None, :, None, None]).sum(dim=1)

        entropy_raw = -torch.where(
            probabilities > 0,
            probabilities * torch.log(probabilities.clamp_min(1e-12)),
            torch.zeros_like(probabilities),
        ).sum(dim=1)
        valid_count = valid.sum(dim=1)
        entropy_denominator = torch.log(
            valid_count.clamp_min(2).to(dtype=probabilities.dtype)
        )
        normalized_entropy = entropy_raw / entropy_denominator
        normalized_entropy = torch.where(
            usable, normalized_entropy, torch.ones_like(normalized_entropy)
        )

        noncenter = [
            index for index in range(candidate_count) if index != self.center_index
        ]
        best_noncenter = masked_scores[:, noncenter].max(dim=1).values
        center_score = masked_scores[:, self.center_index]
        margin = best_noncenter - center_score
        margin = torch.where(usable, margin, torch.zeros_like(margin))

        route = torch.full_like(valid_count, self.ROUTE_INVALID, dtype=torch.uint8)
        aligned = usable & (margin <= self.margin_threshold)
        reliable = (
            usable
            & (margin > self.margin_threshold)
            & (normalized_entropy <= self.entropy_threshold)
        )
        uncertain = (
            usable
            & (margin > self.margin_threshold)
            & (normalized_entropy > self.entropy_threshold)
        )
        route[aligned] = self.ROUTE_ALIGNED
        route[reliable] = self.ROUTE_RELIABLE
        route[uncertain] = self.ROUTE_UNCERTAIN

        return {
            "probabilities": probabilities,
            "pred_dx": pred_dx,
            "pred_dy": pred_dy,
            "entropy": normalized_entropy,
            "margin": margin,
            "route": route,
            "usable": usable,
            "full_search_valid": full_search_valid,
        }

    @staticmethod
    def _base_grid(
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # align_corners=False: feature-cell center x maps to 2*(x+0.5)/W-1.
        ys = (torch.arange(height, device=device, dtype=dtype) + 0.5)
        xs = (torch.arange(width, device=device, dtype=dtype) + 0.5)
        ys = 2.0 * ys / float(height) - 1.0
        xs = 2.0 * xs / float(width) - 1.0
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        return grid.unsqueeze(0).expand(batch, -1, -1, -1).clone()

    def _sample_rgb(
        self,
        rgb: torch.Tensor,
        pred_dx: torch.Tensor,
        pred_dy: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = rgb.shape
        dx = torch.where(active, pred_dx, torch.zeros_like(pred_dx))
        dy = torch.where(active, pred_dy, torch.zeros_like(pred_dy))
        dx = dx.clamp(-float(self.search_radius), float(self.search_radius))
        dy = dy.clamp(-float(self.search_radius), float(self.search_radius))

        grid = self._base_grid(
            batch, height, width, rgb.device, rgb.dtype
        )
        grid[..., 0] = grid[..., 0] + dx.to(rgb.dtype) * (2.0 / float(width))
        grid[..., 1] = grid[..., 1] + dy.to(rgb.dtype) * (2.0 / float(height))
        sampled_rgb = F.grid_sample(
            rgb,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )

        # reliable位置使用重采样结果；
        # aligned、uncertain及边界无效位置严格保留原RGB。
        return torch.where(active[:, None], sampled_rgb, rgb)

    def _save_debug(
        self, route_data: Dict[str, torch.Tensor], active: torch.Tensor
    ) -> None:
        if not self.debug:
            self.last_debug = None
            return
        self.last_debug = {
            "route": route_data["route"].detach().cpu(),
            "margin": route_data["margin"].detach().cpu(),
            "entropy": route_data["entropy"].detach().cpu(),
            "pred_dx": route_data["pred_dx"].detach().cpu(),
            "pred_dy": route_data["pred_dy"].detach().cpu(),
            "active": active.detach().cpu(),
            "full_search_valid": route_data["full_search_valid"].detach().cpu(),
        }

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                "IRGuidedSelectiveOffset expects [rgb_feat, ir_feat]."
            )
        rgb, ir = x
        if not isinstance(rgb, torch.Tensor) or not isinstance(ir, torch.Tensor):
            raise TypeError("Both inputs must be torch.Tensor objects.")
        if rgb.shape != ir.shape or rgb.ndim != 4:
            raise ValueError(
                "RGB/IR must have identical B*C*H*W shapes, got {} and {}."
                .format(tuple(rgb.shape), tuple(ir.shape))
            )
        if int(rgb.shape[1]) != self.c:
            raise ValueError(
                "Input channel mismatch: module c={} but feature channels={}."
                .format(self.c, int(rgb.shape[1]))
            )
        if rgb.device != ir.device or rgb.dtype != ir.dtype:
            ir = ir.to(device=rgb.device, dtype=rgb.dtype)

        if self.route_mode == "center":
            self.last_debug = None
            return ir + rgb

        scores, valid = self._dense_candidate_scores(rgb, ir)
        route_data = self._route_from_scores(scores, valid)
        if self.route_mode == "all":
            active = route_data["usable"]
        else:
            active = route_data["route"].eq(self.ROUTE_RELIABLE)

        routed_rgb = self._sample_rgb(
            rgb, route_data["pred_dx"], route_data["pred_dy"], active
        )
        self._save_debug(route_data, active)
        return ir + routed_rgb


"""
ProgressiveSCI（优化版）
========================

本文件是追加到 ultralytics/nn/modules/block.py 末尾的模块代码，
不是 Ultralytics 原始 block.py 的完整替代文件。

与上一版保持一致的接口：
- ProgressiveSCI 输入：[rgb_feat, ir_feat, optional_rgb_raw]
- ProgressiveSCI 输出：(rgb_out, ir_out)
- YAML 参数顺序：
    [embed_dim, center_win, support_size, num_heads,
     max_iters, relation_dim, selection_tau, lambda_low]
- SCISelect 与 SCIPairFuse 的使用方式不变；tasks.py 不需要修改。

本版主要优化：
1. 9 个候选窗口一次性张量化评分，删除逐候选 Python 循环；
2. 候选池化特征、候选关系投影、位置编码在迭代前只计算一次；
3. 选中候选通过小型候选—支持域映射矩阵一次性组合；
4. 删除训练路径中的 GPU 张量 Python 条件判断，避免频繁同步；
5. 使用轻量显式多头交叉注意力，减少小窗口场景下 MultiheadAttention 调度开销；
6. 仍保持“每轮重算中心相关关系、候选价值与 STOP”的渐进语义。

边界处理：
- 中心窗口不重叠；
- H/W 不能被中心窗口整除时，仅在右侧和下侧 replicate padding；
- 输出前严格裁回原尺寸；
- 兼容 640×512、640×640 对应的 P3/P4/P5 特征尺寸。

当前允许的几何配置：
- 3×3 中心、5×5 支持域；
- 4×4 中心、6×6 支持域；
以及一般形式 support_size = center_win + 2。
上述配置均产生 9 个偏移候选。
"""


def _psci_pad_to_multiple(
    x: torch.Tensor,
    multiple: int,
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """将 BCHW 特征在右侧和下侧补齐到 multiple 的整数倍。"""
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor, got shape={tuple(x.shape)}")
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        # 复制填充避免零填充在边界制造异常暗区。
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return x, (pad_h, pad_w)


def _psci_remove_pad(x: torch.Tensor, pad_hw: Tuple[int, int]) -> torch.Tensor:
    """裁掉 _psci_pad_to_multiple 增加的右侧/下侧区域。"""
    pad_h, pad_w = pad_hw
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


def _psci_partition_nonoverlap(
    x: torch.Tensor,
    win: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    将 BCHW 特征划分为不重叠中心窗口。

    Returns:
        windows: [B, L, win², C]
        gh, gw: 中心窗口网格高宽，L = gh × gw
    """
    b, c, h, w = x.shape
    if h % win != 0 or w % win != 0:
        raise ValueError(f"H={h}, W={w} must be divisible by win={win}")
    gh, gw = h // win, w // win
    x = x.view(b, c, gh, win, gw, win)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, gh * gw, win * win, c), gh, gw


def _psci_reverse_nonoverlap(
    windows: torch.Tensor,
    gh: int,
    gw: int,
    win: int,
) -> torch.Tensor:
    """将 [B,L,win²,C] 的不重叠窗口恢复为 BCHW 特征图。"""
    b, l, n, c = windows.shape
    if l != gh * gw or n != win * win:
        raise ValueError(
            f"Invalid window shape={tuple(windows.shape)}, gh={gh}, gw={gw}, win={win}"
        )
    x = windows.view(b, gh, gw, win, win, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, gh * win, gw * win)


def _psci_extract_support(
    x: torch.Tensor,
    center_win: int,
    support_size: int,
) -> torch.Tensor:
    """
    为每个不重叠中心窗口提取另一模态支持域。

    例如 center_win=3、support_size=5：
    - 特征图四周 replicate padding 1 个特征点；
    - kernel=5、stride=3 执行 unfold；
    - 每个中心得到 25 个支持域 token。

    Returns:
        support: [B, L, support_size², C]
    """
    if center_win <= 0 or support_size < center_win:
        raise ValueError("support_size must be >= center_win > 0")
    diff = support_size - center_win
    if diff % 2 != 0:
        raise ValueError(
            "support_size-center_win must be even so the support is centered; "
            f"got center_win={center_win}, support_size={support_size}"
        )

    margin = diff // 2
    if margin:
        x = F.pad(x, (margin, margin, margin, margin), mode="replicate")

    b, c, _, _ = x.shape
    patches = F.unfold(x, kernel_size=support_size, stride=center_win)
    patches = patches.transpose(1, 2).contiguous()  # [B,L,C*S²]
    return patches.view(b, -1, support_size * support_size, c)


def _psci_candidate_indices(center_win: int, support_size: int) -> torch.Tensor:
    """
    返回 K 个候选窗口在支持域中的扁平 token 索引。

    3×3→5×5 或 4×4→6×6 时，均有 9 个候选；
    每个候选包含 center_win² 个 token。
    """
    shift_count = support_size - center_win + 1
    indices: List[List[int]] = []
    for oy in range(shift_count):
        for ox in range(shift_count):
            ids = []
            for yy in range(center_win):
                for xx in range(center_win):
                    ids.append((oy + yy) * support_size + (ox + xx))
            indices.append(ids)
    return torch.tensor(indices, dtype=torch.long)


def _psci_relative_positions(center_win: int, support_size: int) -> torch.Tensor:
    """生成候选相对中心的二维偏移，顺序与候选索引一致。"""
    shift_count = support_size - center_win + 1
    center_offset = (shift_count - 1) / 2.0
    positions = []
    for oy in range(shift_count):
        for ox in range(shift_count):
            positions.append([oy - center_offset, ox - center_offset])
    return torch.tensor(positions, dtype=torch.float32)


def _psci_candidate_token_matrix(
    candidate_index: torch.Tensor,
    support_token_count: int,
) -> torch.Tensor:
    """
    构造候选到支持域的 token 映射矩阵。

    Returns:
        matrix: [K,N,S²]

    matrix[k,n,s] = 1 表示第 k 个候选的第 n 个 token
    读取支持域中的第 s 个 token。
    """
    k, n = candidate_index.shape
    matrix = torch.zeros(k, n, support_token_count, dtype=torch.float32)
    matrix.scatter_(2, candidate_index.unsqueeze(-1), 1.0)
    return matrix


def _psci_luminance_low_map(
    rgb_raw: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
    ref: torch.Tensor,
) -> Optional[torch.Tensor]:
    """
    从 RGB 原图生成局部低照度图：low = 1 - luminance。

    rgb_raw 通常已归一化到 [0,1]；如果仍在 [0,255]，自动除以 255。
    输出大小与当前特征层一致。
    """
    if rgb_raw is None:
        return None
    if rgb_raw.ndim != 4 or rgb_raw.shape[1] < 3:
        return None

    rgb = rgb_raw[:, :3].to(device=ref.device, dtype=ref.dtype)

    # 该判断完全留在设备端，不调用 .item()，避免 CPU/GPU 同步。
    max_value = rgb.detach().amax() if rgb.numel() else rgb.new_tensor(1.0)
    scale = torch.where(
        max_value > 1.5,
        rgb.new_tensor(255.0),
        rgb.new_tensor(1.0),
    )
    rgb = (rgb / scale).clamp(0.0, 1.0)

    # 若数据加载器实际使用 BGR，需要相应调换通道系数。
    lum = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    low = 1.0 - lum
    return F.interpolate(low, size=target_hw, mode="bilinear", align_corners=False)


class _DirectionalCrossAttention(nn.Module):
    """
    单方向局部多头交叉注意力。

    这里不用 nn.MultiheadAttention，而是显式完成 Q/K/V、分头矩阵乘法和输出投影。
    对 N=9 或 N=16 的大量小窗口，通常能减少通用 MHA 封装的调度开销。
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"embed_dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        hidden = max(dim // 2, 16)
        self.message_gate = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target: [B,L,N,C]，当前待更新中心状态
            source: [B,L,N,C]，本轮选中的另一模态候选

        Returns:
            message: [B,L,N,C]
            gate: [B,L,1,1]
        """
        b, l, n, c = target.shape
        bl = b * l

        q = self.q_proj(self.q_norm(target)).view(
            bl, n, self.num_heads, self.head_dim
        )
        q = q.permute(0, 2, 1, 3)  # [BL,H,N,D]

        kv = self.kv_proj(self.kv_norm(source)).view(
            bl, n, 2, self.num_heads, self.head_dim
        )
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # [BL,H,N,D]

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = F.softmax(attn, dim=-1)
        message = torch.matmul(attn, v)
        message = message.permute(0, 2, 1, 3).contiguous().view(b, l, n, c)
        message = self.out_proj(message)

        t_pool = target.mean(dim=2)
        s_pool = source.mean(dim=2)
        gate = self.message_gate(
            torch.cat([t_pool, s_pool, torch.abs(t_pool - s_pool)], dim=-1)
        )
        return message, gate.unsqueeze(-1)


class _CandidateValueEstimator(nn.Module):
    """
    向量化候选价值评估器。

    与上一版的语义保持一致：
    - 当前中心状态每轮都会变化并重新投影；
    - 候选特征、候选池化、候选关系投影和位置编码属于静态源信息，
      在同一方向的多轮迭代中只计算一次；
    - 评分仍联合中心、候选、差异、乘积、token 结构关系、位置、
      模态可靠性与低照度信息；
    - STOP 仍基于当前中心和未选择候选的摘要动态计算。
    """

    def __init__(
        self,
        dim: int,
        relation_dim: int,
        center_win: int,
        support_size: int,
        pos_dim: int = 16,
    ):
        super().__init__()
        self.dim = dim
        self.center_win = center_win
        self.token_count = center_win * center_win
        self.support_token_count = support_size * support_size

        candidate_index = _psci_candidate_indices(center_win, support_size)
        relative_positions = _psci_relative_positions(center_win, support_size)
        token_matrix = _psci_candidate_token_matrix(
            candidate_index, self.support_token_count
        )
        pool_matrix = token_matrix.sum(dim=1) / float(self.token_count)

        self.register_buffer("candidate_index", candidate_index, persistent=False)
        self.register_buffer(
            "candidate_flat_index", candidate_index.reshape(-1), persistent=False
        )
        self.register_buffer("candidate_pool_matrix", pool_matrix, persistent=False)
        self.register_buffer("relative_positions", relative_positions, persistent=False)
        self.num_candidates = candidate_index.shape[0]

        self.target_rel = nn.Linear(dim, relation_dim, bias=False)
        self.source_rel = nn.Linear(dim, relation_dim, bias=False)
        self.position_encoder = nn.Sequential(
            nn.Linear(2, pos_dim),
            nn.SiLU(),
            nn.Linear(pos_dim, pos_dim),
        )

        # 4*dim：target、candidate、绝对差、逐元素乘积
        # N*N：完整 token 结构关系
        # pos_dim：相对位置编码
        # 2：source_quality 与 target_low
        score_in = 4 * dim + self.token_count * self.token_count + pos_dim + 2
        hidden = max(dim * 2, 64)
        hidden2 = max(hidden // 2, 32)
        self.score_head = nn.Sequential(
            nn.Linear(score_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden2),
            nn.SiLU(),
            nn.Linear(hidden2, 1),
        )

        # STOP：当前中心、剩余候选均值/最大摘要、剩余比例、低照度。
        stop_in = 3 * dim + 2
        self.stop_head = nn.Sequential(
            nn.Linear(stop_in, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def prepare_source(
        self,
        source_support: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预计算多轮迭代中不变的源候选信息。

        Args:
            source_support: [B,L,S²,C]

        Returns:
            candidate_pools: [B,L,K,C]
            source_rel_support: [B,L,S²,D]
            pos_embed: [K,P]
        """
        pool_matrix = self.candidate_pool_matrix.to(dtype=source_support.dtype)

        # [K,S²] × [B,L,S²,C] -> [B,L,K,C]
        candidate_pools = torch.einsum(
            "ks,blsc->blkc", pool_matrix, source_support
        )

        # 源端投影与归一化只需执行一次；下一轮只重算目标端。
        source_rel_support = F.normalize(
            self.source_rel(source_support), dim=-1, eps=1e-6
        )
        pos_embed = self.position_encoder(
            self.relative_positions.to(dtype=source_support.dtype)
        )
        return candidate_pools, source_rel_support, pos_embed

    def forward(
        self,
        target: torch.Tensor,
        candidate_pools: torch.Tensor,
        source_rel_support: torch.Tensor,
        pos_embed: torch.Tensor,
        selected_mask: torch.Tensor,
        source_quality: torch.Tensor,
        target_low: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            target: [B,L,N,C]
            candidate_pools: [B,L,K,C]，由 prepare_source 缓存
            source_rel_support: [B,L,S²,D]，由 prepare_source 缓存
            pos_embed: [K,P]，由 prepare_source 缓存
            selected_mask: [B,L,K] bool
            source_quality: [B,L,K]
            target_low: [B,L]

        Returns:
            candidate_logits: [B,L,K]
            stop_logit: [B,L,1]
        """
        b, l, n, c = target.shape
        if n != self.token_count:
            raise ValueError(f"Expected {self.token_count} target tokens, got {n}")

        k_num = self.num_candidates
        t_pool = target.mean(dim=2)  # [B,L,C]
        t_rel = F.normalize(self.target_rel(target), dim=-1, eps=1e-6)

        # 先计算中心 N 个 token 与完整支持域 S² 个 token 的关系：
        # [B,L,N,D] × [B,L,S²,D] -> [B,L,N,S²]
        # 这样源关系投影只计算一次，也避免为重叠候选重复投影相同 token。
        relation_full = torch.einsum(
            "blnd,blsd->blns", t_rel, source_rel_support
        )

        # 从完整关系中一次性抽取 K 个候选各自的 N 个源 token。
        # [B,L,N,K*N] -> [B,L,K,N,N]
        relation = relation_full.index_select(
            dim=-1, index=self.candidate_flat_index
        )
        relation = relation.view(b, l, n, k_num, n)
        relation = relation.permute(0, 1, 3, 2, 4).contiguous()
        relation_flat = relation.flatten(start_dim=-2)  # [B,L,K,N*N]

        # 所有候选一次性构造评分特征，不再执行 9 次 score_head。
        t_expand = t_pool.unsqueeze(2).expand(-1, -1, k_num, -1)
        pos_expand = pos_embed.view(1, 1, k_num, -1).expand(b, l, -1, -1)
        quality = source_quality.unsqueeze(-1)
        low = target_low.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, k_num, -1)

        score_feature = torch.cat(
            [
                t_expand,
                candidate_pools,
                torch.abs(t_expand - candidate_pools),
                t_expand * candidate_pools,
                relation_flat,
                pos_expand,
                quality,
                low,
            ],
            dim=-1,
        )
        candidate_logits = self.score_head(score_feature).squeeze(-1)

        # STOP 根据尚未选择的候选摘要动态计算。
        remaining = (~selected_mask).to(candidate_pools.dtype)
        remain_count = remaining.sum(dim=-1, keepdim=True).clamp_min(1.0)
        remain_mean = (
            candidate_pools * remaining.unsqueeze(-1)
        ).sum(dim=2) / remain_count

        neg = torch.finfo(candidate_pools.dtype).min
        remain_max = candidate_pools.masked_fill(
            selected_mask.unsqueeze(-1), neg
        ).amax(dim=2)
        all_selected = selected_mask.all(dim=-1, keepdim=True)
        remain_max = torch.where(
            all_selected, torch.zeros_like(remain_max), remain_max
        )
        remain_ratio = remaining.mean(dim=-1, keepdim=True)

        stop_feature = torch.cat(
            [
                t_pool,
                remain_mean,
                remain_max,
                remain_ratio,
                target_low.unsqueeze(-1),
            ],
            dim=-1,
        )
        stop_logit = self.stop_head(stop_feature)
        return candidate_logits, stop_logit


class ProgressiveSCI(nn.Module):
    """
    Progressive Selective Cross-modal Interaction（优化版）。

    YAML 参数顺序（parse_model 自动在最前面插入输入通道 c）：
        [embed_dim, center_win, support_size, num_heads,
         max_iters, relation_dim, selection_tau, lambda_low]

    输入：
        [rgb_feat, ir_feat, rgb_raw] 或 [rgb_feat, ir_feat]

    输出：
        (rgb_out, ir_out)

    模块内部：
    - RGB→IR 与 IR→RGB 平行运行；
    - 每个方向独立维护中心状态、已选掩码与 STOP；
    - 每轮执行“候选评估 → 选择/STOP → 交叉注意力 → 残差更新”；
    - 两个方向始终从模块输入时的原始另一模态支持域读取源信息，
      不使用对方已经更新后的状态，因此不形成循环往返。
    """

    def __init__(
        self,
        c: int,
        embed_dim: int = 64,
        center_win: int = 3,
        support_size: int = 5,
        num_heads: int = 4,
        max_iters: int = 2,
        relation_dim: int = 24,
        selection_tau: float = 0.7,
        lambda_low: float = 0.5,
    ):
        super().__init__()

        # 3×3→5×5 与 4×4→6×6 都满足该关系，并都产生 9 个候选。
        if center_win <= 0 or support_size != center_win + 2:
            raise ValueError(
                "ProgressiveSCI requires support_size = center_win + 2; "
                f"got center_win={center_win}, support_size={support_size}."
            )
        if max_iters < 1:
            raise ValueError("max_iters must be >= 1")
        if max_iters > (support_size - center_win + 1) ** 2:
            raise ValueError("max_iters cannot exceed the number of candidates")
        if not 0.0 <= lambda_low <= 1.0:
            raise ValueError("lambda_low should be in [0,1]")

        self.c = c
        self.embed_dim = embed_dim
        self.center_win = center_win
        self.support_size = support_size
        self.max_iters = max_iters
        self.selection_tau = float(selection_tau)
        self.lambda_low = float(lambda_low)

        candidate_index = _psci_candidate_indices(center_win, support_size)
        token_matrix = _psci_candidate_token_matrix(
            candidate_index, support_size * support_size
        )
        pool_matrix = token_matrix.sum(dim=1) / float(center_win * center_win)

        self.register_buffer("candidate_index", candidate_index, persistent=False)
        self.register_buffer("candidate_token_matrix", token_matrix, persistent=False)
        self.register_buffer("candidate_pool_matrix", pool_matrix, persistent=False)
        self.num_candidates = candidate_index.shape[0]

        # 模态独立输入投影，避免强制共享变换。
        self.rgb_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )
        self.ir_in = nn.Sequential(
            nn.Conv2d(c, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

        # 两个方向拥有独立的价值评估器和交叉注意力参数。
        self.value_r2i = _CandidateValueEstimator(
            embed_dim, relation_dim, center_win, support_size
        )
        self.value_i2r = _CandidateValueEstimator(
            embed_dim, relation_dim, center_win, support_size
        )
        self.attn_r2i = _DirectionalCrossAttention(embed_dim, num_heads)
        self.attn_i2r = _DirectionalCrossAttention(embed_dim, num_heads)

        self.rgb_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )
        self.ir_out = nn.Sequential(
            nn.Conv2d(embed_dim, c, 1, bias=False),
            nn.BatchNorm2d(c),
        )

        # 小正值初始化：保留原分支，同时让新模块早期即可获得梯度。
        self.gamma_rgb = nn.Parameter(torch.tensor(0.1))
        self.gamma_ir = nn.Parameter(torch.tensor(0.1))

        # 默认关闭调试，防止每个 step 的 .item() 触发同步。
        self.debug_enabled = False
        self.last_debug = {}

    def _hard_or_st_choice(self, logits: torch.Tensor) -> torch.Tensor:
        """
        训练：Straight-Through Gumbel-Softmax；
        推理：确定性 argmax。
        """
        if self.training:
            choice = F.gumbel_softmax(
                logits.float(), tau=self.selection_tau, hard=True, dim=-1
            )
            return choice.to(dtype=logits.dtype)
        index = logits.argmax(dim=-1)
        return F.one_hot(index, num_classes=logits.shape[-1]).to(logits.dtype)

    def _candidate_low_values(
        self,
        low_support: Optional[torch.Tensor],
        b: int,
        l: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """一次性计算 K 个 RGB 候选各自的平均低光程度。"""
        if low_support is None:
            return torch.zeros(
                b, l, self.num_candidates, device=device, dtype=dtype
            )

        # low_support: [B,L,S²,1]
        pool_matrix = self.candidate_pool_matrix.to(
            device=low_support.device, dtype=low_support.dtype
        )
        values = torch.einsum("ks,blsc->blkc", pool_matrix, low_support)
        return values.squeeze(-1).to(device=device, dtype=dtype)

    def _compose_selected_source(
        self,
        source_support: torch.Tensor,
        candidate_weight: torch.Tensor,
    ) -> torch.Tensor:
        """
        一次性根据 K 个候选权重构造本轮源窗口。

        candidate_weight 前向为 one-hot；反向保留 ST 路由梯度。
        使用 [K,N,S²] 的固定映射，避免逐候选 index_select 与累加。
        """
        token_matrix = self.candidate_token_matrix.to(
            device=source_support.device, dtype=source_support.dtype
        )

        # 先把候选权重映射为每个目标 token 对支持域 token 的读取权重：
        # [B,L,K] × [K,N,S²] -> [B,L,N,S²]
        token_weight = torch.einsum(
            "blk,kns->blns", candidate_weight, token_matrix
        )

        # [B,L,N,S²] × [B,L,S²,C] -> [B,L,N,C]
        return torch.einsum("blns,blsc->blnc", token_weight, source_support)

    def _run_direction(
        self,
        target_initial: torch.Tensor,
        source_support: torch.Tensor,
        estimator: _CandidateValueEstimator,
        attention: _DirectionalCrossAttention,
        source_quality: torch.Tensor,
        target_low: torch.Tensor,
        low_mode: str,
        source_candidate_low: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在单一方向内执行有限轮渐进选择。

        low_mode:
            "r2i"：RGB 为源，低光时抑制 RGB→IR；
            "i2r"：RGB 为目标，低光时增强 IR→RGB。
        """
        b, l, _, _ = target_initial.shape
        state = target_initial

        selected_mask = torch.zeros(
            b, l, self.num_candidates, device=state.device, dtype=torch.bool
        )
        active = torch.ones(b, l, device=state.device, dtype=state.dtype)
        selected_count = torch.zeros_like(active)

        # 静态源候选信息在所有迭代轮之间复用。
        candidate_pools, source_rel_support, pos_embed = estimator.prepare_source(
            source_support
        )

        for _ in range(self.max_iters):
            candidate_logits, stop_logit = estimator(
                target=state,
                candidate_pools=candidate_pools,
                source_rel_support=source_rel_support,
                pos_embed=pos_embed,
                selected_mask=selected_mask,
                source_quality=source_quality,
                target_low=target_low,
            )

            inactive = active <= 0.0

            # 已选候选不能重复选择；已停止窗口的所有候选也全部屏蔽。
            invalid_candidate = selected_mask | inactive.unsqueeze(-1)
            candidate_logits = candidate_logits.masked_fill(
                invalid_candidate, -1.0e4
            )

            # 已停止窗口强制保持 STOP，不再使用 Python if 判断 GPU 张量。
            stop_logit = torch.where(
                inactive.unsqueeze(-1),
                torch.zeros_like(stop_logit),
                stop_logit,
            )
            logits = torch.cat([candidate_logits, stop_logit], dim=-1)

            choice = self._hard_or_st_choice(logits)
            candidate_weight = (
                choice[..., : self.num_candidates] * active.unsqueeze(-1)
            )
            stop_weight = choice[..., -1] * active
            choose_any = candidate_weight.sum(dim=-1)

            selected_source = self._compose_selected_source(
                source_support, candidate_weight
            )
            message, message_gate = attention(state, selected_source)

            if low_mode == "r2i":
                if source_candidate_low is None:
                    low_factor = torch.ones_like(choose_any)
                else:
                    selected_low = (
                        candidate_weight * source_candidate_low
                    ).sum(dim=-1) / choose_any.clamp_min(1.0e-6)
                    # RGB 源越暗，向 IR 注入越弱。
                    low_factor = (
                        1.0 - self.lambda_low * selected_low
                    ).clamp(0.0, 1.0)
            elif low_mode == "i2r":
                # RGB 目标越暗，越需要 IR 提供稳定结构。
                low_factor = 1.0 + self.lambda_low * target_low
            else:
                raise ValueError(f"Unknown low_mode={low_mode}")

            update_scale = (
                (choose_any * low_factor).unsqueeze(-1).unsqueeze(-1)
                * message_gate
            )
            state = state + update_scale * message

            # 用硬决策更新后续轮次的候选屏蔽和活动状态。
            hard_selected = candidate_weight.detach() > 0.5
            selected_mask = selected_mask | hard_selected
            selected_count = selected_count + (
                choose_any.detach() > 0.5
            ).to(selected_count.dtype)
            active = active * (
                1.0 - (stop_weight.detach() > 0.5).to(active.dtype)
            )

            # 仅推理时允许整体提前退出；训练时不做设备同步。
            if not self.training and bool((active <= 0.0).all()):
                break

        return state, selected_count

    def forward(self, x: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(x, (list, tuple)) or len(x) < 2:
            raise TypeError(
                "ProgressiveSCI expects [rgb_feat, ir_feat, optional_rgb_raw]"
            )

        rgb, ir = x[0], x[1]
        rgb_raw = x[2] if len(x) > 2 else None

        if rgb.shape != ir.shape:
            raise ValueError(
                f"RGB/IR feature shapes must match, got {tuple(rgb.shape)} "
                f"vs {tuple(ir.shape)}"
            )
        b, c, h, w = rgb.shape
        if c != self.c:
            raise ValueError(f"Expected input channels={self.c}, got {c}")

        # 1) 两个模态独立投影到交互嵌入空间。
        rgb_e = self.rgb_in(rgb)
        ir_e = self.ir_in(ir)

        # 2) 将 H/W 补到中心窗口整数倍；输出时裁回。
        rgb_e, pad_hw = _psci_pad_to_multiple(rgb_e, self.center_win)
        ir_e, pad_hw_ir = _psci_pad_to_multiple(ir_e, self.center_win)
        if pad_hw_ir != pad_hw:
            raise RuntimeError("RGB and IR produced inconsistent padding")

        # 3) 构造不重叠中心窗口和固定源支持域。
        rgb_center, gh, gw = _psci_partition_nonoverlap(
            rgb_e, self.center_win
        )
        ir_center, gh_ir, gw_ir = _psci_partition_nonoverlap(
            ir_e, self.center_win
        )
        if (gh, gw) != (gh_ir, gw_ir):
            raise RuntimeError("RGB/IR window grids are inconsistent")

        rgb_support = _psci_extract_support(
            rgb_e, self.center_win, self.support_size
        )
        ir_support = _psci_extract_support(
            ir_e, self.center_win, self.support_size
        )

        # 4) 构造 RGB 局部低照度信息。
        low_map = _psci_luminance_low_map(rgb_raw, (h, w), rgb)
        if low_map is not None:
            low_map, _ = _psci_pad_to_multiple(low_map, self.center_win)
            low_center, _, _ = _psci_partition_nonoverlap(
                low_map, self.center_win
            )
            low_center = low_center.mean(dim=(2, 3))
            low_support = _psci_extract_support(
                low_map, self.center_win, self.support_size
            )
        else:
            low_center = torch.zeros(
                b, gh * gw, device=rgb.device, dtype=rgb_e.dtype
            )
            low_support = None

        rgb_candidate_low = self._candidate_low_values(
            low_support,
            b=b,
            l=gh * gw,
            device=rgb_e.device,
            dtype=rgb_e.dtype,
        )
        rgb_source_quality = 1.0 - rgb_candidate_low
        ir_source_quality = torch.ones_like(rgb_source_quality)

        # 5) 平行双向渐进交互。
        # RGB→IR：固定读取原始 rgb_support，更新 IR 中心状态。
        ir_state, count_r2i = self._run_direction(
            target_initial=ir_center,
            source_support=rgb_support,
            estimator=self.value_r2i,
            attention=self.attn_r2i,
            source_quality=rgb_source_quality,
            target_low=low_center,
            low_mode="r2i",
            source_candidate_low=rgb_candidate_low,
        )

        # IR→RGB：固定读取原始 ir_support，更新 RGB 中心状态。
        rgb_state, count_i2r = self._run_direction(
            target_initial=rgb_center,
            source_support=ir_support,
            estimator=self.value_i2r,
            attention=self.attn_i2r,
            source_quality=ir_source_quality,
            target_low=low_center,
            low_mode="i2r",
            source_candidate_low=None,
        )

        # 6) 只将交互产生的增量投影回原通道并残差注入原分支。
        rgb_delta = rgb_state - rgb_center
        ir_delta = ir_state - ir_center

        rgb_delta = _psci_reverse_nonoverlap(
            rgb_delta, gh, gw, self.center_win
        )
        ir_delta = _psci_reverse_nonoverlap(
            ir_delta, gh, gw, self.center_win
        )
        rgb_delta = _psci_remove_pad(rgb_delta, pad_hw)
        ir_delta = _psci_remove_pad(ir_delta, pad_hw)

        rgb_out = rgb + self.gamma_rgb.to(rgb.dtype) * self.rgb_out(
            rgb_delta
        ).to(rgb.dtype)
        ir_out = ir + self.gamma_ir.to(ir.dtype) * self.ir_out(
            ir_delta
        ).to(ir.dtype)

        if self.debug_enabled:
            # 只在显式开启调试时同步读取统计量。
            self.last_debug = {
                "mean_selected_r2i": float(count_r2i.detach().mean().item()),
                "mean_selected_i2r": float(count_i2r.detach().mean().item()),
                "grid_hw": (gh, gw),
                "input_hw": (h, w),
                "pad_hw": pad_hw,
            }

        return rgb_out, ir_out


class SCISelect(nn.Module):
    """
    从 ProgressiveSCI 输出的 (rgb, ir) 元组中取一条分支。

    YAML：
        - [pair_layer, 1, SCISelect, [0]]  # RGB
        - [pair_layer, 1, SCISelect, [1]]  # IR
    """

    def __init__(self, index: int = 0):
        super().__init__()
        if index not in (0, 1):
            raise ValueError("SCISelect index must be 0 (RGB) or 1 (IR)")
        self.index = index

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                "SCISelect expects the (rgb, ir) tuple from ProgressiveSCI"
            )
        return x[self.index]


class SCIPairFuse(nn.Module):
    """
    将 ProgressiveSCI 的双路增强特征融合为送入 YOLO Neck 的单路特征。

    mode="add"：直接相加；
    mode="concat"：拼接后使用 1×1 Conv-BN-SiLU 压回 c 通道。
    """

    def __init__(self, c: int, mode: str = "add"):
        super().__init__()
        mode = str(mode).lower()
        if mode not in {"add", "concat"}:
            raise ValueError("SCIPairFuse mode must be 'add' or 'concat'")
        self.mode = mode

        if mode == "concat":
            self.fuse = nn.Sequential(
                nn.Conv2d(c * 2, c, 1, bias=False),
                nn.BatchNorm2d(c),
                nn.SiLU(),
            )
        else:
            self.fuse = nn.Identity()

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise TypeError(
                "SCIPairFuse expects the (rgb, ir) tuple from ProgressiveSCI"
            )
        rgb, ir = x
        if self.mode == "add":
            return rgb + ir
        return self.fuse(torch.cat([rgb, ir], dim=1))



# ============================================================
# FRCSF: Frequency Restructuring and Cross-modal State Fusion
# ============================================================

class HaarDWT2D(nn.Module):
    """Fixed orthonormal 2D Haar DWT for BxCxHxW feature maps."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        low = torch.tensor([s, s], dtype=torch.float32)
        high = torch.tensor([-s, s], dtype=torch.float32)
        kernels = torch.stack(
            [
                torch.outer(low, low),    # LL
                torch.outer(low, high),   # LH
                torch.outer(high, low),   # HL
                torch.outer(high, high),  # HH
            ],
            dim=0,
        ).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer("haar_kernels", kernels, persistent=False)

    def forward(self, x):
        B, C, H, W = x.shape
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        weight = self.haar_kernels.to(device=x.device, dtype=x.dtype).repeat(C, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=C)
        h2, w2 = y.shape[-2:]
        y = y.view(B, C, 4, h2, w2)

        ll = y[:, :, 0].contiguous()
        lh = y[:, :, 1].contiguous()
        hl = y[:, :, 2].contiguous()
        hh = y[:, :, 3].contiguous()
        return ll, lh, hl, hh, (H, W)


class HaarIDWT2D(nn.Module):
    """Inverse transform paired with :class:`HaarDWT2D`."""

    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        low = torch.tensor([s, s], dtype=torch.float32)
        high = torch.tensor([-s, s], dtype=torch.float32)
        kernels = torch.stack(
            [
                torch.outer(low, low),
                torch.outer(low, high),
                torch.outer(high, low),
                torch.outer(high, high),
            ],
            dim=0,
        ).unsqueeze(1)
        self.register_buffer("haar_kernels", kernels, persistent=False)

    def forward(self, ll, lh, hl, hh, output_hw=None):
        B, C, H, W = ll.shape
        bands = torch.stack([ll, lh, hl, hh], dim=2).reshape(B, C * 4, H, W)
        weight = self.haar_kernels.to(device=ll.device, dtype=ll.dtype).repeat(C, 1, 1, 1)
        x = F.conv_transpose2d(bands, weight, stride=2, padding=0, groups=C)

        if output_hw is not None:
            out_h, out_w = output_hw
            x = x[:, :, :out_h, :out_w].contiguous()
        return x


class LowFrequencyEnhancement(nn.Module):
    """
    Stage-1 low-frequency enhancement.

    Default: parallel 3x3 / 5x5 / 7x7 convolutions, concatenate, then 1x1 projection.
    Ablation switch:
        mode="standard"  -> ordinary convolutions (default / first version)
        mode="depthwise" -> DWConv + PWConv lightweight variant
    """

    def __init__(self, channels, mode="standard"):
        super().__init__()
        if mode not in {"standard", "depthwise"}:
            raise ValueError(f"Unsupported low-frequency mode: {mode}")
        self.mode = mode

        def make_branch(k):
            if mode == "standard":
                return nn.Sequential(
                    nn.Conv2d(channels, channels, k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.SiLU(inplace=True),
                )
            return nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.BatchNorm2d(channels),
                nn.SiLU(inplace=True),
            )

        self.branch3 = make_branch(3)
        self.branch5 = make_branch(5)
        self.branch7 = make_branch(7)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x3 = self.branch3(x)
        x5 = self.branch5(x)
        x7 = self.branch7(x)
        return self.fuse(torch.cat([x3, x5, x7], dim=1))


class HighFrequencyEnhancement(nn.Module):
    """
    Stage-1 MGFF-style high-frequency spatial selection.

    Channel AvgPool + MaxPool -> concat -> 1x1 conv -> sigmoid spatial weight.

    Default / first version:
        mode="multiply": F' = A * F

    Ablation:
        mode="residual": F' = F + A * F
    """

    def __init__(self, mode="multiply"):
        super().__init__()
        if mode not in {"multiply", "residual"}:
            raise ValueError(f"Unsupported high-frequency mode: {mode}")
        self.mode = mode
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.amax(dim=1, keepdim=True)
        attn = self.spatial_gate(torch.cat([avg_map, max_map], dim=1))
        if self.mode == "multiply":
            return attn * x
        return x + attn * x


class IntraModalFrequencyEnhancement(nn.Module):
    """Stage 1: frequency-specific enhancement inside one modality only."""

    def __init__(self, channels, lf_mode="standard", hf_mode="multiply"):
        super().__init__()
        self.low = LowFrequencyEnhancement(channels, mode=lf_mode)
        # Directional high-frequency bands stay separate and use independent tiny gates.
        self.high_lh = HighFrequencyEnhancement(mode=hf_mode)
        self.high_hl = HighFrequencyEnhancement(mode=hf_mode)
        self.high_hh = HighFrequencyEnhancement(mode=hf_mode)

    def forward(self, bands):
        ll, lh, hl, hh = bands
        return (
            self.low(ll),
            self.high_lh(lh),
            self.high_hl(hl),
            self.high_hh(hh),
        )


class FactorizedBilinearInteraction(nn.Module):
    """
    Stage-2 factorized same-frequency bilinear interaction.

    For one frequency band:
        A_rgb = U_rgb(X_rgb), A_ir = U_ir(X_ir)
        J = A_rgb * A_ir
        dRGB = P_rgb(J), dIR = P_ir(J)

    interaction_direction:
        "bidirectional" -> update RGB and IR (default)
        "rgb2ir"        -> only update IR
        "ir2rgb"        -> only update RGB
    """

    def __init__(self, channels, rank_ratio=4, interaction_direction="bidirectional"):
        super().__init__()
        if rank_ratio <= 0:
            raise ValueError(f"rank_ratio must be positive, got {rank_ratio}")
        if interaction_direction not in {"bidirectional", "rgb2ir", "ir2rgb"}:
            raise ValueError(f"Unsupported Stage-2 direction: {interaction_direction}")

        rank = max(1, channels // int(rank_ratio))
        self.rank = rank
        self.interaction_direction = interaction_direction

        self.u_rgb = nn.Conv2d(channels, rank, 1, bias=True)
        self.u_ir = nn.Conv2d(channels, rank, 1, bias=True)
        self.p_rgb = nn.Conv2d(rank, channels, 1, bias=True)
        self.p_ir = nn.Conv2d(rank, channels, 1, bias=True)

    def forward(self, rgb, ir):
        joint = self.u_rgb(rgb) * self.u_ir(ir)

        if self.interaction_direction in {"bidirectional", "ir2rgb"}:
            rgb = rgb + self.p_rgb(joint)
        if self.interaction_direction in {"bidirectional", "rgb2ir"}:
            ir = ir + self.p_ir(joint)
        return rgb, ir


class SameFrequencyInteraction(nn.Module):
    """
    Stage 2 wrapper.

    share_mode:
        "low_high"       : LL uses theta_L; LH/HL/HH share theta_H (default / first version)
        "all_independent" : LL/LH/HL/HH all use independent modules
        "all_shared"      : all four bands reuse the same interaction module
    """

    def __init__(
        self,
        channels,
        rank_ratio=4,
        share_mode="low_high",
        interaction_direction="bidirectional",
    ):
        super().__init__()
        if share_mode not in {"low_high", "all_independent", "all_shared"}:
            raise ValueError(f"Unsupported Stage-2 share_mode: {share_mode}")
        self.share_mode = share_mode

        make_interaction = lambda: FactorizedBilinearInteraction(
            channels,
            rank_ratio=rank_ratio,
            interaction_direction=interaction_direction,
        )

        if share_mode == "low_high":
            self.low_interaction = make_interaction()
            self.high_interaction = make_interaction()
        elif share_mode == "all_independent":
            self.interactions = nn.ModuleList([make_interaction() for _ in range(4)])
        else:
            self.shared_interaction = make_interaction()

    def forward(self, rgb_bands, ir_bands):
        if self.share_mode == "low_high":
            rgb_ll, ir_ll = self.low_interaction(rgb_bands[0], ir_bands[0])
            rgb_lh, ir_lh = self.high_interaction(rgb_bands[1], ir_bands[1])
            rgb_hl, ir_hl = self.high_interaction(rgb_bands[2], ir_bands[2])
            rgb_hh, ir_hh = self.high_interaction(rgb_bands[3], ir_bands[3])
        elif self.share_mode == "all_independent":
            outputs = [
                module(rgb_band, ir_band)
                for module, rgb_band, ir_band in zip(self.interactions, rgb_bands, ir_bands)
            ]
            (rgb_ll, ir_ll), (rgb_lh, ir_lh), (rgb_hl, ir_hl), (rgb_hh, ir_hh) = outputs
        else:
            outputs = [
                self.shared_interaction(rgb_band, ir_band)
                for rgb_band, ir_band in zip(rgb_bands, ir_bands)
            ]
            (rgb_ll, ir_ll), (rgb_lh, ir_lh), (rgb_hl, ir_hl), (rgb_hh, ir_hh) = outputs

        return (rgb_ll, rgb_lh, rgb_hl, rgb_hh), (ir_ll, ir_lh, ir_hl, ir_hh)


class _FRCSFLayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class CrossModalStateFusion(nn.Module):
    """
    Stage 3: private dual SS2D + Cross-C readout interaction.

    Private per modality:
        normalization, content projection, gate projection, DWConv,
        x-projection (dt/B/C), dt projection, A, D, selective-scan hidden state,
        output normalization and output projection.

    Cross-modal interaction occurs ONLY on C after both modalities produce
    their own C parameters and BEFORE selective scan.

    cross_c_mode:
        "cross"       : proposed Cross-C (default)
        "independent" : C_rgb*=C_rgb, C_ir*=C_ir
        "shared"      : C_rgb*=C_ir*=C_rgb+C_ir

    cross_direction (only for cross_c_mode="cross"):
        "bidirectional" : IR->RGB and RGB->IR (default)
        "ir2rgb"        : only C_ir contributes to RGB readout
        "rgb2ir"        : only C_rgb contributes to IR readout
    """

    def __init__(
        self,
        channels,
        d_state=16,
        ssm_ratio=1.0,
        dt_rank="auto",
        d_conv=3,
        dropout=0.0,
        cross_c_mode="cross",
        cross_direction="bidirectional",
        dt_min=0.001,
        dt_max=0.1,
        dt_scale=1.0,
        dt_init_floor=1e-4,
    ):
        super().__init__()
        if cross_c_mode not in {"cross", "independent", "shared"}:
            raise ValueError(f"Unsupported Cross-C mode: {cross_c_mode}")
        if cross_direction not in {"bidirectional", "ir2rgb", "rgb2ir"}:
            raise ValueError(f"Unsupported Cross-C direction: {cross_direction}")
        if d_conv < 1 or d_conv % 2 == 0:
            raise ValueError(f"d_conv must be a positive odd number, got {d_conv}")
        if selective_scan_fn is None:
            raise ImportError("FRCSFModule Stage 3 requires mamba_ssm selective_scan_fn.")

        self.channels = channels
        self.d_state = int(d_state)
        self.d_inner = max(1, int(channels * ssm_ratio))
        self.K = 4
        self.cross_c_mode = cross_c_mode
        self.cross_direction = cross_direction
        self.dt_rank = math.ceil(channels / 16) if dt_rank == "auto" else int(dt_rank)

        # Private pre-normalization.
        self.norm_rgb = _FRCSFLayerNorm2d(channels)
        self.norm_ir = _FRCSFLayerNorm2d(channels)

        # Independent content/state projections.
        self.state_proj_rgb = nn.Conv2d(channels, self.d_inner, 1, bias=False)
        self.state_proj_ir = nn.Conv2d(channels, self.d_inner, 1, bias=False)
        self.dwconv_rgb = nn.Conv2d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv // 2, groups=self.d_inner, bias=True,
        )
        self.dwconv_ir = nn.Conv2d(
            self.d_inner, self.d_inner, d_conv,
            padding=d_conv // 2, groups=self.d_inner, bias=True,
        )

        # Independent Mamba output gates: not cross-modal reliability gates.
        self.gate_proj_rgb = nn.Conv2d(channels, self.d_inner, 1, bias=False)
        self.gate_proj_ir = nn.Conv2d(channels, self.d_inner, 1, bias=False)
        self.act = nn.SiLU()

        proj_out_dim = self.dt_rank + 2 * self.d_state

        # Private x-projections: each direction produces dt, B, C.
        self.x_proj_rgb_weight = nn.Parameter(torch.empty(self.K, proj_out_dim, self.d_inner))
        self.x_proj_ir_weight = nn.Parameter(torch.empty(self.K, proj_out_dim, self.d_inner))

        # Private dt projections.
        self.dt_proj_rgb_weight = nn.Parameter(torch.empty(self.K, self.d_inner, self.dt_rank))
        self.dt_proj_ir_weight = nn.Parameter(torch.empty(self.K, self.d_inner, self.dt_rank))
        self.dt_proj_rgb_bias = nn.Parameter(torch.empty(self.K, self.d_inner))
        self.dt_proj_ir_bias = nn.Parameter(torch.empty(self.K, self.d_inner))

        # Private A and D. Hidden states are private because scans are invoked separately.
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32)
        A = A.view(1, 1, self.d_state).repeat(self.K, self.d_inner, 1)
        self.A_logs_rgb = nn.Parameter(torch.log(A).reshape(self.K * self.d_inner, self.d_state))
        self.A_logs_ir = nn.Parameter(torch.log(A.clone()).reshape(self.K * self.d_inner, self.d_state))
        self.Ds_rgb = nn.Parameter(torch.ones(self.K * self.d_inner))
        self.Ds_ir = nn.Parameter(torch.ones(self.K * self.d_inner))

        # Cross-C coefficients: zero initialization => exact Independent-C at initialization.
        self.a_i2r = nn.Parameter(torch.zeros(1))
        self.a_r2i = nn.Parameter(torch.zeros(1))

        # Private output normalization / projection.
        self.out_norm_rgb = nn.LayerNorm(self.d_inner)
        self.out_norm_ir = nn.LayerNorm(self.d_inner)
        self.out_proj_rgb = nn.Conv2d(self.d_inner, channels, 1, bias=True)
        self.out_proj_ir = nn.Conv2d(self.d_inner, channels, 1, bias=True)
        self.dropout_rgb = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.dropout_ir = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights(dt_min, dt_max, dt_scale, dt_init_floor)

    def _init_dt_one(self, weight, bias, dt_min, dt_max, dt_scale, dt_init_floor):
        std = (self.dt_rank ** -0.5) * dt_scale
        nn.init.uniform_(weight, -std, std)

        dt = torch.exp(
            torch.rand(self.K, self.d_inner)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            bias.copy_(inv_dt)

    def _init_weights(self, dt_min, dt_max, dt_scale, dt_init_floor):
        nn.init.xavier_uniform_(self.x_proj_rgb_weight)
        nn.init.xavier_uniform_(self.x_proj_ir_weight)
        self._init_dt_one(
            self.dt_proj_rgb_weight,
            self.dt_proj_rgb_bias,
            dt_min,
            dt_max,
            dt_scale,
            dt_init_floor,
        )
        self._init_dt_one(
            self.dt_proj_ir_weight,
            self.dt_proj_ir_bias,
            dt_min,
            dt_max,
            dt_scale,
            dt_init_floor,
        )

    @staticmethod
    def _cross_scan(x):
        """
        BxCxHxW -> Bx4xCxL.
        0: left->right, 1: right->left, 2: top->bottom, 3: bottom->top.
        """
        x_lr = x.flatten(2)
        x_rl = torch.flip(x, dims=[3]).flatten(2)
        x_tb = x.transpose(2, 3).contiguous().flatten(2)
        x_bt = torch.flip(x, dims=[2]).transpose(2, 3).contiguous().flatten(2)
        return torch.stack([x_lr, x_rl, x_tb, x_bt], dim=1)

    @staticmethod
    def _cross_merge(ys, H, W):
        B, K, C, L = ys.shape
        assert K == 4

        y_lr = ys[:, 0].reshape(B, C, H, W)
        y_rl = torch.flip(ys[:, 1].reshape(B, C, H, W), dims=[3])

        y_tb = ys[:, 2].reshape(B, C, W, H).transpose(2, 3).contiguous()
        y_bt = ys[:, 3].reshape(B, C, W, H).transpose(2, 3).contiguous()
        y_bt = torch.flip(y_bt, dims=[2])

        return (y_lr + y_rl + y_tb + y_bt) * 0.25

    def _project_ssm_params(self, xs, x_proj_weight, dt_proj_weight, dt_proj_bias):
        """Generate modality-private dt, B and C for four matching scan directions."""
        B, K, D, L = xs.shape
        x_dbl = torch.einsum("bkdl,kcd->bkcl", xs, x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2,
        )
        dts = torch.einsum("bkrl,kdr->bkdl", dts, dt_proj_weight)
        dts = dts + dt_proj_bias.view(1, K, D, 1)
        return dts, Bs.contiguous(), Cs.contiguous()

    def _mix_cross_c(self, c_rgb, c_ir):
        # IMPORTANT: compute both from the original C tensors; never update in-place.
        if self.cross_c_mode == "independent":
            return c_rgb, c_ir

        if self.cross_c_mode == "shared":
            c_shared = c_rgb + c_ir
            return c_shared, c_shared

        alpha = torch.tanh(self.a_i2r)
        beta = torch.tanh(self.a_r2i)

        if self.cross_direction == "bidirectional":
            c_rgb_star = c_rgb + alpha * c_ir
            c_ir_star = c_ir + beta * c_rgb
        elif self.cross_direction == "ir2rgb":
            c_rgb_star = c_rgb + alpha * c_ir
            c_ir_star = c_ir
        else:  # rgb2ir
            c_rgb_star = c_rgb
            c_ir_star = c_ir + beta * c_rgb
        return c_rgb_star, c_ir_star

    def _selective_scan_one(self, xs, dts, Bs, Cs, A_logs, Ds):
        B, K, D, L = xs.shape
        xs_scan = xs.reshape(B, K * D, L)
        dts_scan = dts.reshape(B, K * D, L)
        As = -torch.exp(A_logs.float())
        Ds = Ds.float()

        y = selective_scan_fn(
            xs_scan.float(),
            dts_scan.float(),
            As,
            Bs.float(),
            Cs.float(),
            Ds,
            z=None,
            delta_bias=None,
            delta_softplus=True,
            return_last_state=False,
        )
        return y.reshape(B, K, D, L).to(xs.dtype)

    @staticmethod
    def _apply_out_norm(y, norm):
        B, C, H, W = y.shape
        y = y.flatten(2).transpose(1, 2).contiguous()
        y = norm(y)
        return y.transpose(1, 2).reshape(B, C, H, W).contiguous()

    def forward(self, rgb, ir):
        if rgb.shape != ir.shape:
            raise ValueError(f"CrossModalStateFusion expects equal RGB/IR shapes, got {rgb.shape} vs {ir.shape}")

        B, C, H, W = rgb.shape
        rgb_skip = rgb
        ir_skip = ir

        rgb_n = self.norm_rgb(rgb)
        ir_n = self.norm_ir(ir)

        u_rgb = self.act(self.dwconv_rgb(self.state_proj_rgb(rgb_n)))
        u_ir = self.act(self.dwconv_ir(self.state_proj_ir(ir_n)))
        g_rgb = self.act(self.gate_proj_rgb(rgb_n))
        g_ir = self.act(self.gate_proj_ir(ir_n))

        # Ultralytics builds model strides once with a CPU dummy tensor.
        # mamba_ssm selective_scan is CUDA-dependent in this project, so only
        # that build-time path bypasses state scanning while preserving shape.
        if not rgb.is_cuda:
            return rgb_skip, ir_skip

        xs_rgb = self._cross_scan(u_rgb)
        xs_ir = self._cross_scan(u_ir)

        dt_rgb, B_rgb, C_rgb = self._project_ssm_params(
            xs_rgb,
            self.x_proj_rgb_weight,
            self.dt_proj_rgb_weight,
            self.dt_proj_rgb_bias,
        )
        dt_ir, B_ir, C_ir = self._project_ssm_params(
            xs_ir,
            self.x_proj_ir_weight,
            self.dt_proj_ir_weight,
            self.dt_proj_ir_bias,
        )

        # The ONLY cross-modal state-space interaction.
        C_rgb_star, C_ir_star = self._mix_cross_c(C_rgb, C_ir)

        ys_rgb = self._selective_scan_one(
            xs_rgb, dt_rgb, B_rgb, C_rgb_star, self.A_logs_rgb, self.Ds_rgb
        )
        ys_ir = self._selective_scan_one(
            xs_ir, dt_ir, B_ir, C_ir_star, self.A_logs_ir, self.Ds_ir
        )

        y_rgb = self._cross_merge(ys_rgb, H, W)
        y_ir = self._cross_merge(ys_ir, H, W)

        y_rgb = self._apply_out_norm(y_rgb, self.out_norm_rgb)
        y_ir = self._apply_out_norm(y_ir, self.out_norm_ir)

        y_rgb = self.dropout_rgb(self.out_proj_rgb(y_rgb * g_rgb))
        y_ir = self.dropout_ir(self.out_proj_ir(y_ir * g_ir))

        return rgb_skip + y_rgb, ir_skip + y_ir


class FRCSFModule(nn.Module):
    """
    Frequency Restructuring and Cross-modal State Fusion Module.

    First-version/default path:
        1) fixed Haar DWT on RGB and IR separately;
        2) LL multi-scale intra-modal enhancement;
        3) MGFF-style multiplicative enhancement on LH/HL/HH separately;
        4) same-frequency factorized bilinear interaction:
           theta_L for LL, one shared theta_H reused for LH/HL/HH, rank=C/4;
        5) separate Haar IDWT for RGB and IR;
        6) private dual four-direction SS2D with Cross-C only;
        7) modality-private residual outputs;
        8) concat + 1x1 projection to one fused feature.

    YAML arguments (channel c is inferred by tasks.py and must NOT be written):
        [rank_ratio, d_state, ssm_ratio, dt_rank, d_conv, dropout,
         lf_mode, hf_mode, stage2_share_mode, stage2_direction,
         cross_c_mode, cross_direction, fusion_mode]

    Recommended first version:
        [4, 16, 1.0, "auto", 3, 0.0,
         "standard", "multiply", "low_high", "bidirectional",
         "cross", "bidirectional", "concat"]

    Main ablation switches:
        hf_mode="residual"                      -> residual HF enhancement
        rank_ratio=8 / 2                         -> C/8 / C/2
        stage2_share_mode="all_independent"     -> four independent Stage-2 modules
        stage2_share_mode="all_shared"          -> all bands share one Stage-2 module
        stage2_direction="rgb2ir" / "ir2rgb"   -> one-way Stage-2 interaction
        cross_c_mode="independent"              -> remove Cross-C
        cross_c_mode="shared"                   -> Shared-C
        cross_direction="rgb2ir" / "ir2rgb"    -> one-way Cross-C
        fusion_mode="add"                       -> Add + 3x3 projection
    """

    def __init__(
        self,
        c,
        rank_ratio=4,
        d_state=16,
        ssm_ratio=1.0,
        dt_rank="auto",
        d_conv=3,
        dropout=0.0,
        lf_mode="standard",
        hf_mode="multiply",
        stage2_share_mode="low_high",
        stage2_direction="bidirectional",
        cross_c_mode="cross",
        cross_direction="bidirectional",
        fusion_mode="concat",
    ):
        super().__init__()
        if fusion_mode not in {"concat", "add"}:
            raise ValueError(f"Unsupported FRCSF fusion_mode: {fusion_mode}")
        self.channels = c
        self.fusion_mode = fusion_mode

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # Stage 1: RGB and IR are strictly modality-private.
        self.stage1_rgb = IntraModalFrequencyEnhancement(c, lf_mode=lf_mode, hf_mode=hf_mode)
        self.stage1_ir = IntraModalFrequencyEnhancement(c, lf_mode=lf_mode, hf_mode=hf_mode)

        # Stage 2: only matching frequency bands interact.
        self.stage2 = SameFrequencyInteraction(
            c,
            rank_ratio=rank_ratio,
            share_mode=stage2_share_mode,
            interaction_direction=stage2_direction,
        )

        # Stage 3: private state evolution + Cross-C readout.
        self.stage3 = CrossModalStateFusion(
            c,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=dropout,
            cross_c_mode=cross_c_mode,
            cross_direction=cross_direction,
        )

        if fusion_mode == "concat":
            self.final_proj = nn.Conv2d(c * 2, c, 1, bias=True)
        else:
            # Ablation from the design: Add + 3x3 projection.
            self.final_proj = nn.Conv2d(c, c, 3, padding=1, bias=True)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("FRCSFModule expects [rgb_feature, ir_feature].")
        rgb, ir = x
        if rgb.shape != ir.shape:
            raise ValueError(f"FRCSFModule RGB/IR shape mismatch: {rgb.shape} vs {ir.shape}")

        # -------------------- DWT --------------------
        rgb_ll, rgb_lh, rgb_hl, rgb_hh, rgb_hw = self.dwt(rgb)
        ir_ll, ir_lh, ir_hl, ir_hh, ir_hw = self.dwt(ir)
        if rgb_hw != ir_hw:
            raise ValueError(f"FRCSFModule RGB/IR spatial mismatch: {rgb_hw} vs {ir_hw}")

        # -------------------- Stage 1 --------------------
        rgb_bands = self.stage1_rgb((rgb_ll, rgb_lh, rgb_hl, rgb_hh))
        ir_bands = self.stage1_ir((ir_ll, ir_lh, ir_hl, ir_hh))

        # -------------------- Stage 2 --------------------
        rgb_bands, ir_bands = self.stage2(rgb_bands, ir_bands)

        # -------------------- IDWT --------------------
        rgb_restructured = self.idwt(*rgb_bands, output_hw=rgb_hw)
        ir_restructured = self.idwt(*ir_bands, output_hw=ir_hw)

        # -------------------- Stage 3 --------------------
        rgb_state, ir_state = self.stage3(rgb_restructured, ir_restructured)

        # -------------------- Final fusion --------------------
        if self.fusion_mode == "concat":
            return self.final_proj(torch.cat([rgb_state, ir_state], dim=1))
        return self.final_proj(rgb_state + ir_state)


"""
Frequency-domain Mixture-of-Experts fusion for two-stream RGB/IR detection.

Two experimental variants are provided:
    1) FreqMoERouterSpecific
       - LL: independent router + independent expert bank
       - LH/HL/HH: independent routers + one shared high-frequency expert bank

    2) FreqMoEFullIndependent
       - LL/LH/HL/HH: four fully independent routers + expert banks

The module follows the agreed pipeline:
    RGB/IR features
      -> fixed Haar DWT
      -> modality-specific frequency enhancement (no BN)
      -> same-frequency RGB/IR MoE fusion
      -> fixed Haar IDWT
      -> one fused spatial feature

MoE auxiliary statistics are cached on the top-level fusion module and are
collected by the modified ultralytics/utils/loss.py during OBB training.
"""


# -----------------------------------------------------------------------------
# Fixed Haar wavelet transform
# -----------------------------------------------------------------------------

class HaarDWT2D(nn.Module):
    """Fixed orthonormal 2-D Haar DWT implemented by 2x2 slicing."""

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"HaarDWT2D expects [B,C,H,W], got {tuple(x.shape)}")

        h0, w0 = x.shape[-2:]
        pad_h = h0 & 1
        pad_w = w0 & 1
        if pad_h or pad_w:
            # Only right/bottom padding is required. Replicate is safe even for H/W=1.
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        # Orthonormal Haar basis; factor 1/2 comes from two 1/sqrt(2) filters.
        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5
        return (ll, lh, hl, hh), (h0, w0)


class HaarIDWT2D(nn.Module):
    """Exact inverse of HaarDWT2D."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        coeffs: Sequence[torch.Tensor],
        output_hw: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(coeffs) != 4:
            raise ValueError(f"HaarIDWT2D expects four subbands, got {len(coeffs)}")
        ll, lh, hl, hh = coeffs
        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError("LL/LH/HL/HH must have identical shapes for IDWT")

        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        bsz, ch, h, w = ll.shape
        out = ll.new_empty((bsz, ch, h * 2, w * 2))
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d

        if output_hw is not None:
            oh, ow = output_hw
            out = out[..., :oh, :ow]
        return out


# -----------------------------------------------------------------------------
# Stage 1: modality-specific frequency enhancement (no BatchNorm)
# -----------------------------------------------------------------------------

class LowFrequencyEnhance(nn.Module):
    """
    Lightweight MGFF-inspired low-frequency multi-granularity refinement.

    Instead of full C->C 3x3/5x5/7x7 convolutions with BN, use depthwise
    3x3/5x5/7x7 branches and learn only three scalar branch weights.
    A small residual scale keeps the LL path close to the original coefficient
    at the beginning of training.
    """

    def __init__(self, channels: int, gamma_init: float = 0.05):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False)
                for k in (3, 5, 7)
            ]
        )
        self.branch_logits = nn.Parameter(torch.zeros(3))
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.softmax(self.branch_logits, dim=0).to(dtype=x.dtype)
        delta = sum(a * branch(x) for a, branch in zip(alpha, self.branches))
        return x + self.gamma.to(dtype=x.dtype) * delta


class HighFrequencyGate(nn.Module):
    """
    MGFF-style multiplicative high-frequency selection.

    The gate is estimated from the magnitude |H| to avoid cancellation between
    positive and negative wavelet coefficients, while the gate is multiplied
    back onto the original signed coefficient H.

    No BN and no residual bypass are used:
        H_enhanced = A(H) * H
    """

    def __init__(self, gate_bias_init: float = 1.0):
        super().__init__()
        self.gate = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        # Start from a spatially uniform but not overly suppressive gate.
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, float(gate_bias_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.abs()
        avg_map = mag.mean(dim=1, keepdim=True)
        max_map = mag.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.gate(torch.cat([avg_map, max_map], dim=1)))
        return x * attn


class ModalityFrequencyEnhance(nn.Module):
    """One modality's LL/LH/HL/HH enhancement; all subband modules are independent."""

    def __init__(self, channels: int, ll_gamma_init: float = 0.05, hf_gate_bias_init: float = 1.0):
        super().__init__()
        self.ll = LowFrequencyEnhance(channels, gamma_init=ll_gamma_init)
        # Three directional high-frequency bands use independent tiny gates.
        self.lh = HighFrequencyGate(hf_gate_bias_init)
        self.hl = HighFrequencyGate(hf_gate_bias_init)
        self.hh = HighFrequencyGate(hf_gate_bias_init)

    def forward(self, coeffs: Sequence[torch.Tensor]):
        ll, lh, hl, hh = coeffs
        return self.ll(ll), self.lh(lh), self.hl(hl), self.hh(hh)


# -----------------------------------------------------------------------------
# Stage 2: FDA-inspired modality-aware router and lightweight Fusion Masters
# -----------------------------------------------------------------------------

class ModalityAwareRouter(nn.Module):
    """
    RGB/IR joint Top-K router.

    signed mode:
        descriptor = GAP(X) + GMP(X)

    magnitude mode:
        descriptor = GAP(|X|) + GMP(|X|)

    RGB/IR descriptors are kept separate before concatenation.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        init_std: float = 0.01,
        use_abs_descriptor: bool = False,
    ):
        super().__init__()

        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")

        if not (1 <= top_k <= num_experts):
            raise ValueError(
                f"top_k must satisfy 1 <= top_k <= num_experts, "
                f"got {top_k}/{num_experts}"
            )

        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.use_abs_descriptor = bool(use_abs_descriptor)

        self.fc = nn.Linear(
            channels * 2,
            self.num_experts,
            bias=True
        )

        # 接近均匀，但避免完全相同导致固定 tie
        nn.init.normal_(
            self.fc.weight,
            mean=0.0,
            std=float(init_std)
        )
        nn.init.zeros_(self.fc.bias)

    def _descriptor(self, x: torch.Tensor) -> torch.Tensor:
        # 高频实验使用幅值统计，避免 signed coefficient 正负抵消
        if self.use_abs_descriptor:
            x = x.abs()

        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)

        return avg + mx

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        if rgb.shape != ir.shape:
            raise ValueError(
                f"Router expects same RGB/IR shapes, "
                f"got {rgb.shape} vs {ir.shape}"
            )

        dr = self._descriptor(rgb)
        di = self._descriptor(ir)

        logits = self.fc(
            torch.cat([dr, di], dim=1)
        )

        # -----------------------------------------------------
        # Router 概率计算统一转 FP32
        # 避免 AMP 下 scatter FP16/FP32 dtype 冲突
        # -----------------------------------------------------
        routing_logits = logits.float()

        # 完整 N-expert 分布
        # 只用于 balance / debug
        probs = torch.softmax(
            routing_logits,
            dim=-1
        )

        # 真正用于融合：
        # Top-K -> selected Softmax
        top_values, top_indices = torch.topk(
            routing_logits,
            k=self.top_k,
            dim=-1
        )

        top_weights = torch.softmax(
            top_values,
            dim=-1
        )

        sparse_weights = torch.zeros_like(
            routing_logits
        )

        sparse_weights.scatter_(
            1,
            top_indices,
            top_weights
        )

        selected_mask = torch.zeros_like(
            routing_logits,
            dtype=torch.bool
        )

        selected_mask.scatter_(
            1,
            top_indices,
            True
        )

        return {
            "logits": routing_logits,
            "probs": probs,
            "weights": sparse_weights,
            "mask": selected_mask,
            "top_indices": top_indices,
            "top_weights": top_weights,
        }


class ExpertBranch(nn.Module):
    """No-BN bottleneck residual transform for one modality inside one Fusion Master."""

    def __init__(self, channels: int, ratio: float = 0.25, gamma_init: float = 0.10):
        super().__init__()
        hidden = max(int(round(channels * float(ratio))), 8)
        self.reduce = nn.Conv2d(channels, hidden, 1, bias=True)
        self.dw = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=True)
        self.expand = nn.Conv2d(hidden, channels, 1, bias=True)
        self.act = nn.SiLU(inplace=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.reduce(x)
        delta = self.act(delta)
        delta = self.dw(delta)
        delta = self.act(delta)
        delta = self.expand(delta)
        return x + self.gamma.to(dtype=x.dtype) * delta


class FusionMaster(nn.Module):
    """
    One homogeneous MoE expert with independent RGB/IR parameters.

    RGB and IR are processed separately first, and are added only afterwards:
        E_n(R, I) = P_n^RGB(R) + P_n^IR(I)

    This is the agreed modification relative to the original FDA sum-first path.
    """

    def __init__(self, channels: int, expert_ratio: float = 0.25, gamma_init: float = 0.10):
        super().__init__()
        self.rgb_branch = ExpertBranch(channels, expert_ratio, gamma_init)
        self.ir_branch = ExpertBranch(channels, expert_ratio, gamma_init)

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        return self.rgb_branch(rgb) + self.ir_branch(ir)


class FusionExpertBank(nn.Module):
    """A bank of homogeneous-architecture but independently parameterized Fusion Masters."""

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        expert_ratio: float = 0.25,
        expert_gamma_init: float = 0.10,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [
                FusionMaster(
                    channels=channels,
                    expert_ratio=expert_ratio,
                    gamma_init=expert_gamma_init,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        True sample-wise Top-K conditional execution.

        Only experts selected by the router are executed for the samples assigned
        to them. With the default N=4, K=2, each sample is processed by exactly
        two experts instead of all four. The selected expert outputs are weighted
        by the post-TopK Softmax weights and accumulated back to the original batch.
        """
        if top_indices.ndim != 2 or top_indices.shape[0] != rgb.shape[0]:
            raise ValueError(
                f"top_indices must be [B,K], got {tuple(top_indices.shape)} for B={rgb.shape[0]}"
            )
        if top_weights.shape != top_indices.shape:
            raise ValueError(
                f"top_weights shape must equal top_indices shape, got "
                f"{tuple(top_weights.shape)} vs {tuple(top_indices.shape)}"
            )

        out = torch.zeros_like(rgb)

        # Dispatch samples to experts. torch.topk returns unique expert indices
        # within each sample, so each (sample, expert) pair occurs at most once.
        for expert_id, expert in enumerate(self.experts):
            selected = top_indices.eq(expert_id)
            sample_idx, slot_idx = torch.where(selected)
            if sample_idx.numel() == 0:
                continue

            rgb_sub = rgb.index_select(0, sample_idx)
            ir_sub = ir.index_select(0, sample_idx)
            expert_out = expert(rgb_sub, ir_sub)

            w = top_weights[sample_idx, slot_idx].to(
                device=expert_out.device,
                dtype=expert_out.dtype
            )

            weighted = expert_out * w.view(-1, 1, 1, 1)

            # One sample is selected by K experts, so contributions from different
            # experts are summed into the corresponding batch position.
            out = out.index_add(0, sample_idx, weighted)

        return out


# -----------------------------------------------------------------------------
# MoE balance/debug helpers
# -----------------------------------------------------------------------------

def _balance_loss_from_routing(
    probs: torch.Tensor,
    selected_mask: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    """Switch-style balancing term adapted to Top-K routing."""
    n = probs.shape[-1]
    p = probs.mean(dim=0)  # sums to 1
    # selection_rate sums to K; divide by K so f sums to 1.
    f = selected_mask.float().mean(dim=0) / float(top_k)
    return float(n) * torch.sum(f * p)


def _routing_debug(
    route_state: Dict[str, torch.Tensor],
    bank: str,
    route: str,
    balance_loss: torch.Tensor,
    feature_hw: Tuple[int, int],
):
    probs = route_state["probs"]
    mask = route_state["mask"]
    weights = route_state["weights"]
    logits = route_state["logits"]
    top_indices = route_state["top_indices"]
    # -----------------------------------------------------
    # 1. Expert selection rate
    # Top-2 下四个 expert 的和约为 2
    # -----------------------------------------------------
    selection_rate = mask.float().mean(dim=0)
    selected_count = mask.float().sum(dim=0)
    selected_weight_mean = (
        weights.sum(dim=0)
        / selected_count.clamp_min(1.0)
    )
    # -----------------------------------------------------
    # 2. Router entropy
    # 归一化到 [0,1]
    # -----------------------------------------------------
    entropy = -(
        probs.clamp_min(1e-9)
        * probs.clamp_min(1e-9).log()
    ).sum(dim=-1)
    if probs.shape[-1] > 1:
        entropy = entropy / math.log(
            probs.shape[-1]
        )
    entropy = entropy.mean()
    # -----------------------------------------------------
    # 3. probability batch std
    #
    # 用来区分：所有样本都0.25 和：不同样本有不同偏好，只是batch mean恰好0.25
    # -----------------------------------------------------
    prob_std = probs.std(
        dim=0,
        unbiased=False
    )
    # -----------------------------------------------------
    # 4. Top2 - Top3 logit margin
    # margin≈0:
    # Top-K 选择没有明显置信度
    # -----------------------------------------------------
    if logits.shape[-1] >= 3:
        sorted_logits = torch.sort(
            logits,
            dim=-1,
            descending=True
        ).values

        top2_top3_margin = (
            sorted_logits[:, 1]
            - sorted_logits[:, 2]
        ).mean()
    else:
        top2_top3_margin = logits.new_tensor(0.0)

    # -----------------------------------------------------
    # 5. Top-K pair histogram
    # N=4,K=2:
    # 01 / 02 / 03 / 12 / 13 / 23
    # -----------------------------------------------------
    pair_freq = {}
    if top_indices.shape[1] == 2:
        pair_idx = torch.sort(
            top_indices,
            dim=-1
        ).values
        for i in range(probs.shape[-1]):
            for j in range(i + 1, probs.shape[-1]):
                freq = (
                    (pair_idx[:, 0] == i)
                    & (pair_idx[:, 1] == j)
                ).float().mean()

                pair_freq[f"pair_{i}{j}"] = freq.detach()

    result = {
        "bank": bank,
        "route": route,

        "feature_h": int(feature_hw[0]),
        "feature_w": int(feature_hw[1]),

        "prob_mean": probs.mean(dim=0).detach(),
        "prob_std": prob_std.detach(),

        "selection_rate": selection_rate.detach(),

        "selected_weight_mean":
            selected_weight_mean.detach(),

        "entropy": entropy.detach(),

        "top2_top3_margin":
            top2_top3_margin.detach(),

        "balance_loss":
            balance_loss.detach(),
    }

    result.update(pair_freq)

    return result


class _FrequencyMoEBase(nn.Module):
    """Shared DWT/enhancement/cache logic for the two experimental variants."""

    band_names = ("ll", "lh", "hl", "hh")

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_ratio: float = 0.25,
        ll_gamma_init: float = 0.05,
        hf_gate_bias_init: float = 1.0,
        expert_gamma_init: float = 0.10,
        router_init_std: float = 0.01,
    ):
        super().__init__()
        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.expert_ratio = float(expert_ratio)
        self.expert_gamma_init = float(expert_gamma_init)
        self.router_init_std = float(router_init_std)

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # RGB and IR enhancement paths do not share parameters.
        self.rgb_enhance = ModalityFrequencyEnhance(
            channels, ll_gamma_init=ll_gamma_init, hf_gate_bias_init=hf_gate_bias_init
        )
        self.ir_enhance = ModalityFrequencyEnhance(
            channels, ll_gamma_init=ll_gamma_init, hf_gate_bias_init=hf_gate_bias_init
        )

        # Interface consumed by v8OBBLoss.
        self.cache_moe_aux = False
        self.last_moe_balance_loss = None
        self.last_moe_debug_stats = None
        self.moe_variant = "base"

        self.last_moe_balance_terms = None
        # 用于低频率的 expert similarity 诊断
        self.expert_sim_interval = int(
            os.getenv("MOE_EXPERT_SIM_INTERVAL", "1000")
        )
        self._moe_forward_step = 0

    def _new_router(
        self,
        use_abs_descriptor: bool = False
    ) -> ModalityAwareRouter:

        return ModalityAwareRouter(
            self.channels,
            num_experts=self.num_experts,
            top_k=self.top_k,
            init_std=self.router_init_std,
            use_abs_descriptor=use_abs_descriptor,
        )

    def _new_bank(self) -> FusionExpertBank:
        return FusionExpertBank(
            self.channels,
            num_experts=self.num_experts,
            expert_ratio=self.expert_ratio,
            expert_gamma_init=self.expert_gamma_init,
        )

    @staticmethod
    def _split_inputs(x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("Frequency MoE fusion expects x=[rgb_feat, ir_feat]")
        rgb, ir = x
        if rgb.ndim != 4 or ir.ndim != 4:
            raise ValueError("RGB/IR features must be [B,C,H,W]")
        if rgb.shape != ir.shape:
            raise ValueError(f"RGB/IR feature shapes must match, got {rgb.shape} vs {ir.shape}")
        return rgb, ir

    def _decompose_enhance(self, rgb: torch.Tensor, ir: torch.Tensor):
        rgb_coeffs, rgb_hw = self.dwt(rgb)
        ir_coeffs, ir_hw = self.dwt(ir)
        if rgb_hw != ir_hw:
            raise ValueError("RGB/IR spatial shapes must match before DWT")
        rgb_coeffs = self.rgb_enhance(rgb_coeffs)
        ir_coeffs = self.ir_enhance(ir_coeffs)
        return rgb_coeffs, ir_coeffs, rgb_hw

    def _cache(
        self,
        balance_loss: torch.Tensor,
        debug_stats: List[Dict],
        balance_terms: Dict[str, torch.Tensor] = None,
    ):
        if self.cache_moe_aux:
            self.last_moe_balance_loss = balance_loss
            self.last_moe_debug_stats = debug_stats
            self.last_moe_balance_terms = balance_terms
        else:
            self.last_moe_balance_loss = None
            self.last_moe_debug_stats = None
            self.last_moe_balance_terms = None


class FreqMoERouterSpecific(_FrequencyMoEBase):
    """
    Experiment A: Router-specific / shared-HF-expert-bank.

    LL:
        Router_LL + Bank_LL

    High-frequency:
        Router_LH --\
        Router_HL ----> one shared Bank_HF
        Router_HH --/

    LH/HL/HH have independent routing weights; only the Fusion Master parameters
    are shared. The HF load-balancing term is computed jointly over all three
    high-frequency routing populations because they consume the same expert bank.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_ratio: float = 0.25,
        ll_gamma_init: float = 0.05,
        hf_gate_bias_init: float = 1.0,
        expert_gamma_init: float = 0.10,
        router_init_std: float = 0.01,
    ):
        super().__init__(
            channels,
            num_experts,
            top_k,
            expert_ratio,
            ll_gamma_init,
            hf_gate_bias_init,
            expert_gamma_init,
            router_init_std,
        )
        self.moe_variant = "router_specific"

        self.ll_router = self._new_router()
        self.ll_bank = self._new_bank()

        self.lh_router = self._new_router()
        self.hl_router = self._new_router()
        self.hh_router = self._new_router()
        self.hf_bank = self._new_bank()

    def forward(self, x):
        rgb, ir = self._split_inputs(x)
        rgb_bands, ir_bands, output_hw = self._decompose_enhance(rgb, ir)
        r_ll, r_lh, r_hl, r_hh = rgb_bands
        i_ll, i_lh, i_hl, i_hh = ir_bands

        # LL independent bank.
        s_ll = self.ll_router(r_ll, i_ll)
        f_ll = self.ll_bank(r_ll, i_ll, s_ll["top_indices"], s_ll["top_weights"])
        l_ll = _balance_loss_from_routing(s_ll["probs"], s_ll["mask"], self.top_k)

        # Three independent HF routers, one shared HF expert bank.
        s_lh = self.lh_router(r_lh, i_lh)
        s_hl = self.hl_router(r_hl, i_hl)
        s_hh = self.hh_router(r_hh, i_hh)

        f_lh = self.hf_bank(r_lh, i_lh, s_lh["top_indices"], s_lh["top_weights"])
        f_hl = self.hf_bank(r_hl, i_hl, s_hl["top_indices"], s_hl["top_weights"])
        f_hh = self.hf_bank(r_hh, i_hh, s_hh["top_indices"], s_hh["top_weights"])

        # One balance term for the shared HF bank: aggregate LH/HL/HH routes.
        hf_probs = torch.cat([s_lh["probs"], s_hl["probs"], s_hh["probs"]], dim=0)
        hf_mask = torch.cat([s_lh["mask"], s_hl["mask"], s_hh["mask"]], dim=0)
        l_hf = _balance_loss_from_routing(hf_probs, hf_mask, self.top_k)

        module_balance = 0.5 * (l_ll + l_hf)

        debug_stats = [
            _routing_debug(s_ll, bank="ll", route="ll", balance_loss=l_ll, feature_hw=r_ll.shape[-2:]),
            _routing_debug(s_lh, bank="hf", route="lh", balance_loss=l_hf, feature_hw=r_lh.shape[-2:]),
            _routing_debug(s_hl, bank="hf", route="hl", balance_loss=l_hf, feature_hw=r_hl.shape[-2:]),
            _routing_debug(s_hh, bank="hf", route="hh", balance_loss=l_hf, feature_hw=r_hh.shape[-2:]),
        ]
        self._cache(module_balance, debug_stats)

        return self.idwt((f_ll, f_lh, f_hl, f_hh), output_hw=output_hw)


class FreqMoEFullIndependent(_FrequencyMoEBase):
    """
    Experiment B: four fully independent MoEs.

    LL/LH/HL/HH each own:
        independent Router + independent Expert Bank

    The four bank-level balance losses are averaged, so this variant does not
    receive a larger auxiliary-loss coefficient merely because it has four banks.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_ratio: float = 0.25,
        ll_gamma_init: float = 0.05,
        hf_gate_bias_init: float = 1.0,
        expert_gamma_init: float = 0.10,
        router_init_std: float = 0.01,
    ):
        super().__init__(
            channels,
            num_experts,
            top_k,
            expert_ratio,
            ll_gamma_init,
            hf_gate_bias_init,
            expert_gamma_init,
            router_init_std,
        )
        self.moe_variant = "full_independent"

        # ---------------------------------------------------------
        # 是否使用 magnitude-aware HF Router
        # 0:
        #   LL: H
        #   HF: H
        # 1:
        #   LL: H
        #   HF: |H|
        # 注意 LL 永远保持原始 descriptor，不参与这个实验变量
        # ---------------------------------------------------------
        self.hf_router_abs = bool(
            int(os.getenv("MOE_HF_ROUTER_ABS", "0"))
        )

        self.routers = nn.ModuleDict({
            "ll": self._new_router(
                use_abs_descriptor=False
            ),

            "lh": self._new_router(
                use_abs_descriptor=self.hf_router_abs
            ),

            "hl": self._new_router(
                use_abs_descriptor=self.hf_router_abs
            ),

            "hh": self._new_router(
                use_abs_descriptor=self.hf_router_abs
            ),
        })

        self.banks = nn.ModuleDict({
            band: self._new_bank()
            for band in self.band_names
        })


    def forward(self, x):
        rgb, ir = self._split_inputs(x)

        rgb_bands, ir_bands, output_hw = \
            self._decompose_enhance(rgb, ir)

        fused_bands = []
        balance_terms = []
        balance_dict = {}
        debug_stats = []

        self._moe_forward_step += 1

        for band, rb, ib in zip(
            self.band_names,
            rgb_bands,
            ir_bands
        ):
            # -------------------------
            # Router
            # -------------------------
            state = self.routers[band](rb, ib)

            # -------------------------
            # Top-K Expert fusion
            # -------------------------
            fused = self.banks[band](
                rb,
                ib,
                state["top_indices"],
                state["top_weights"]
            )

            # -------------------------
            # 当前 band balance metric
            # 无论 gain 是否为0都计算
            # 方便 debug
            # -------------------------
            balance = _balance_loss_from_routing(
                state["probs"],
                state["mask"],
                self.top_k
            )

            fused_bands.append(fused)
            balance_terms.append(balance)

            balance_dict[band] = balance

            debug_item = _routing_debug(
                state,
                bank=band,
                route=band,
                balance_loss=balance,
                feature_hw=rb.shape[-2:]
            )

            debug_stats.append(debug_item)

        # 保留旧定义，仅用于兼容/debug
        module_balance = torch.stack(
            [v.reshape(()) for v in balance_terms]
        ).mean()

        # 关键：
        # loss.py 后面会分别读取 LL/LH/HL/HH
        self._cache(
            module_balance,
            debug_stats,
            balance_terms=balance_dict
        )

        return self.idwt(
            fused_bands,
            output_hw=output_hw
        )

# =============================================================================
# Frequency-MoE V2
# =============================================================================
#
# V2 keeps ALL V1 components before Stage 2 unchanged:
#   - Haar DWT / IDWT
#   - RGB / IR independent Stage-1 enhancement
#   - LL multi-scale residual refinement
#   - LH / HL / HH magnitude-based spatial gates
#   - Full-independent LL/LH/HL/HH routers
#   - Top-K sparse routing
#   - balance/debug cache interface consumed by v8OBBLoss
#
# The ONLY structural change is the Fusion Master:
#
# V1:
#   RGB -> private residual branch --\
#                                    + -> E_n
#   IR  -> private residual branch --/
#
#   E_n = R + I + gamma_R * delta_R + gamma_I * delta_I
#
# V2:
#   [RGB || IR]
#       -> joint PW 1x1
#       -> SiLU
#       -> joint DW 3x3
#       -> SiLU
#       -> joint PW 1x1
#       -> E_n
#
# No internal residual and no external/base residual are used in V2-0.
#
# All experts remain homogeneous in architecture and independent in parameters.
# =============================================================================


class JointFusionMasterV2(nn.Module):
    """
    V2 homogeneous joint RGB-IR Fusion Master.

    The two same-frequency modality features are concatenated BEFORE any
    learnable expert transformation, so each Master learns a true joint
    fusion function:

        E_n(R, I) = M_n([R || I])

    There is deliberately NO:
        - R + I identity term
        - per-modality residual branch
        - external projection residual

    This is the V2-0 baseline used to isolate the effect of:
        1) joint cross-modal expert modeling
        2) removing the strong common V1 residual term

    joint_hidden_ratio is defined relative to the single-modality channel C.
    Default ~= 1/3 is chosen to approximately match the parameter count of
    one V1 FusionMaster (two C -> C/4 -> C branches).

    Approximate dominant point-wise parameter counts:

        V1:
            2 * [C*(C/4) + (C/4)*C]
            ~= 1.0 * C^2

        V2:
            (2C)*H + H*C
            = 3*C*H

        Setting H ~= C/3 gives ~= 1.0 * C^2.

    This avoids confounding the V1->V2 comparison with a large capacity jump.
    """

    def __init__(
        self,
        channels: int,
        joint_hidden_ratio: float = 1.0 / 3.0,
    ):
        super().__init__()

        if joint_hidden_ratio <= 0:
            raise ValueError(
                f"joint_hidden_ratio must be positive, got {joint_hidden_ratio}"
            )

        self.channels = int(channels)
        self.joint_hidden_ratio = float(joint_hidden_ratio)

        hidden = max(
            int(round(self.channels * self.joint_hidden_ratio)),
            8
        )
        self.hidden_channels = hidden

        # [RGB || IR]: 2C -> H
        self.reduce = nn.Conv2d(
            self.channels * 2,
            hidden,
            kernel_size=1,
            bias=True,
        )

        # Local same-frequency joint spatial transform.
        self.dw = nn.Conv2d(
            hidden,
            hidden,
            kernel_size=3,
            padding=1,
            groups=hidden,
            bias=True,
        )

        # H -> C fused representation.
        self.expand = nn.Conv2d(
            hidden,
            self.channels,
            kernel_size=1,
            bias=True,
        )

        # Keep V1 ExpertBranch activation placement unchanged
        # so the V1->V2 ablation changes only modality organization/residual.
        self.act = nn.SiLU(inplace=True)

    def forward(
        self,
        joint: torch.Tensor,
    ) -> torch.Tensor:

        if joint.ndim != 4:
            raise ValueError(
                f"JointFusionMasterV2 expects [B,2C,H,W], got {tuple(joint.shape)}"
            )

        expected_c = self.channels * 2
        if joint.shape[1] != expected_c:
            raise ValueError(
                f"JointFusionMasterV2 expects {expected_c} input channels, "
                f"got {joint.shape[1]}"
            )

        out = self.reduce(joint)
        out = self.act(out)
        out = self.dw(out)
        out = self.act(out)
        out = self.expand(out)

        # IMPORTANT:
        # No residual is added here.
        return out


class JointFusionExpertBankV2(nn.Module):
    """
    V2 bank of homogeneous-architecture, independently-parameterized
    JointFusionMasterV2 experts.

    The RGB/IR concatenation is performed once before expert dispatch.
    Only sample-wise Top-K selected experts are executed.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        joint_hidden_ratio: float = 1.0 / 3.0,
    ):
        super().__init__()

        if num_experts < 1:
            raise ValueError(
                f"num_experts must be >= 1, got {num_experts}"
            )

        self.channels = int(channels)
        self.num_experts = int(num_experts)
        self.joint_hidden_ratio = float(joint_hidden_ratio)

        self.experts = nn.ModuleList(
            [
                JointFusionMasterV2(
                    channels=self.channels,
                    joint_hidden_ratio=self.joint_hidden_ratio,
                )
                for _ in range(self.num_experts)
            ]
        )

    def forward(
        self,
        rgb: torch.Tensor,
        ir: torch.Tensor,
        top_indices: torch.Tensor,
        top_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        True sample-wise Top-K conditional execution.

        Input:
            rgb, ir:
                [B,C,H,W]

            top_indices:
                [B,K]

            top_weights:
                [B,K], already normalized by post-TopK Softmax

        Output:
            fused:
                [B,C,H,W]
        """

        if rgb.shape != ir.shape:
            raise ValueError(
                f"JointFusionExpertBankV2 expects matching RGB/IR shapes, "
                f"got {tuple(rgb.shape)} vs {tuple(ir.shape)}"
            )

        if top_indices.ndim != 2 or top_indices.shape[0] != rgb.shape[0]:
            raise ValueError(
                f"top_indices must be [B,K], got {tuple(top_indices.shape)} "
                f"for B={rgb.shape[0]}"
            )

        if top_weights.shape != top_indices.shape:
            raise ValueError(
                f"top_weights shape must equal top_indices shape, got "
                f"{tuple(top_weights.shape)} vs {tuple(top_indices.shape)}"
            )

        # The core V2 change:
        # concatenate the two same-frequency modality features BEFORE
        # any learnable expert transformation.
        joint = torch.cat([rgb, ir], dim=1)

        # Every Master outputs C channels, so the MoE output shape is
        # identical to one modality feature and remains IDWT-compatible.
        out = torch.zeros_like(rgb)

        for expert_id, expert in enumerate(self.experts):
            selected = top_indices.eq(expert_id)
            sample_idx, slot_idx = torch.where(selected)

            if sample_idx.numel() == 0:
                continue

            joint_sub = joint.index_select(
                0,
                sample_idx
            )

            expert_out = expert(
                joint_sub
            )

            w = top_weights[
                sample_idx,
                slot_idx
            ].to(
                device=expert_out.device,
                dtype=expert_out.dtype,
            )

            weighted = (
                expert_out
                * w.view(-1, 1, 1, 1)
            )

            out = out.index_add(
                0,
                sample_idx,
                weighted
            )

        return out


class FreqMoEFullIndependentV2(_FrequencyMoEBase):
    """
    Frequency-MoE V2-0: Full-independent Joint Fusion Masters.

    Pipeline:
        RGB / IR feature
            -> fixed Haar DWT
            -> V1 Stage-1 modality-specific frequency enhancement
            -> for each of LL/LH/HL/HH independently:
                   V1 Router
                   + V2 JointFusionExpertBank
                   + Top-K weighted sum
            -> fixed Haar IDWT
            -> one fused spatial feature

    Full-independent organization is kept from the best V1 configuration:
        LL: Router_LL + Bank_LL
        LH: Router_LH + Bank_LH
        HL: Router_HL + Bank_HL
        HH: Router_HH + Bank_HH

    V2-0 intentionally keeps:
        - same Router implementation as V1
        - same optional MOE_HF_ROUTER_ABS switch
        - same balance/debug cache interface
        - same Top-K routing

    V2-0 intentionally removes:
        - V1 dual modality-specific ExpertBranches
        - V1 Expert internal residual
        - common R + I term
        - any external/base fusion residual
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        joint_hidden_ratio: float = 1.0 / 3.0,
        ll_gamma_init: float = 0.05,
        hf_gate_bias_init: float = 1.0,
        router_init_std: float = 0.01,
    ):
        # _FrequencyMoEBase owns the V1 DWT / Stage1 / cache logic.
        #
        # Its old expert_ratio / expert_gamma_init fields are irrelevant
        # for V2 banks, but harmless. Fixed values are passed only to reuse
        # the tested base implementation without modifying V1 code.
        super().__init__(
            channels=channels,
            num_experts=num_experts,
            top_k=top_k,
            expert_ratio=0.25,
            ll_gamma_init=ll_gamma_init,
            hf_gate_bias_init=hf_gate_bias_init,
            expert_gamma_init=0.10,
            router_init_std=router_init_std,
        )

        self.joint_hidden_ratio = float(
            joint_hidden_ratio
        )

        self.moe_variant = (
            "full_independent_v2_joint"
        )

        # Keep the V1 Router experimental switch unchanged.
        #
        # Default 0:
        #   LL descriptor: signed
        #   HF descriptor: signed
        #
        # Optional 1:
        #   LL descriptor: signed
        #   HF descriptor: magnitude
        #
        # IMPORTANT:
        # Stage-1 HighFrequencyGate already uses |H| in V1 and remains
        # unchanged. This environment variable affects only Stage-2 Router.
        self.hf_router_abs = bool(
            int(
                os.getenv(
                    "MOE_HF_ROUTER_ABS",
                    "0"
                )
            )
        )

        # Four independent V1 routers.
        self.routers = nn.ModuleDict(
            {
                "ll": self._new_router(
                    use_abs_descriptor=False
                ),
                "lh": self._new_router(
                    use_abs_descriptor=self.hf_router_abs
                ),
                "hl": self._new_router(
                    use_abs_descriptor=self.hf_router_abs
                ),
                "hh": self._new_router(
                    use_abs_descriptor=self.hf_router_abs
                ),
            }
        )

        # Four fully independent V2 joint expert banks.
        self.banks = nn.ModuleDict(
            {
                band: JointFusionExpertBankV2(
                    channels=self.channels,
                    num_experts=self.num_experts,
                    joint_hidden_ratio=self.joint_hidden_ratio,
                )
                for band in self.band_names
            }
        )

    def forward(self, x):
        rgb, ir = self._split_inputs(x)

        # EXACTLY reuse V1 Stage-1.
        rgb_bands, ir_bands, output_hw = \
            self._decompose_enhance(
                rgb,
                ir
            )

        fused_bands = []
        balance_terms = []
        balance_dict = {}
        debug_stats = []

        self._moe_forward_step += 1

        for band, rb, ib in zip(
            self.band_names,
            rgb_bands,
            ir_bands,
        ):
            # -------------------------------------------------
            # Router:
            # Keep V1 implementation and inputs unchanged.
            #
            # Conceptually equivalent to pooling [RGB || IR],
            # because pooling is channel-wise before the FC.
            # -------------------------------------------------
            state = self.routers[band](
                rb,
                ib
            )

            # -------------------------------------------------
            # V2 Joint Top-K Master fusion.
            #
            # The Bank concatenates RGB/IR before the first
            # learnable expert transform.
            # -------------------------------------------------
            fused = self.banks[band](
                rb,
                ib,
                state["top_indices"],
                state["top_weights"],
            )

            # -------------------------------------------------
            # Balance metric:
            # Keep exactly the V1 definition so loss.py can use
            # LL and HF gains independently.
            # -------------------------------------------------
            balance = _balance_loss_from_routing(
                state["probs"],
                state["mask"],
                self.top_k,
            )

            fused_bands.append(
                fused
            )
            balance_terms.append(
                balance
            )
            balance_dict[band] = (
                balance
            )

            debug_item = _routing_debug(
                state,
                bank=band,
                route=band,
                balance_loss=balance,
                feature_hw=rb.shape[-2:],
            )

            debug_stats.append(
                debug_item
            )

        # Kept only for legacy/debug compatibility.
        module_balance = torch.stack(
            [
                value.reshape(())
                for value in balance_terms
            ]
        ).mean()

        # Same cache contract as V1:
        # loss.py needs no V2-specific branch.
        self._cache(
            module_balance,
            debug_stats,
            balance_terms=balance_dict,
        )

        # No external residual.
        # The four MoEs have already generated complete fused subbands.
        return self.idwt(
            fused_bands,
            output_hw=output_hw,
        )

class FreqMoEFullIndependentV2Residual(_FrequencyMoEBase):
    """
    V2-R: Full-independent Joint Fusion MoE with an external joint residual.

    For each frequency band b:

        X_b = Concat(RGB_b, IR_b)                       # [B, 2C, H, W]
        F_base_b = P_b(X_b)                             # 1x1 Conv: 2C -> C
        F_moe_b  = sum_{n in TopK} w_n M_n(X_b)        # Joint V2 Masters
        F_b      = F_base_b + gamma_b * F_moe_b

    Important:
        1. The residual base comes from Concat(RGB, IR), NOT RGB + IR.
        2. P_b is independent for LL/LH/HL/HH.
        3. gamma_b is a learnable scalar, also independent per band.
        4. No residual exists inside JointFusionMasterV2.
        5. Router / Top-K / balance / debug remain identical to V2-0.
        6. Stage1 remains exactly the V1/V2-0 implementation.

    Default residual_gamma_init = 0.5:
        larger than V1's 0.1 expert-internal correction, because here the MoE
        branch is intended to make a meaningful contribution rather than act
        as a tiny perturbation around a dominant base path.

    YAML args:
        [num_experts, top_k, joint_hidden_ratio, residual_gamma_init]

    parse_model() still prepends channels automatically:
        args = [c, *args]
    """

    def __init__(
        self,
        channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        joint_hidden_ratio: float = 1.0 / 3.0,
        residual_gamma_init: float = 0.5,
        ll_gamma_init: float = 0.05,
        hf_gate_bias_init: float = 1.0,
        router_init_std: float = 0.01,
    ):
        super().__init__(
            channels=channels,
            num_experts=num_experts,
            top_k=top_k,
            expert_ratio=0.25,          # unused by V2 Joint Banks
            ll_gamma_init=ll_gamma_init,
            hf_gate_bias_init=hf_gate_bias_init,
            expert_gamma_init=0.10,    # unused by V2 Joint Banks
            router_init_std=router_init_std,
        )

        if residual_gamma_init < 0:
            raise ValueError(
                f"residual_gamma_init must be >= 0, got {residual_gamma_init}"
            )

        self.joint_hidden_ratio = float(joint_hidden_ratio)
        self.residual_gamma_init = float(residual_gamma_init)

        self.moe_variant = "full_independent_v2_joint_residual"

        # Keep V2-0 / V1 router switch unchanged.
        self.hf_router_abs = bool(
            int(os.getenv("MOE_HF_ROUTER_ABS", "0"))
        )

        # ---------------------------------------------------------
        # Four independent routers: unchanged from V2-0.
        # ---------------------------------------------------------
        self.routers = nn.ModuleDict(
            {
                "ll": self._new_router(use_abs_descriptor=False),
                "lh": self._new_router(use_abs_descriptor=self.hf_router_abs),
                "hl": self._new_router(use_abs_descriptor=self.hf_router_abs),
                "hh": self._new_router(use_abs_descriptor=self.hf_router_abs),
            }
        )

        # ---------------------------------------------------------
        # Four independent V2 Joint Expert Banks.
        # ---------------------------------------------------------
        self.banks = nn.ModuleDict(
            {
                band: JointFusionExpertBankV2(
                    channels=self.channels,
                    num_experts=self.num_experts,
                    joint_hidden_ratio=self.joint_hidden_ratio,
                )
                for band in self.band_names
            }
        )

        # ---------------------------------------------------------
        # External joint residual base:
        #
        #   [RGB_b || IR_b] : 2C
        #            ↓
        #        Conv 1x1
        #            ↓
        #             C
        #
        # Each frequency band owns an independent projection.
        # No activation is used: this path is kept as a simple learnable
        # linear joint-fusion shortcut.
        # ---------------------------------------------------------
        self.base_proj = nn.ModuleDict(
            {
                band: nn.Conv2d(
                    2 * self.channels,
                    self.channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True,
                )
                for band in self.band_names
            }
        )

        # ---------------------------------------------------------
        # Learnable external MoE scale.
        #
        # One scalar per band rather than one global scalar because the
        # useful MoE correction strength can differ between LL and the
        # three directional high-frequency bands.
        #
        # Unconstrained parameter:
        #   gamma can adapt freely during training.
        # Initial value is exposed in YAML.
        # ---------------------------------------------------------
        self.residual_gamma = nn.ParameterDict(
            {
                band: nn.Parameter(
                    torch.tensor(
                        self.residual_gamma_init,
                        dtype=torch.float32,
                    )
                )
                for band in self.band_names
            }
        )

    def forward(self, x):
        rgb, ir = self._split_inputs(x)

        # Exactly reuse V1/V2-0 DWT + Stage1 enhancement.
        rgb_bands, ir_bands, output_hw = self._decompose_enhance(
            rgb,
            ir,
        )

        fused_bands = []
        balance_terms = []
        balance_dict = {}
        debug_stats = []

        self._moe_forward_step += 1

        for band, rb, ib in zip(
            self.band_names,
            rgb_bands,
            ir_bands,
        ):
            # -----------------------------------------------------
            # Router: unchanged.
            # -----------------------------------------------------
            state = self.routers[band](
                rb,
                ib,
            )

            # -----------------------------------------------------
            # V2 Joint MoE branch.
            # JointFusionExpertBankV2 internally concatenates
            # [rb || ib] before the first learnable expert layer.
            # -----------------------------------------------------
            moe_out = self.banks[band](
                rb,
                ib,
                state["top_indices"],
                state["top_weights"],
            )

            # -----------------------------------------------------
            # External joint residual base.
            # IMPORTANT: concat-derived projection, not rb + ib.
            # -----------------------------------------------------
            joint = torch.cat(
                [rb, ib],
                dim=1,
            )

            base = self.base_proj[band](
                joint
            )

            gamma = self.residual_gamma[band].to(
                device=moe_out.device,
                dtype=moe_out.dtype,
            )

            # V2-R:
            # complete projection base + learnable-scaled Joint-MoE branch.
            fused = base + gamma * moe_out

            # -----------------------------------------------------
            # Same balance definition as V1 / V2-0.
            # -----------------------------------------------------
            balance = _balance_loss_from_routing(
                state["probs"],
                state["mask"],
                self.top_k,
            )

            fused_bands.append(
                fused
            )
            balance_terms.append(
                balance
            )
            balance_dict[band] = (
                balance
            )

            debug_item = _routing_debug(
                state,
                bank=band,
                route=band,
                balance_loss=balance,
                feature_hw=rb.shape[-2:],
            )

            debug_stats.append(
                debug_item
            )

        module_balance = torch.stack(
            [
                value.reshape(())
                for value in balance_terms
            ]
        ).mean()

        # Same cache contract as V1/V2-0:
        # current loss.py needs no modification.
        self._cache(
            module_balance,
            debug_stats,
            balance_terms=balance_dict,
        )

        return self.idwt(
            fused_bands,
            output_hw=output_hw,
        )



# =============================================================================
# Fixed Haar DWT / IDWT
# =============================================================================

class HaarDWT2D(nn.Module):
    """Fixed orthonormal 2-D Haar DWT by 2x2 slicing."""

    def forward(self, x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"HaarDWT2D expects [B,C,H,W], got {tuple(x.shape)}")

        h0, w0 = x.shape[-2:]
        pad_h = h0 & 1
        pad_w = w0 & 1
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        a = x[..., 0::2, 0::2]
        b = x[..., 0::2, 1::2]
        c = x[..., 1::2, 0::2]
        d = x[..., 1::2, 1::2]

        ll = (a + b + c + d) * 0.5
        lh = (-a - b + c + d) * 0.5
        hl = (-a + b - c + d) * 0.5
        hh = (a - b - c + d) * 0.5

        return (ll, lh, hl, hh), (h0, w0)


class HaarIDWT2D(nn.Module):
    """Exact inverse of HaarDWT2D."""

    def forward(
        self,
        coeffs: Sequence[torch.Tensor],
        output_hw: Tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if len(coeffs) != 4:
            raise ValueError(f"HaarIDWT2D expects 4 bands, got {len(coeffs)}")

        ll, lh, hl, hh = coeffs
        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError("LL/LH/HL/HH must have identical shapes")

        a = (ll - lh - hl + hh) * 0.5
        b = (ll - lh + hl - hh) * 0.5
        c = (ll + lh - hl - hh) * 0.5
        d = (ll + lh + hl + hh) * 0.5

        B, C, H, W = ll.shape
        out = ll.new_empty(B, C, H * 2, W * 2)
        out[..., 0::2, 0::2] = a
        out[..., 0::2, 1::2] = b
        out[..., 1::2, 0::2] = c
        out[..., 1::2, 1::2] = d

        if output_hw is not None:
            h0, w0 = output_hw
            out = out[..., :h0, :w0]
        return out


# =============================================================================
# LL local multi-scale enhancement
# =============================================================================

class MultiScaleLowFreqEnhance(nn.Module):
    """
    Modality-specific LL enhancement.

        x
        |-- DWConv 3x3 --|
        |-- DWConv 5x5 --| -> concat -> PWConv 1x1 -> residual add
        |-- DWConv 7x7 --|

    No BN. No cross-modal interaction occurs here.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dw3 = nn.Conv2d(
            channels, channels, 3, 1, 1, groups=channels, bias=True
        )
        self.dw5 = nn.Conv2d(
            channels, channels, 5, 1, 2, groups=channels, bias=True
        )
        self.dw7 = nn.Conv2d(
            channels, channels, 7, 1, 3, groups=channels, bias=True
        )
        self.pw = nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.pw(torch.cat([self.dw3(x), self.dw5(x), self.dw7(x)], dim=1))
        return x + delta


# =============================================================================
# Coupled SSM core: paper Eq.(6)
# =============================================================================

class _ModalitySSMParams(nn.Module):
    """
    Generate modality-specific B_t, C_t and Delta_t from the current sequence.

    Input : [B,L,E]
    B,C   : [B,L,N]
    Delta : [B,L,E]

    A is input-independent and modality-specific:
        A = -exp(A_log)
    with A_log initialized to zero. The paper does not specify A initialization;
    the negative exponential follows the stable continuous-time Mamba convention.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        dt_init_floor: float = 1e-4,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        self.B_proj = nn.Linear(d_inner, d_state, bias=False)
        self.C_proj = nn.Linear(d_inner, d_state, bias=False)
        self.delta_proj = nn.Linear(d_inner, d_inner, bias=False)

        # Input-independent Delta bias, as in Algorithm 1.
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.delta_bias = nn.Parameter(inv_dt)

        # Stable negative continuous A. Zero log-init => A starts at -1.
        self.A_log = nn.Parameter(torch.zeros(d_inner, d_state))
        self.A_log._no_weight_decay = True

    def forward(self, u: torch.Tensor):
        B_t = self.B_proj(u)
        C_t = self.C_proj(u)
        delta = F.softplus(self.delta_proj(u) + self.delta_bias)
        A = -torch.exp(self.A_log.float()).to(dtype=u.dtype, device=u.device)
        return A, B_t, C_t, delta


class _AffineCoupledScanFn(torch.autograd.Function):
    """
    Exact parallel implementation of the two-modal Eq.(6) recurrence.

    This is NOT the ordinary Mamba selective_scan_cuda kernel.
    It exploits the algebraic reduction:

        s_t = h_R,t + h_I,t
            = P_t * s_{t-1} + U_t

        P_t = S_R,t + S_I,t
        U_t = Bbar_R,t * x_R,t + Bbar_I,t * x_I,t

    and evaluates the affine recurrence with an associative prefix scan
    in O(log L) tensor stages instead of an O(L) Python loop.

    A custom backward is supplied so autograd does not retain every
    prefix-scan stage. Backward is also reduced to a reverse affine scan.

    Inputs
    ------
    x_rgb/x_ir : [B,L,E]
    S_*        : [B,L,E,N]
    Bbar_*     : [B,L,E,N]
    C_*        : [B,L,N]

    Outputs
    -------
    y_rgb/y_ir : [B,L,E]
    """

    @staticmethod
    def _affine_prefix(P: torch.Tensor, U: torch.Tensor):
        """
        Inclusive prefix of affine maps T_t(s)=P_t*s+U_t.

        Composition:
            (P_b,U_b) o (P_a,U_a)
              = (P_b*P_a, U_b + P_b*U_a)

        The operator is associative, so Hillis-Steele style doubling is exact
        up to floating-point reassociation.
        """
        if P.shape != U.shape:
            raise ValueError("P and U must have identical shapes")

        L = P.shape[1]
        pref_P = P
        pref_U = U
        offset = 1

        while offset < L:
            # Use values from the PREVIOUS stage on both sides.
            tail_P = pref_P[:, offset:]
            left_P = pref_P[:, :-offset]
            tail_U = pref_U[:, offset:]
            left_U = pref_U[:, :-offset]

            new_tail_P = tail_P * left_P
            new_tail_U = tail_U + tail_P * left_U

            pref_P = torch.cat((pref_P[:, :offset], new_tail_P), dim=1)
            pref_U = torch.cat((pref_U[:, :offset], new_tail_U), dim=1)
            offset <<= 1

        return pref_P, pref_U

    @staticmethod
    def forward(
        ctx,
        x_rgb,
        x_ir,
        S_rgb,
        S_ir,
        Bbar_rgb,
        Bbar_ir,
        C_rgb,
        C_ir,
    ):
        # Forward of custom autograd.Function runs without graph construction.
        P = S_rgb + S_ir
        U_rgb = Bbar_rgb * x_rgb.unsqueeze(-1)
        U_ir = Bbar_ir * x_ir.unsqueeze(-1)
        U = U_rgb + U_ir

        _, shared_all = _AffineCoupledScanFn._affine_prefix(P, U)

        # Eq.(6) needs the PREVIOUS shared state at each position.
        zero = torch.zeros_like(shared_all[:, :1])
        shared_prev = torch.cat((zero, shared_all[:, :-1]), dim=1)

        h_rgb = S_rgb * shared_prev + U_rgb
        h_ir = S_ir * shared_prev + U_ir

        y_rgb = torch.einsum("blen,bln->ble", h_rgb, C_rgb)
        y_ir = torch.einsum("blen,bln->ble", h_ir, C_ir)

        # One shared-state tensor is enough to reconstruct h in backward.
        ctx.save_for_backward(
            x_rgb,
            x_ir,
            S_rgb,
            S_ir,
            Bbar_rgb,
            Bbar_ir,
            C_rgb,
            C_ir,
            shared_prev,
        )
        return y_rgb, y_ir

    @staticmethod
    def backward(ctx, grad_y_rgb, grad_y_ir):
        (
            x_rgb,
            x_ir,
            S_rgb,
            S_ir,
            Bbar_rgb,
            Bbar_ir,
            C_rgb,
            C_ir,
            shared_prev,
        ) = ctx.saved_tensors

        # ============================================================
        # AMP-safe backward:
        # custom Function backward 不依赖 autocast，
        # 因此内部统一使用 FP32，避免 Float/Half 混合。
        # ============================================================
        gy_rgb = grad_y_rgb.float()
        gy_ir = grad_y_ir.float()

        xr = x_rgb.float()
        xi = x_ir.float()

        Sr = S_rgb.float()
        Si = S_ir.float()

        Br = Bbar_rgb.float()
        Bi = Bbar_ir.float()

        Cr = C_rgb.float()
        Ci = C_ir.float()

        shared = shared_prev.float()

        # ------------------------------------------------------------
        # reconstruct current states
        # ------------------------------------------------------------
        U_rgb = Br * xr.unsqueeze(-1)
        U_ir = Bi * xi.unsqueeze(-1)

        h_rgb = Sr * shared + U_rgb
        h_ir = Si * shared + U_ir

        # ------------------------------------------------------------
        # direct gradient from:
        # y_t = C_t h_t
        # ------------------------------------------------------------
        direct_rgb = gy_rgb.unsqueeze(-1) * Cr.unsqueeze(2)
        direct_ir = gy_ir.unsqueeze(-1) * Ci.unsqueeze(2)

        # ------------------------------------------------------------
        # reverse coupled recurrence
        #
        # lambda_{t-1}
        # = (S_R,t + S_I,t) * lambda_t
        #   + direct_R,t * S_R,t
        #   + direct_I,t * S_I,t
        # ------------------------------------------------------------
        P = Sr + Si
        Q = direct_rgb * Sr + direct_ir * Si

        B, L, E, N = P.shape

        lambda_all = torch.zeros_like(P)

        if L > 1:
            P_rev = torch.flip(P[:, 1:], dims=[1])
            Q_rev = torch.flip(Q[:, 1:], dims=[1])

            _, lambda_rev = _AffineCoupledScanFn._affine_prefix(
                P_rev,
                Q_rev,
            )

            lambda_all = torch.cat(
                (
                    torch.flip(lambda_rev, dims=[1]),
                    torch.zeros_like(P[:, :1]),
                ),
                dim=1,
            )

        grad_h_rgb = direct_rgb + lambda_all
        grad_h_ir = direct_ir + lambda_all

        # ------------------------------------------------------------
        # gradients of
        # h_m = S_m * shared_prev + Bbar_m * x_m
        # ------------------------------------------------------------
        grad_S_rgb = grad_h_rgb * shared
        grad_S_ir = grad_h_ir * shared

        grad_Bbar_rgb = grad_h_rgb * xr.unsqueeze(-1)
        grad_Bbar_ir = grad_h_ir * xi.unsqueeze(-1)

        grad_x_rgb = torch.sum(
            grad_h_rgb * Br,
            dim=-1,
        )

        grad_x_ir = torch.sum(
            grad_h_ir * Bi,
            dim=-1,
        )

        # ------------------------------------------------------------
        # gradients of C
        # y_m = sum_n C_m[n] * h_m[n]
        # ------------------------------------------------------------
        grad_C_rgb = torch.einsum(
            "ble,blen->bln",
            gy_rgb,
            h_rgb,
        )

        grad_C_ir = torch.einsum(
            "ble,blen->bln",
            gy_ir,
            h_ir,
        )

        # ============================================================
        # IMPORTANT:
        # return gradients in the dtype of their corresponding inputs.
        # ============================================================
        return (
            grad_x_rgb.to(x_rgb.dtype),
            grad_x_ir.to(x_ir.dtype),

            grad_S_rgb.to(S_rgb.dtype),
            grad_S_ir.to(S_ir.dtype),

            grad_Bbar_rgb.to(Bbar_rgb.dtype),
            grad_Bbar_ir.to(Bbar_ir.dtype),

            grad_C_rgb.to(C_rgb.dtype),
            grad_C_ir.to(C_ir.dtype),
        )


class CoupledSelectiveScan1D(nn.Module):
    """
    Accelerated exact implementation of paper Eq.(6) for two modalities.

    Mathematical model is unchanged from the reference implementation.
    Only the execution algorithm is changed:

        reference: O(L) Python recurrent loop
        fast     : O(log L) associative affine prefix scan

    transition_mode:
      - "algorithm1" (V1 default): follows Algorithm 1 lines 13-14
      - "zoh": follows paper Eq.(2), diagonal-A specialization
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        transition_mode: str = "algorithm1",
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        if transition_mode not in {"algorithm1", "zoh"}:
            raise ValueError(
                f"transition_mode must be 'algorithm1' or 'zoh', got {transition_mode}"
            )

        self.d_inner = d_inner
        self.d_state = d_state
        self.transition_mode = transition_mode

        self.rgb_params = _ModalitySSMParams(
            d_inner, d_state, dt_min=dt_min, dt_max=dt_max
        )
        self.ir_params = _ModalitySSMParams(
            d_inner, d_state, dt_min=dt_min, dt_max=dt_max
        )

    @staticmethod
    def _transition_algorithm1(
        A: torch.Tensor,
        B_t: torch.Tensor,
        delta_t: torch.Tensor,
    ):
        # Vectorized over the whole sequence:
        # A       : [E,N]
        # B_t     : [B,L,N]
        # delta_t : [B,L,E]
        S = delta_t.unsqueeze(-1) * A.view(1, 1, *A.shape)
        Bbar = delta_t.unsqueeze(-1) * B_t.unsqueeze(2)
        return S, Bbar

    @staticmethod
    def _transition_zoh(
        A: torch.Tensor,
        B_t: torch.Tensor,
        delta_t: torch.Tensor,
    ):
        delta_A = delta_t.unsqueeze(-1) * A.view(1, 1, *A.shape)
        S = torch.exp(delta_A)

        gain = torch.expm1(delta_A) / A.view(1, 1, *A.shape)
        Bbar = gain * B_t.unsqueeze(2)
        return S, Bbar

    def _transition(self, A, B_t, delta_t):
        if self.transition_mode == "algorithm1":
            return self._transition_algorithm1(A, B_t, delta_t)
        return self._transition_zoh(A, B_t, delta_t)

    def forward(
        self,
        x_rgb: torch.Tensor,
        x_ir: torch.Tensor,
        h0_rgb: torch.Tensor | None = None,
        h0_ir: torch.Tensor | None = None,
        return_states: bool = False,
    ):
        if return_states:
            raise NotImplementedError(
                "Fast training path does not materialize all hidden states. "
                "Use the slow reference implementation only for debugging states."
            )
        if h0_rgb is not None or h0_ir is not None:
            raise NotImplementedError(
                "V1 visual training always uses zero initial states; "
                "non-zero h0 is not implemented in the fast path."
            )
        if x_rgb.ndim != 3 or x_ir.ndim != 3:
            raise ValueError(
                f"CoupledSelectiveScan1D expects [B,L,E], got "
                f"{tuple(x_rgb.shape)} and {tuple(x_ir.shape)}"
            )
        if x_rgb.shape != x_ir.shape:
            raise ValueError("RGB and IR sequences must have the same shape")
        if x_rgb.shape[-1] != self.d_inner:
            raise ValueError(
                f"Expected inner dim {self.d_inner}, got {x_rgb.shape[-1]}"
            )

        A_rgb, B_rgb, C_rgb, dt_rgb = self.rgb_params(x_rgb)
        A_ir, B_ir, C_ir, dt_ir = self.ir_params(x_ir)

        S_rgb, Bbar_rgb = self._transition(A_rgb, B_rgb, dt_rgb)
        S_ir, Bbar_ir = self._transition(A_ir, B_ir, dt_ir)

        return _AffineCoupledScanFn.apply(
            x_rgb,
            x_ir,
            S_rgb,
            S_ir,
            Bbar_rgb,
            Bbar_ir,
            C_rgb,
            C_ir,
        )

# =============================================================================
# Full-map four-direction Cross Scan / Merge
# =============================================================================

class CrossScan2D:
    """
    Convert [B,E,H,W] into four full-map sequences [B,L,E], L=H*W.

    0: row-major forward
    1: row-major backward
    2: column-major forward
    3: column-major backward
    """

    @staticmethod
    def apply(x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim != 4:
            raise ValueError(f"CrossScan2D expects [B,E,H,W], got {tuple(x.shape)}")

        row = x.flatten(2).transpose(1, 2).contiguous()
        row_rev = torch.flip(row, dims=[1])

        col = (
            x.transpose(2, 3)
            .contiguous()
            .flatten(2)
            .transpose(1, 2)
            .contiguous()
        )
        col_rev = torch.flip(col, dims=[1])

        return [row, row_rev, col, col_rev]


class CrossMerge2D:
    """Inverse geometry of CrossScan2D, then sum the four directional outputs."""

    @staticmethod
    def apply(
        ys: Sequence[torch.Tensor],
        H: int,
        W: int,
        reduce: str = "sum",
    ) -> torch.Tensor:
        if len(ys) != 4:
            raise ValueError("CrossMerge2D expects exactly four directional sequences")
        if reduce not in {"sum", "mean"}:
            raise ValueError("reduce must be 'sum' or 'mean'")

        y0, y1, y2, y3 = ys
        B, L, E = y0.shape
        if L != H * W:
            raise ValueError(f"Sequence length {L} != H*W={H*W}")

        row_f = y0.view(B, H, W, E)
        row_b = torch.flip(y1, dims=[1]).view(B, H, W, E)

        col_f = (
            y2.view(B, W, H, E)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        col_b = (
            torch.flip(y3, dims=[1])
            .view(B, W, H, E)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        out = row_f + row_b + col_f + col_b
        if reduce == "mean":
            out = out * 0.25
        return out


class CoupledSS2D(nn.Module):
    """
    Four-direction visual extension of CoupledSelectiveScan1D.

    Each scan direction has its own modality-specific A/B/C/Delta parameters,
    analogous to direction-specific SS2D projections.
    """

    def __init__(
        self,
        d_inner: int,
        d_state: int = 16,
        transition_mode: str = "algorithm1",
    ):
        super().__init__()
        self.d_inner = d_inner
        self.scans = nn.ModuleList(
            [
                CoupledSelectiveScan1D(
                    d_inner=d_inner,
                    d_state=d_state,
                    transition_mode=transition_mode,
                )
                for _ in range(4)
            ]
        )

    def forward(self, x_rgb: torch.Tensor, x_ir: torch.Tensor):
        if x_rgb.shape != x_ir.shape:
            raise ValueError("RGB and IR 2D features must have identical shapes")
        if x_rgb.ndim != 4:
            raise ValueError("CoupledSS2D expects NCHW inputs")
        if x_rgb.shape[1] != self.d_inner:
            raise ValueError(
                f"Expected {self.d_inner} channels, got {x_rgb.shape[1]}"
            )

        _, _, H, W = x_rgb.shape
        seq_rgb = CrossScan2D.apply(x_rgb)
        seq_ir = CrossScan2D.apply(x_ir)

        y_rgb_dirs = []
        y_ir_dirs = []

        for k in range(4):
            yr, yi = self.scans[k](seq_rgb[k], seq_ir[k])
            y_rgb_dirs.append(yr)
            y_ir_dirs.append(yi)

        # Standard cross-merge uses directional summation. A subsequent
        # LayerNorm in the VSS outer block controls scale.
        y_rgb = CrossMerge2D.apply(y_rgb_dirs, H, W, reduce="sum")
        y_ir = CrossMerge2D.apply(y_ir_dirs, H, W, reduce="sum")

        # Return BHWC, matching LayerNorm/Linear convention.
        return y_rgb, y_ir


# =============================================================================
# Dual VSS outer block
# =============================================================================

class CoupledVSSBlock2D(nn.Module):
    """
    Two modality-specific VSS shells sharing one Coupled SS2D interaction core.

    For each modality:
        input
          -> LayerNorm
          -> Linear -> split [u, z]
          -> u: DWConv2D + SiLU
          -> CoupledSS2D(u_rgb, u_ir)
          -> LayerNorm
          -> multiply SiLU(z)
          -> Linear
          -> own residual

    Finally returns TWO coupled modality representations.
    Fusion into one LL band is performed outside this block by simple average.
    """

    def __init__(
        self,
        channels: int,
        d_state: int = 16,
        expand: float = 2.0,
        d_conv: int = 3,
        transition_mode: str = "algorithm1",
    ):
        super().__init__()
        if d_conv % 2 == 0:
            raise ValueError("For 2D same-size DWConv, d_conv must be odd")

        inner = int(channels * expand)
        self.channels = channels
        self.inner = inner

        # Structure is symmetric; parameters are NOT shared across modalities.
        self.norm_in_rgb = nn.LayerNorm(channels)
        self.norm_in_ir = nn.LayerNorm(channels)

        self.in_proj_rgb = nn.Linear(channels, inner * 2, bias=False)
        self.in_proj_ir = nn.Linear(channels, inner * 2, bias=False)

        pad = d_conv // 2
        self.dw_rgb = nn.Conv2d(
            inner, inner, d_conv, 1, pad, groups=inner, bias=True
        )
        self.dw_ir = nn.Conv2d(
            inner, inner, d_conv, 1, pad, groups=inner, bias=True
        )

        self.act = nn.SiLU()

        self.coupled_ss2d = CoupledSS2D(
            d_inner=inner,
            d_state=d_state,
            transition_mode=transition_mode,
        )

        self.norm_out_rgb = nn.LayerNorm(inner)
        self.norm_out_ir = nn.LayerNorm(inner)

        self.out_proj_rgb = nn.Linear(inner, channels, bias=False)
        self.out_proj_ir = nn.Linear(inner, channels, bias=False)

    @staticmethod
    def _nchw_to_bhwc(x):
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _bhwc_to_nchw(x):
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor):
        if rgb.shape != ir.shape:
            raise ValueError("RGB and IR LL features must have the same shape")
        if rgb.ndim != 4:
            raise ValueError("CoupledVSSBlock2D expects NCHW inputs")

        residual_rgb = rgb
        residual_ir = ir

        rgb_h = self.norm_in_rgb(self._nchw_to_bhwc(rgb))
        ir_h = self.norm_in_ir(self._nchw_to_bhwc(ir))

        rgb_u, rgb_z = self.in_proj_rgb(rgb_h).chunk(2, dim=-1)
        ir_u, ir_z = self.in_proj_ir(ir_h).chunk(2, dim=-1)

        rgb_u = self._bhwc_to_nchw(rgb_u)
        ir_u = self._bhwc_to_nchw(ir_u)

        rgb_u = self.act(self.dw_rgb(rgb_u))
        ir_u = self.act(self.dw_ir(ir_u))

        y_rgb, y_ir = self.coupled_ss2d(rgb_u, ir_u)  # BHWC

        y_rgb = self.norm_out_rgb(y_rgb) * F.silu(rgb_z, inplace=False)
        y_ir = self.norm_out_ir(y_ir) * F.silu(ir_z, inplace=False)

        y_rgb = self.out_proj_rgb(y_rgb)
        y_ir = self.out_proj_ir(y_ir)

        out_rgb = residual_rgb + self._bhwc_to_nchw(y_rgb)
        out_ir = residual_ir + self._bhwc_to_nchw(y_ir)

        return out_rgb, out_ir


# =============================================================================
# High-frequency local-energy + signed-consistency fusion
# =============================================================================

class HighFreqLocalEnergyConsistencyFusion(nn.Module):
    """
    Apply independently to one corresponding RGB/IR high-frequency subband.

    1) Local 3x3 energy:
        E_m = mean_N(H_m^2)

    2) Energy-based soft candidate:
        w_R = E_R / (E_R + E_I + eps)
        w_I = E_I / (E_R + E_I + eps)
        H_soft = w_R H_R + w_I H_I

    3) Dominant signed candidate:
        H_dom = H_R if E_R >= E_I else H_I

    4) Signed local consistency:
        C = 2 mean_N(H_R H_I) /
            (mean_N(H_R^2) + mean_N(H_I^2) + eps)

    5) Continuous consistency coefficient:
        eta = clamp(C, 0, 1)

    6) Final:
        H_F = eta H_soft + (1-eta) H_dom

    Original signed coefficients are preserved throughout.
    There is NO magnitude reconstruction and NO sign recovery.
    """

    def __init__(self, kernel_size: int = 3, eps: float = 1e-6):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.eps = eps

    def _local_mean(self, x: torch.Tensor) -> torch.Tensor:
        # count_include_pad=False exactly implements valid-neighbor averaging
        # at edges/corners without injecting zero-padding into the denominator.
        return F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            count_include_pad=False,
        )

    def forward(self, h_rgb: torch.Tensor, h_ir: torch.Tensor):
        if h_rgb.shape != h_ir.shape:
            raise ValueError("Corresponding RGB/IR HF bands must have the same shape")

        e_rgb = self._local_mean(h_rgb.square())
        e_ir = self._local_mean(h_ir.square())

        denom_e = e_rgb + e_ir + self.eps
        w_rgb = e_rgb / denom_e
        w_ir = e_ir / denom_e

        h_soft = w_rgb * h_rgb + w_ir * h_ir
        h_dom = torch.where(e_rgb >= e_ir, h_rgb, h_ir)

        cross = self._local_mean(h_rgb * h_ir)
        consistency = (2.0 * cross) / (e_rgb + e_ir + self.eps)
        consistency = consistency.clamp(-1.0, 1.0)

        eta = consistency.clamp(0.0, 1.0)
        h_fused = eta * h_soft + (1.0 - eta) * h_dom

        return h_fused


# =============================================================================
# Top-level frequency fusion module
# =============================================================================

class FreqCoupledWaveletFusion(nn.Module):
    """
    Input:
        [rgb_feat, ir_feat], both [B,C,H,W]

    Output:
        fused feature [B,C,H,W]

    Pipeline:
        RGB/IR
          -> Haar DWT
          -> LL:
               independent multi-scale enhancement
               -> Coupled VSS
               -> average the two coupled outputs
          -> LH/HL/HH:
               local energy + signed consistency fusion
          -> Haar IDWT
    """

    def __init__(
        self,
        c: int,
        d_state: int = 16,
        expand: float = 2.0,
        d_conv: int = 3,
        hf_kernel: int = 3,
        transition_mode: str = "algorithm1",
    ):
        super().__init__()
        self.c = c

        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        self.ll_enh_rgb = MultiScaleLowFreqEnhance(c)
        self.ll_enh_ir = MultiScaleLowFreqEnhance(c)

        self.ll_coupled = CoupledVSSBlock2D(
            channels=c,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            transition_mode=transition_mode,
        )

        self.hf_fusion = HighFreqLocalEnergyConsistencyFusion(
            kernel_size=hf_kernel
        )

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError(
                "FreqCoupledWaveletFusion expects [rgb_feat, ir_feat]"
            )

        rgb, ir = x
        if rgb.shape != ir.shape:
            raise ValueError(
                f"RGB/IR shape mismatch: {tuple(rgb.shape)} vs {tuple(ir.shape)}"
            )
        if rgb.ndim != 4:
            raise ValueError("Expected NCHW feature maps")
        if rgb.shape[1] != self.c:
            raise ValueError(f"Configured c={self.c}, got input C={rgb.shape[1]}")

        (r_ll, r_lh, r_hl, r_hh), output_hw = self.dwt(rgb)
        (i_ll, i_lh, i_hl, i_hh), _ = self.dwt(ir)

        # LL: local intra-modal structure -> cross-modal coupled long-range state.
        r_ll = self.ll_enh_rgb(r_ll)
        i_ll = self.ll_enh_ir(i_ll)

        r_ll_out, i_ll_out = self.ll_coupled(r_ll, i_ll)

        # V1 keeps the terminal fusion deliberately non-learned.
        f_ll = 0.5 * (r_ll_out + i_ll_out)

        # HF: each corresponding subband is processed independently.
        f_lh = self.hf_fusion(r_lh, i_lh)
        f_hl = self.hf_fusion(r_hl, i_hl)
        f_hh = self.hf_fusion(r_hh, i_hh)

        return self.idwt(
            (f_ll, f_lh, f_hl, f_hh),
            output_hw=output_hw,
        )
