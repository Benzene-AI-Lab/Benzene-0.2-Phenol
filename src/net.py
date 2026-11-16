#!/usr/bin/env python3
"""
net.py —— 策略/价值双头 ResNet

结构（AlphaZero 风格）：
    输入  19 × 8 × 8
    主干  若干残差块，每个 = Conv3×3 + BN + ReLU + Conv3×3 + BN + 残差
    头 1  policy：Conv1×1 → FC → 4,672 维 logits
    头 2  value ：Conv1×1 → FC → FC → tanh → 1 维（[-1, 1]）

参数规模（实测）：
    6 块 × 128 通道，policy 16 通道    ≈ 6.9M    ← Alkane
    12 块 × 160 通道                   ≈ 15.4M   ← Alkene
    16 块 × 192 通道                   ≈ 20.5M   ← Phenol

注意：policy 头直连 4,672 维，占了参数的大头。
     policy 通道 32 → 9.6M（总 11.6M）
     policy 通道 16 → 4.8M（总 6.9M）  ← 默认
     通道再压低会削弱走法分布的表达能力

与 0.1 的对比：0.1 是自回归 Transformer，这里是 CNN。
原因：棋盘的局部结构（三格连线、马腿、兵的斜吃）是平移不变的，
     卷积先验比自注意力更匹配；同时 CNN 在 CPU 上快得多。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_PLANES = 19
NUM_MOVES = 4672


class ResidualBlock(nn.Module):
    __slots__ = ()

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    def __init__(self, blocks: int = 6, channels: int = 128,
                 policy_channels: int = 16, value_channels: int = 32):
        super().__init__()
        self.blocks = blocks
        self.channels = channels

        # 入口
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 主干
        self.trunk = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])

        # policy 头
        self.policy_conv = nn.Sequential(
            nn.Conv2d(channels, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(inplace=True),
        )
        self.policy_fc = nn.Linear(policy_channels * 64, NUM_MOVES)

        # value 头
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_channels * 64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, 19, 8, 8) float32 → (policy_logits (B,4672), value (B,1))"""
        h = self.trunk(self.stem(x))

        p = self.policy_conv(h).flatten(1)
        policy = self.policy_fc(p)

        v = self.value_conv(h).flatten(1)
        value = self.value_fc(v)

        return policy, value

    # ---- 便捷方法 ----
    def predict(self, planes: torch.Tensor) -> tuple[torch.Tensor, float]:
        """单局面推理。planes: (19,8,8) → (policy_probs (4672,), value float)"""
        self.eval()
        with torch.inference_mode():
            x = planes.unsqueeze(0)
            logits, v = self(x)
            probs = F.softmax(logits, dim=1)[0]
        return probs, float(v.item())

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build(scale: str = "alkane") -> ChessNet:
    if scale == "local":
        # 本机 CPU 训练用的精简版：3.34M，769 ms/步
        return ChessNet(blocks=4, channels=96, policy_channels=8)
    if scale == "alkane":
        return ChessNet(blocks=6, channels=128)
    if scale == "alkene":
        return ChessNet(blocks=12, channels=160)
    if scale == "phenol":
        return ChessNet(blocks=16, channels=192)
    raise ValueError(f"未知规模: {scale}")


if __name__ == "__main__":
    import time

    print("=" * 64)
    print("net.py 自检")
    print("=" * 64)

    for name in ("alkane", "alkene", "phenol"):
        m = build(name)
        n = m.count_params()
        m.eval()
        x = torch.randn(1, NUM_PLANES, 8, 8)
        with torch.inference_mode():
            p, v = m(x)
        assert p.shape == (1, NUM_MOVES), p.shape
        assert v.shape == (1, 1), v.shape
        assert -1.0 <= v.item() <= 1.0
        print(f"  {name:8s}  {n/1e6:6.2f} M  params   policy{p.shape[1]}  value ok")

    # 延迟测试
    print()
    m = build("alkane").eval()
    for bs in (1, 8, 32):
        x = torch.randn(bs, NUM_PLANES, 8, 8)
        with torch.inference_mode():
            for _ in range(3):
                m(x)
        t = time.time()
        N = 10
        with torch.inference_mode():
            for _ in range(N):
                m(x)
        dt = (time.time() - t) / N
        print(f"  batch {bs:3d}: {dt*1000:7.1f} ms   → {bs/dt:8.1f} 局面/秒")

    print("=" * 64)
    print("通过")
