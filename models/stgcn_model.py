"""
stgcn_model.py
──────────────
Spatio-Temporal Graph Convolutional Network (STGCN) for traffic prediction.

Architecture:
  Input (batch, T, N, F) → ST-Conv Block ×2 → Output Layer → (batch, N, F_out)

Where:
  T = time window (e.g., 12 steps = 1 hour at 5-min intervals)
  N = number of road segments (nodes in the graph)
  F = input features per node (speed, vehicle_count, occupancy, congestion_ratio)
  F_out = predicted features for T+1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv(nn.Module):
    """1D causal convolution along the time axis."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(1, kernel_size),
            padding=(0, kernel_size // 2),
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (batch, channels, N, T)
        return F.relu(self.bn(self.conv(x)))


class GraphConv(nn.Module):
    """
    Graph convolution using the normalized adjacency matrix.
    Operates on spatial dimension (N nodes).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x:   (batch, C, N, T)
        adj: (N, N) normalized adjacency
        """
        # Reshape for matmul: (batch, T, N, C)
        batch, C, N, T = x.shape
        x_perm = x.permute(0, 3, 2, 1)  # (batch, T, N, C)

        # Graph convolution: A @ X @ W
        support = torch.matmul(x_perm, self.weight)  # (batch, T, N, out)
        output = torch.matmul(adj, support) + self.bias  # (batch, T, N, out)

        # Back to (batch, out, N, T)
        return F.relu(output.permute(0, 3, 2, 1))


class STConvBlock(nn.Module):
    """
    Spatio-Temporal Convolution Block:
      Temporal Conv → Graph Conv → Temporal Conv → ReLU + Dropout
    """

    def __init__(self, in_channels, spatial_channels, out_channels,
                 num_nodes, kernel_size=3):
        super().__init__()
        self.temporal1 = TemporalConv(in_channels, spatial_channels, kernel_size)
        self.graph_conv = GraphConv(spatial_channels, spatial_channels)
        self.temporal2 = TemporalConv(spatial_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(num_nodes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, adj):
        """x: (batch, C, N, T), adj: (N, N)"""
        t1 = self.temporal1(x)
        gc = self.graph_conv(t1, adj)
        t2 = self.temporal2(gc)
        # BatchNorm on node dimension
        out = self.bn(t2.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        return self.dropout(out)


class STGCN(nn.Module):
    """
    Spatio-Temporal Graph Convolutional Network.

    Takes historical traffic states across the road graph over time window T
    and predicts the traffic state at T+1.
    """

    def __init__(self, num_nodes, in_features, out_features,
                 time_steps=12, hidden_channels=32):
        super().__init__()
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.out_features = out_features

        # Two ST-Conv blocks
        self.block1 = STConvBlock(
            in_channels=in_features,
            spatial_channels=hidden_channels,
            out_channels=hidden_channels,
            num_nodes=num_nodes,
        )
        self.block2 = STConvBlock(
            in_channels=hidden_channels,
            spatial_channels=hidden_channels,
            out_channels=hidden_channels,
            num_nodes=num_nodes,
        )

        # Final temporal conv to collapse time dimension
        self.final_temporal = nn.Conv2d(
            hidden_channels, hidden_channels,
            kernel_size=(1, time_steps),
            padding=0,
        )

        # Output projection
        self.fc = nn.Linear(hidden_channels, out_features)

    def forward(self, x, adj):
        """
        Args:
            x:   (batch, T, N, F) — historical traffic states
            adj: (N, N) — normalized adjacency matrix

        Returns:
            (batch, N, F_out) — predicted next-step traffic state
        """
        # Reshape to (batch, F, N, T) for conv layers
        x = x.permute(0, 3, 2, 1)  # (batch, F, N, T)

        # ST-Conv blocks
        h = self.block1(x, adj)
        h = self.block2(h, adj)

        # Collapse time dimension
        # Pad if needed to match expected time dimension
        if h.shape[3] < self.final_temporal.kernel_size[1]:
            pad_size = self.final_temporal.kernel_size[1] - h.shape[3]
            h = F.pad(h, (0, pad_size))

        h = F.relu(self.final_temporal(h))  # (batch, hidden, N, 1)
        h = h.squeeze(3)  # (batch, hidden, N)
        h = h.permute(0, 2, 1)  # (batch, N, hidden)

        # Project to output features
        out = self.fc(h)  # (batch, N, F_out)
        return out


if __name__ == "__main__":
    # Quick sanity check
    N, F_in, F_out, T = 24, 4, 4, 12
    batch = 8

    model = STGCN(num_nodes=N, in_features=F_in, out_features=F_out,
                  time_steps=T, hidden_channels=32)

    x = torch.randn(batch, T, N, F_in)
    adj = torch.eye(N) + torch.randn(N, N).abs() * 0.1
    adj = (adj + adj.T) / 2  # symmetric

    out = model(x, adj)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Expected: ({batch}, {N}, {F_out})")
    assert out.shape == (batch, N, F_out), "Shape mismatch!"
    print("✓ STGCN model forward pass OK")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
