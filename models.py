from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", bs=64, n_steps=17, output_dim=256):
        super().__init__()
        self.device = device
        self.bs = bs
        self.n_steps = n_steps
        self.repr_dim = 256

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        return torch.randn((self.bs, self.n_steps, self.repr_dim)).to(self.device)


class SimpleCNNEncoder(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Linear(128*9*9, output_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.flatten(start_dim=1)
        s = self.fc(h)
        return s

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=512, output_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, s):
        return self.net(s)

class Predictor(nn.Module):
    def __init__(self, state_dim=256, action_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=action_dim, hidden_size=state_dim, batch_first=True)

    def forward(self, s_0, actions):
        B = s_0.size(0)
        h0 = s_0.unsqueeze(0) # [1, B, D]
        c0 = torch.zeros_like(h0) # [1, B, D]
        outputs, (hT, cT) = self.lstm(actions, (h0, c0))
        return outputs  # [B, T-1, state_dim]


class JEPAWithVICReg(nn.Module):
    def __init__(self, state_dim=256, action_dim=2, proj_dim=256):
        super().__init__()
        self.enc = SimpleCNNEncoder(output_dim=state_dim)
        self.enc_y = SimpleCNNEncoder(output_dim=state_dim)
        self.predictor = Predictor(state_dim=state_dim, action_dim=action_dim)
        self.proj_x = ProjectionHead(input_dim=state_dim, output_dim=proj_dim)
        self.proj_y = ProjectionHead(input_dim=state_dim, output_dim=proj_dim)
        self.repr_dim = state_dim  # Needed by evaluator

    def forward(self, states, actions, future_states=None):
        # states might be [B,1,2,64,64] at evaluation time. Squeeze the time dimension if present.
        if states.ndim == 5:
            # [B,1,2,64,64] -> [B,2,64,64]
            states = states.squeeze(1)

        # Encode initial state
        s_0 = self.enc(states)  # [B, state_dim]

        # Predict future states embeddings given s_0 and actions
        pred_s_y_seq = self.predictor(s_0, actions)  # [B, T-1, state_dim]

        # Concatenate s_0 as the first step
        pred_encs = torch.cat([s_0.unsqueeze(1), pred_s_y_seq], dim=1)  # [B,T,state_dim]

        if future_states is not None:
            # Training mode (future_states provided)
            B, T_minus_1 = actions.shape[:2]
            future_states_flat = future_states.view(B*(T_minus_1), 2, 65, 65)
            s_y_flat = self.enc_y(future_states_flat)  # [B*(T-1), state_dim]
            s_y_seq = s_y_flat.view(B, T_minus_1, -1)  # [B, T-1, state_dim]
            v_x = self.proj_x(s_0)                     # [B, proj_dim]
            v_y_true = self.proj_y(s_y_flat)           # [B*(T-1), proj_dim]
            return pred_encs, s_0, s_y_seq, v_x, v_y_true
        else:
            # Evaluation mode (no future_states)
            # Just return the predicted embeddings
            return pred_encs

class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output
