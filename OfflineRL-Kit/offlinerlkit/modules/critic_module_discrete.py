import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union, Optional


class CriticDiscrete(nn.Module):
    def __init__(self, backbone: nn.Module, device: str = "cpu") -> None:
        super().__init__()

        self.device = torch.device(device)
        self.backbone = backbone.to(device)
        latent_dim = getattr(backbone, "output_dim")
        self.last = nn.Linear(latent_dim, 10).to(device)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        actions: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> torch.Tensor:
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        if actions is not None:
            # actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32).flatten(1)
            actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
            # multiply ground truth a with the action value
            action_value = torch.tensor([[-0.9], [-0.7], [-0.5], [-0.3], [-0.1], [0.1], [0.3], [0.5], [0.7], [0.9]], device=self.device)
            action_value = torch.unsqueeze(action_value, dim=0)
            new_action = torch.squeeze(torch.matmul(actions, action_value))
            obs = torch.cat([obs, new_action], dim=1)
        logits = self.backbone(obs)
        values = self.last(logits)
        return values