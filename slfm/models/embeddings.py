import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchvision.transforms as transforms
import torchaudio
import math


class FloatEmbeddingSine(nn.Module):
    """
    This is a simple version of the float embedding which convert a float number to a high dimension vector.
    It is adopted from PositionEmbeddingSine from paper DETR.
    """
    def __init__(self, num_pos_feats=512, temperature=10000, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        if scale is None:
            self.scale = 2 * math.pi
        else:
            self.scale = scale

    def forward(self, x):
        '''
            x: (N, K)
        '''
        # import pdb; pdb.set_trace()
        x = x * self.scale # we use 1 for scale since it's convert to 2 pi already
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        x = x / dim_t
        x = torch.stack([x[:, 0::2].sin(), x[:, 0::2].cos()], dim=-1).flatten(-2)
        return x


if __name__ == '__main__':
    net = FloatEmbeddingSine(num_pos_feats=512, temperature=10000, scale=None)
    x = torch.rand(10, 1)
    x = net(x)
