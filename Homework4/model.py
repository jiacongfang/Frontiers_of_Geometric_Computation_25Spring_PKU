import torch
import torch.nn as nn
import numpy as np


class GaussianFourierFeatureTransform(nn.Module):
    """
    An implementation of Gaussian Fourier feature transform.

    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
        https://arxiv.org/abs/2006.10739
        https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html

    Modified from:
        https://github.com/ndahlquist/pytorch-fourier-feature-networks
        https://github.com/tancik/fourier-feature-networks?tab=readme-ov-file

    Input: a batch of cloud points with shape [Batches, 3]
    Output: a batch of Fourier features with shape [Batches, 2*mapping_size]
    """

    def __init__(self, num_input_channels=3, mapping_size=64, scale=5):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        # Random Gaussian sampled projection matrix
        self._B = torch.randn((num_input_channels, mapping_size)) * scale
        self.output_dim = 2 * mapping_size

    def forward(self, x):
        """
        x: Point cloud coordinates of shape [B, 3]
        returns: Fourier features of shape [B, 2*mapping_size]
        """
        assert x.dim() == 2, "Expected 2D input (got {}D input)".format(x.dim())

        _, channels = x.shape

        assert channels == self._num_input_channels, (
            "Expected input to have {} channels (got {} channels)".format(
                self._num_input_channels, channels
            )
        )

        # Project points to higher dimension: [B, 3] @ [3, M] -> [B, M]
        x = x @ self._B.to(x.device)

        # Apply sinusoidal activation: [B, M] -> [B, 2M]
        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


class RecNet(nn.Module):
    """
    Followed the architecture of IGR [ICML 2021]
    https://github.com/amosgropp/IGR/blob/master/code/model/network.py#L59
    """

    def __init__(
        self,
        in_dim=3,
        num_hidden_layers=8,
        hidden_dim=512,
        out_dim=1,
        skip_in=[4],
        geometric_init=True,
        radius_init=1.0,
        fourier_transform=None,
    ):
        super(RecNet, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.fourier_transform = fourier_transform

        self.dims = [in_dim] + [hidden_dim] * num_hidden_layers + [out_dim]
        self.num_layers = len(self.dims) - 1
        self.skip_in = skip_in

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            if i + 1 in self.skip_in:
                layer = nn.Linear(self.dims[i], self.dims[i + 1] - self.in_dim)
            else:
                layer = nn.Linear(self.dims[i], self.dims[i + 1])

            if geometric_init:
                if i == self.num_layers - 1:
                    torch.nn.init.normal_(
                        layer.weight,
                        mean=np.sqrt(np.pi) / np.sqrt(self.dims[i]),
                        std=0.00001,
                    )
                    torch.nn.init.constant_(layer.bias, -radius_init)
                else:
                    torch.nn.init.constant_(layer.bias, 0.0)
                    out_dim = self.dims[i + 1]
                    if i in self.skip_in:
                        out_dim = self.dims[i + 1] - self.in_dim
                    torch.nn.init.normal_(
                        layer.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim)
                    )

            self.layers.append(layer)

    def forward(self, x):
        """
        x: [B, in_dim]
        """
        if self.fourier_transform is not None:
            x = self.fourier_transform(x)

        input_x = x

        for i, layer in enumerate(self.layers):
            if i in self.skip_in:
                x = torch.cat([x, input_x], dim=-1) / np.sqrt(2)
            x = layer(x)
            # Add ReLU activation for all layers except the last one
            if i < len(self.layers) - 1:
                x = torch.relu(x)

        return x
