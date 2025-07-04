from torch import nn
import torch
import numpy as np

from .buildingblocks import (
    DoubleConv,
    ResNetBlock,
    create_decoders,
    create_encoders,
)
from .utils import get_class, number_of_features_per_level


class AbstractUNet(nn.Module):
    """
    Base class for standard and residual UNet.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the final 1x1 convolution,
            otherwise apply nn.Softmax. In effect only if `self.training == False`, i.e. during validation/testing
        basic_module: basic model for the encoder/decoder (DoubleConv, ResNetBlock, ....)
        layer_order (string): determines the order of layers in `SingleConv` module.
            E.g. 'crg' stands for GroupNorm3d+Conv3d+ReLU. See `SingleConv` for more info
        num_groups (int): number of groups for the GroupNorm
        num_levels (int): number of levels in the encoder/decoder path (applied only if f_maps is an int)
            default: 4
        is_segmentation (bool): if True and the model is in eval mode, Sigmoid/Softmax normalization is applied
            after the final convolution; if False (regression problem) the normalization layer is skipped
        conv_kernel_size (int or tuple): size of the convolving kernel in the basic_module
        pool_kernel_size (int or tuple): the size of the window
        conv_padding (int or tuple): add zero-padding added to all three sides of the input
        conv_upscale (int): number of the convolution to upscale in encoder if DoubleConv, default: 2
        upsample (str): algorithm used for decoder upsampling:
            InterpolateUpsampling:   'nearest' | 'linear' | 'bilinear' | 'trilinear' | 'area'
            TransposeConvUpsampling: 'deconv'
            No upsampling:           None
            Default: 'default' (chooses automatically)
        dropout_prob (float or tuple): dropout probability, default: 0.1
        is3d (bool): if True the model is 3D, otherwise 2D, default: True
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid,
        basic_module,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_kernel_size=3,
        pool_kernel_size=2,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        is3d=True,
    ):
        super(AbstractUNet, self).__init__()

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        assert isinstance(f_maps, list) or isinstance(f_maps, tuple)
        assert len(f_maps) > 1, "Required at least 2 levels in the U-Net"
        if "g" in layer_order:
            assert num_groups is not None, (
                "num_groups must be specified if GroupNorm is used"
            )

        # create encoder path
        self.encoders = create_encoders(
            in_channels,
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            conv_upscale,
            dropout_prob,
            layer_order,
            num_groups,
            pool_kernel_size,
            is3d,
        )

        # create decoder path
        self.decoders = create_decoders(
            f_maps,
            basic_module,
            conv_kernel_size,
            conv_padding,
            layer_order,
            num_groups,
            upsample,
            dropout_prob,
            is3d,
        )

        # in the last layer a 1×1 convolution reduces the number of output channels to the number of labels
        if is3d:
            self.final_conv = nn.Conv3d(f_maps[0], out_channels, 1)
        else:
            self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        if is_segmentation:
            # semantic segmentation problem
            if final_sigmoid:
                self.final_activation = nn.Sigmoid()
            else:
                self.final_activation = nn.Softmax(dim=1)
        else:
            # regression problem
            self.final_activation = None

    def forward(self, x, return_logits=False):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, D, H, W) for 3D or (N, C, H, W) for 2D,
                              where N is the batch size, C is the number of channels,
                              D is the depth, H is the height, and W is the width.
            return_logits (bool): If True, returns both the output and the logits.
                                  If False, returns only the output. Default is False.

        Returns:
            torch.Tensor: The output tensor after passing through the network.
                          If return_logits is True, returns a tuple of (output, logits).
        """
        output, logits = self._forward_logits(x)
        if return_logits:
            return output, logits
        return output

    def _forward_logits(self, x):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            # compute final activation
            out = self.final_activation(x)
            # return both probabilities and logits
            return out, x

        return x, x


class UNet3D(AbstractUNet):
    """
    3DUnet model from
    `"3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation"
        <https://arxiv.org/pdf/1606.06650.pdf>`.

    Uses `DoubleConv` as a basic_module and nearest neighbor upsampling in the decoder
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=4,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(UNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=DoubleConv,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


class ResidualUNet3D(AbstractUNet):
    """
    Residual 3DUnet model implementation based on https://arxiv.org/pdf/1706.00120.pdf.
    Uses ResNetBlock as a basic building block, summation joining instead
    of concatenation joining and transposed convolutions for upsampling (watch out for block artifacts).
    Since the model effectively becomes a residual net, in theory it allows for deeper UNet.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        final_sigmoid=True,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        is_segmentation=True,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
        **kwargs,
    ):
        super(ResidualUNet3D, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=final_sigmoid,
            basic_module=ResNetBlock,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            is_segmentation=is_segmentation,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
            is3d=True,
        )


def get_model(model_config):
    model_class = get_class(
        model_config["name"], modules=["pytorch3dunet.unet3d.model"]
    )
    return model_class(**model_config)


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for time step embedding.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((torch.sin(embeddings), torch.cos(embeddings)), dim=-1)
        return embeddings


class DiffusionUNet3D(nn.Module):
    """
    Diffusion model based on ResidualUNet3D architecture.
    This model predicts noise at each step of the diffusion process.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=64,
        f_maps=64,
        layer_order="gcr",
        num_groups=8,
        num_levels=5,
        conv_padding=1,
        conv_upscale=2,
        upsample="default",
        dropout_prob=0.1,
    ):
        super(DiffusionUNet3D, self).__init__()

        self.unet = ResidualUNet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            final_sigmoid=False,
            is_segmentation=False,
            f_maps=f_maps,
            layer_order=layer_order,
            num_groups=num_groups,
            num_levels=num_levels,
            conv_padding=conv_padding,
            conv_upscale=conv_upscale,
            upsample=upsample,
            dropout_prob=dropout_prob,
        )

        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.time_projections = nn.ModuleList()
        for i, features in enumerate(self.unet.encoders):
            if i == 0:
                feat_dim = f_maps
            else:
                feat_dim = f_maps * (2 ** (i))

            self.time_projections.append(
                nn.Sequential(nn.Linear(time_emb_dim, feat_dim), nn.GELU())
            )

    def forward(self, x, timesteps):
        """
        Forward pass with timestep embedding.

        Args:
            x (torch.Tensor): Input tensor (noisy volume)
            timesteps (torch.Tensor): Current timestep in the diffusion process

        Returns:
            torch.Tensor: Predicted noise
        """
        time_emb = self.time_mlp(timesteps)

        encoders_features = []
        for idx, encoder in enumerate(self.unet.encoders):
            x = encoder(x)
            if idx < len(self.time_projections):
                t_emb = self.time_projections[idx](time_emb)
                t_emb = t_emb.view(t_emb.shape[0], -1, 1, 1, 1)
                x = x + t_emb

            encoders_features.insert(0, x)

        encoders_features = encoders_features[1:]

        for decoder, encoder_features in zip(self.unet.decoders, encoders_features):
            x = decoder(encoder_features, x)

        # Final convolution
        x = self.unet.final_conv(x)

        return x


class DiffusionModel:
    def __init__(
        self, vqvae, unet, beta_start=1e-4, beta_end=0.02, timesteps=1000, device="cuda"
    ):
        self.vqvae = vqvae
        self.unet = unet
        self.timesteps = timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]).to(device), self.alphas_cumprod[:-1]]
        )

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1 / self.alphas)

        self.posterior_variance = (
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        )

    def forward_diffusion(self, x_0, t):
        """
        Forward diffusion process: q(x_t | x_0)

        Args:
            x_0: Initial clean data
            t: Time step

        Returns:
            x_t: Noisy data at time t
            noise: The noise added
        """
        # encode x_0 to latent space
        with torch.no_grad():
            z = self.vqvae(x_0, forward_no_quant=True, encode_only=True)

        noise = torch.randn_like(z)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(
            -1, 1, 1, 1, 1
        )

        # import ipdb; ipdb.set_trace()
        x_t = sqrt_alphas_cumprod_t * z + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def sample_timesteps(self, batch_size):
        """Sample random timesteps for a batch"""
        return torch.randint(
            0, self.timesteps, (batch_size,), device=self.device
        ).long()

    def training_loss(self, x_0):
        """
        Calculate the training loss for a batch of data

        Args:
            x_0: Clean data batch

        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size)

        x_t, noise = self.forward_diffusion(x_0, t)

        predicted_noise = self.unet(x_t, t)

        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, shape, steps=None):
        """
        Sample new data by running the reverse diffusion process

        Args:
            shape: Shape of the data to generate
            steps: Number of denoising steps (defaults to self.timesteps)

        Returns:
            x_0: Generated data
        """
        if steps is None:
            steps = self.timesteps

        x_t = torch.randn(shape).to(self.device)

        for i in reversed(range(steps)):
            t = torch.tensor([i], device=self.device).repeat(shape[0])

            predicted_noise = self.unet(x_t, t)

            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]

            if i > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0

            x_t = (1 / torch.sqrt(alpha)) * (
                x_t - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * predicted_noise
            ) + torch.sqrt(beta) * noise

        return x_t
