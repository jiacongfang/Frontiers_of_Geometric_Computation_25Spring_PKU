from torch import nn
import torch
import numpy as np
import math

from .utils import get_class, number_of_features_per_level

from transformers import T5EncoderModel, AutoTokenizer

from tqdm import tqdm


# Add attention modules
class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, (
                f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            )
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = nn.Conv3d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        b, c, d, h, w = x.shape
        x_flat = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, c * 3, -1)
        h_out = self.attention(qkv)
        h_out = self.proj_out(h_out.reshape(b, c, d, h, w))
        return x + h_out


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block for conditioning.
    """

    def __init__(self, query_dim, context_dim, num_heads=8, head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = head_dim or query_dim // num_heads
        self.scale = head_dim**-0.5

        self.norm = nn.GroupNorm(8, query_dim)
        self.norm_context = nn.LayerNorm(context_dim)

        self.to_q = nn.Linear(query_dim, num_heads * head_dim, bias=False)
        self.to_k = nn.Linear(context_dim, num_heads * head_dim, bias=False)
        self.to_v = nn.Linear(context_dim, num_heads * head_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(num_heads * head_dim, query_dim), nn.Dropout(0.1)
        )

    def forward(self, x, context):
        b, c, d, h, w = x.shape

        # Normalize inputs
        x_norm = self.norm(x)
        context_norm = self.norm_context(context)

        # Flatten spatial dimensions
        q = x_norm.reshape(b, c, -1).permute(0, 2, 1)  # [b, dhw, c]

        # Compute Q, K, V
        q = self.to_q(q)
        k = self.to_k(context_norm)
        v = self.to_v(context_norm)

        # Reshape for multi-head attention
        q = q.reshape(b, -1, self.num_heads, q.shape[-1] // self.num_heads).permute(
            0, 2, 1, 3
        )
        k = k.reshape(b, -1, self.num_heads, k.shape[-1] // self.num_heads).permute(
            0, 2, 1, 3
        )
        v = v.reshape(b, -1, self.num_heads, v.shape[-1] // self.num_heads).permute(
            0, 2, 1, 3
        )

        # Attention
        attn = torch.einsum("bhqd,bhkd->bhqk", q, k) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum("bhqk,bhvd->bhqd", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, -1, out.shape[-1] * self.num_heads)

        out = self.to_out(out)
        out = out.permute(0, 2, 1).reshape(b, c, d, h, w)

        return x + out


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb, context=None):
        raise NotImplementedError


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, context)
            elif isinstance(layer, (AttentionBlock, CrossAttentionBlock)):
                if isinstance(layer, CrossAttentionBlock) and context is not None:
                    x = layer(x, context)
                else:
                    x = layer(x)
            else:
                x = layer(x)
        return x


class ResBlockWithAttention(TimestepBlock):
    """
    A residual block with optional attention mechanism.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        use_checkpoint=False,
        up=False,
        down=False,
        use_attention=False,
        use_cross_attention=False,
        context_dim=None,
        num_heads=8,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.use_attention = use_attention
        self.use_cross_attention = use_cross_attention

        # Main conv layers
        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )

        # Timestep embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        # Zero init the final conv
        nn.init.zeros_(self.out_layers[-1].weight)
        nn.init.zeros_(self.out_layers[-1].bias)

        # Skip connection
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = nn.Conv3d(channels, self.out_channels, 1)

        # Attention layers
        if use_attention:
            self.self_attn = AttentionBlock(self.out_channels, num_heads=num_heads)
        else:
            self.self_attn = None

        if use_cross_attention:
            assert context_dim is not None, (
                "context_dim must be provided for cross attention"
            )
            self.cross_attn = CrossAttentionBlock(
                self.out_channels, context_dim, num_heads=num_heads
            )
        else:
            self.cross_attn = None

    def forward(self, x, emb, context=None):
        h = self.in_layers(x)

        # Apply timestep embedding
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]

        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)

        # Skip connection
        h = self.skip_connection(x) + h

        # Apply attention
        if self.self_attn is not None:
            h = self.self_attn(h)

        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)

        return h


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


class DiffusionUNet3DWithAttention(nn.Module):
    """
    Diffusion model based on UNet3D architecture with attention mechanisms.
    This model supports both self-attention and cross-attention for conditional generation.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        time_emb_dim=64,
        f_maps=64,
        num_res_blocks=2,
        num_levels=4,
        dropout_prob=0.1,
        attention_resolutions=[2, 4],
        use_cross_attention=True,
        context_dim=768,
        num_heads=8,
    ):
        super(DiffusionUNet3DWithAttention, self).__init__()

        self.use_cross_attention = use_cross_attention
        self.context_dim = context_dim

        if isinstance(f_maps, int):
            f_maps = number_of_features_per_level(f_maps, num_levels=num_levels)

        # import ipdb; ipdb.set_trace()

        # Time embedding
        self.time_emb_dim = time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.GELU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder
        self.input_blocks = nn.ModuleList()

        # Initial conv
        self.input_blocks.append(
            TimestepEmbedSequential(nn.Conv3d(in_channels, f_maps[0], 3, padding=1))
        )

        # Encoder blocks
        ch = f_maps[0]
        input_block_chans = [ch]
        ds = 1

        for level, mult in enumerate(f_maps):
            for _ in range(num_res_blocks):
                use_attn = ds in attention_resolutions
                layers = [
                    ResBlockWithAttention(
                        ch,
                        time_emb_dim,
                        dropout_prob,
                        out_channels=mult,
                        use_attention=use_attn,
                        use_cross_attention=use_cross_attention and use_attn,
                        context_dim=context_dim,
                        num_heads=num_heads,
                    )
                ]
                ch = mult
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)

            # Downsampling (except for the last level)
            if level != len(f_maps) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(nn.Conv3d(ch, ch, 3, stride=2, padding=1))
                )
                input_block_chans.append(ch)
                ds *= 2

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlockWithAttention(
                ch,
                time_emb_dim,
                dropout_prob,
                use_attention=True,
                use_cross_attention=use_cross_attention,
                context_dim=context_dim,
                num_heads=num_heads,
            ),
            ResBlockWithAttention(
                ch,
                time_emb_dim,
                dropout_prob,
            ),
        )

        # Decoder
        self.output_blocks = nn.ModuleList()

        for level, mult in list(enumerate(f_maps))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                use_attn = ds in attention_resolutions

                layers = [
                    ResBlockWithAttention(
                        ch + ich,
                        time_emb_dim,
                        dropout_prob,
                        out_channels=mult,
                        use_attention=use_attn,
                        use_cross_attention=use_cross_attention and use_attn,
                        context_dim=context_dim,
                        num_heads=num_heads,
                    )
                ]
                ch = mult

                # Upsampling
                if level > 0 and i == num_res_blocks:
                    layers.append(nn.ConvTranspose3d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2

                self.output_blocks.append(TimestepEmbedSequential(*layers))

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, f_maps[0]),
            nn.SiLU(),
            nn.Conv3d(f_maps[0], out_channels, 3, padding=1),
        )
        # Zero init final conv
        nn.init.zeros_(self.final_conv[-1].weight)
        nn.init.zeros_(self.final_conv[-1].bias)

    def forward(self, x, timesteps, context=None):
        """
        Forward pass with timestep embedding and optional context.

        Args:
            x (torch.Tensor): Input tensor (noisy volume)
            timesteps (torch.Tensor): Current timestep in the diffusion process
            context (torch.Tensor, optional): Context for cross-attention conditioning

        Returns:
            torch.Tensor: Predicted noise
        """
        if self.use_cross_attention:
            assert context is not None, (
                "Context must be provided when using cross attention"
            )

        # Time embedding
        time_emb = self.time_mlp(timesteps)

        # Encoder
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, time_emb, context)
            hs.append(h)

        # Middle
        h = self.middle_block(h, time_emb, context)

        # Decoder
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, time_emb, context)

        # Final output
        return self.final_conv(h)


class DiffusionModelWithAttention:
    def __init__(
        self,
        vqvae,
        unet,
        beta_start=1e-4,
        beta_end=0.02,
        timesteps=1000,
        device="cuda",
        cfg_dropout_prob=0.1,
    ):
        self.vqvae = vqvae
        self.unet = unet
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google-t5/t5-base", device_map="cpu"
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            "google-t5/t5-base", device_map="cpu"
        ).to(device)

        self.timesteps = timesteps
        self.device = device
        self.cfg_dropout_prob = cfg_dropout_prob

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

    def encode_text(self, text, drop_conditioning=False):
        """
        Encode text to embeddings using T5 encoder
        Args:
            text: List of text strings or single string
            drop_conditioning: Whether to drop conditioning for classifier-free guidance
        Returns:
            embeddings: Text embeddings tensor
        """
        if isinstance(text, str):
            text = [text]

        if drop_conditioning:
            text = [""] * len(text)

        # Tokenize text
        tokenized = self.tokenizer(
            text, padding=True, truncation=True, max_length=192, return_tensors="pt"
        )

        # Move to device
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # Encode
        with torch.no_grad():
            outputs = self.text_encoder(**tokenized)
            embeddings = outputs.last_hidden_state

        return embeddings

    def training_loss(self, x_0, text):
        """
        Calculate the training loss for a batch of data with classifier-free guidance training

        Args:
            x_0: Clean data batch
            text: Text input for conditioning

        Returns:
            loss: MSE between predicted and actual noise
        """
        batch_size = x_0.shape[0]
        t = self.sample_timesteps(batch_size)

        x_t, noise = self.forward_diffusion(x_0, t)

        drop_mask = torch.rand(batch_size) < self.cfg_dropout_prob

        text_with_dropout = []
        for i, txt in enumerate(text):
            if drop_mask[i].item():
                text_with_dropout.append("")
            else:
                text_with_dropout.append(txt)

        txt_embeddings = self.encode_text(text_with_dropout, drop_conditioning=False)

        predicted_noise = self.unet(x_t, t, txt_embeddings)

        loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        return loss

    @torch.no_grad()
    def sample(self, initial_latent, text, steps=None, guidance_scale=7.5):
        """
        Sample new data by running the reverse diffusion process with classifier-free guidance

        Args:
            initial_latent.
            text: Text conditioning
            steps: Number of denoising steps (defaults to self.timesteps)
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            x_0: Generated data
        """
        if steps is None:
            steps = self.timesteps

        x_t = initial_latent.to(self.device)
        shape = x_t.shape

        if isinstance(text, str):
            text = [text]

        combined_text = text + [""] * len(text)
        combined_embeddings = self.encode_text(combined_text, drop_conditioning=False)

        batch_size = len(text)
        conditional_embeddings = combined_embeddings[:batch_size]
        unconditional_embeddings = combined_embeddings[batch_size:]

        for i in reversed(tqdm(range(steps))):
            t = torch.tensor([i], device=self.device).repeat(shape[0])

            if guidance_scale != 1.0:
                x_t_combined = torch.cat([x_t, x_t], dim=0)
                t_combined = torch.cat([t, t], dim=0)
                embeddings_combined = torch.cat(
                    [unconditional_embeddings, conditional_embeddings], dim=0
                )
                noise_combined = self.unet(
                    x_t_combined, t_combined, embeddings_combined
                )
                unconditional_noise, conditional_noise = noise_combined.chunk(2, dim=0)
                predicted_noise = unconditional_noise + guidance_scale * (
                    conditional_noise - unconditional_noise
                )
            else:
                predicted_noise = self.unet(x_t, t, conditional_embeddings)

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
