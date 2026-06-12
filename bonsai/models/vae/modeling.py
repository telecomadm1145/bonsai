import dataclasses

from enum import Enum
from typing import Sequence

import jax
import jax.image
import jax.numpy as jnp
from flax import nnx
from jax import P
from jax.sharding import PartitionSpec, reshard

from bonsai.utils.attention import flex_attention

class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"

@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    conv_kernel: PartitionSpec | None = None
    linear_weight: PartitionSpec | None = None
    group_norm: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    activation_3d: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return ShardConfig()

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            conv_kernel=P(None, None, tp, fsdp),
            linear_weight=P(tp, fsdp),
            group_norm=P(tp),
            activation=P(fsdp, None, None, tp),
            activation_3d=P(fsdp, None, tp),
            attn_qk_activation=P(fsdp, None, tp, None),
        )

def shard(x: jnp.ndarray, s: PartitionSpec | None):
    if s is None:
        return x
    else:
        return reshard(x, s)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    block_out_channels: Sequence[int] = (128, 256, 512, 512)
    latent_channels: int = 4
    norm_num_groups: int = 32
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.no_sharding)

    @classmethod
    def stable_diffusion_v1_5(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(
            block_out_channels=[128, 256, 512, 512],
            latent_channels=4,
            norm_num_groups=32,
            shd_cfg=shd_cfg,
        )


class ResnetBlock(nnx.Module):
    conv_shortcut: nnx.Data[nnx.Conv | None]

    def __init__(self, in_channels: int, out_channels: int, groups: int, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = nnx.Conv(
                in_features=in_channels,
                out_features=out_channels,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding="VALID",
                use_bias=True,
                kernel_metadata={"out_sharding": self.shd.conv_kernel},
                rngs=rngs,
            )
        from functools import partial
        self.norm1 = nnx.GroupNorm(num_groups=groups, num_features=in_channels, epsilon=1e-6, scale_init=partial(jax.nn.initializers.ones, out_sharding=self.shd.group_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=self.shd.group_norm), rngs=rngs)
        self.conv1 = nnx.Conv(
            in_features=in_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )
        self.norm2 = nnx.GroupNorm(num_groups=groups, num_features=out_channels, epsilon=1e-6, scale_init=partial(jax.nn.initializers.ones, out_sharding=self.shd.group_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=self.shd.group_norm), rngs=rngs)
        self.conv2 = nnx.Conv(
            in_features=out_channels,
            out_features=out_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

    def __call__(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = nnx.silu(hidden_states)
        hidden_states = shard(self.conv1(hidden_states), self.shd.activation)

        hidden_states = self.norm2(hidden_states)
        hidden_states = nnx.silu(hidden_states)
        hidden_states = shard(self.conv2(hidden_states), self.shd.activation)

        if self.conv_shortcut is not None:
            input_tensor = shard(self.conv_shortcut(input_tensor), self.shd.activation)

        output_tensor = (input_tensor + hidden_states) / 1.0

        return shard(output_tensor, self.shd.activation)


class DownEncoderBlock2D(nnx.Module):
    downsamplers: nnx.Data[nnx.Conv | None]

    def __init__(self, in_channels: int, out_channels: int, groups: int, is_final_block: bool, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        self.resnets = nnx.List([])

        for i in range(2):
            current_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=current_in_channels, out_channels=out_channels, groups=groups, rngs=rngs, shd=shd)
            )

        self.downsamplers = None

        if not is_final_block:
            self.downsamplers = nnx.Conv(
                in_features=out_channels,
                out_features=out_channels,
                kernel_size=(3, 3),
                strides=(2, 2),
                padding="SAME",
                kernel_metadata={"out_sharding": self.shd.conv_kernel},
                rngs=rngs,
            )

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if self.downsamplers is not None:
            x = shard(self.downsamplers(x), self.shd.activation)

        return x





class Attention(nnx.Module):
    def __init__(self, channels: int, groups: int, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        from functools import partial
        self.group_norm = nnx.GroupNorm(num_groups=groups, num_features=channels, epsilon=1e-6, scale_init=partial(jax.nn.initializers.ones, out_sharding=self.shd.group_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=self.shd.group_norm), rngs=rngs)

        self.to_q = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)
        self.to_k = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)
        self.to_v = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)

        self.to_out = nnx.Linear(in_features=channels, out_features=channels, use_bias=True, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)

    def __call__(self, hidden_states):
        heads = 1
        rescale_output_factor = 1
        residual = hidden_states

        batch_size, height, width, channel = None, None, None, None

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, height, width, channel = hidden_states.shape
            hidden_states = hidden_states.reshape(batch_size, height * width, channel)

        batch_size, _, _ = hidden_states.shape
        hidden_states = self.group_norm(hidden_states)

        query = shard(self.to_q(hidden_states), self.shd.activation_3d)

        encoder_hidden_states = hidden_states

        key = shard(self.to_k(encoder_hidden_states), self.shd.activation_3d)
        value = shard(self.to_v(encoder_hidden_states), self.shd.activation_3d)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // heads

        query = query.reshape(batch_size, -1, heads, head_dim)
        key = key.reshape(batch_size, -1, heads, head_dim)
        value = value.reshape(batch_size, -1, heads, head_dim)

        hidden_states = flex_attention(query, key, value, is_causal=False)
        hidden_states = shard(hidden_states, self.shd.attn_qk_activation)

        B, L, H, D = hidden_states.shape
        hidden_states = hidden_states.reshape(B, L, H * D)

        hidden_states = shard(self.to_out(hidden_states), self.shd.activation_3d)

        if input_ndim == 4:
            hidden_states = hidden_states.reshape(batch_size, height, width, channel)
            hidden_states = shard(hidden_states, self.shd.activation)

        hidden_states = hidden_states + residual
        hidden_states = hidden_states / rescale_output_factor

        return hidden_states


class UNetMidBlock2D(nnx.Module):
    def __init__(self, channels: int, groups: int, num_res_blocks: int, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.resnets = nnx.List([])

        for i in range(num_res_blocks):
            self.resnets.append(ResnetBlock(in_channels=channels, out_channels=channels, groups=groups, rngs=rngs, shd=shd))

        self.attentions = nnx.List([Attention(channels=channels, groups=groups, rngs=rngs, shd=shd)])

    def __call__(self, x):
        x = self.resnets[0](x)
        x = self.attentions[0](x)
        x = self.resnets[1](x)

        return x


class Encoder(nnx.Module):
    def __init__(self, block_out_channels, latent_channels, groups, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        self.conv_in = nnx.Conv(
            in_features=3,
            out_features=block_out_channels[0],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

        self.down_blocks = nnx.List([])

        in_channels = block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            is_final_block = i == len(block_out_channels) - 1

            self.down_blocks.append(
                DownEncoderBlock2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    groups=groups,
                    is_final_block=is_final_block,
                    rngs=rngs,
                    shd=shd,
                )
            )

            in_channels = out_channels

        self.mid_block = UNetMidBlock2D(channels=in_channels, groups=groups, num_res_blocks=2, rngs=rngs, shd=shd)

        from functools import partial
        self.conv_norm_out = nnx.GroupNorm(
            num_groups=groups, num_features=block_out_channels[-1], epsilon=1e-6, scale_init=partial(jax.nn.initializers.ones, out_sharding=self.shd.group_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=self.shd.group_norm), rngs=rngs
        )

        self.conv_out = nnx.Conv(
            in_features=block_out_channels[-1],
            out_features=2 * latent_channels,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

    def __call__(self, x):
        x = shard(self.conv_in(x), self.shd.activation)

        for down_block in self.down_blocks:
            x = down_block(x)

        x = self.mid_block(x)
        x = self.conv_norm_out(x)
        x = nnx.silu(x)
        x = shard(self.conv_out(x), self.shd.activation)

        return x


def upsample_nearest2d(input_tensor, scale_factors):
    # (N, C, H_in, W_in) -> (N, H_in, W_in, C)
    input_permuted = jnp.transpose(input_tensor, (0, 2, 3, 1))

    # Nearest neighbor interpolation using jax.image.resize
    output_permuted = jax.image.resize(
        input_permuted,
        shape=(
            input_permuted.shape[0],
            int(input_permuted.shape[1] * scale_factors[0]),  # H_out
            int(input_permuted.shape[2] * scale_factors[1]),  # W_out
            input_permuted.shape[3],  # C
        ),
        method="nearest",
    )

    # (N, C, H_out, W_out)
    output_tensor = jnp.transpose(output_permuted, (0, 3, 1, 2))

    return output_tensor


def interpolate(input, scale_factor):
    dim = input.ndim - 2  # 4 - 2
    scale_factors = [scale_factor for _ in range(dim)]  # 2.0, 2.0
    return upsample_nearest2d(input, scale_factors)


class Upsample2D(nnx.Module):
    def __init__(self, channel: int, scale_factor: int, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        self.scale_factor = scale_factor
        self.conv = nnx.Conv(
            in_features=channel,
            out_features=channel,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            use_bias=True,
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

    def __call__(self, x):
        b, h, w, c = x.shape
        new_shape = (b, int(h * self.scale_factor), int(w * self.scale_factor), c)
        x = jax.image.resize(x, shape=new_shape, method="nearest")
        x = shard(self.conv(x), self.shd.activation)

        return x


class UpDecoderBlock2D(nnx.Module):
    upsamplers: nnx.Data[Upsample2D | None]

    def __init__(self, in_channels: int, out_channels: int, groups: int, is_final_block: bool, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.resnets = nnx.List([])

        for i in range(3):
            current_in_channels = in_channels if i == 0 else out_channels
            self.resnets.append(
                ResnetBlock(in_channels=current_in_channels, out_channels=out_channels, groups=groups, rngs=rngs, shd=shd)
            )

        if not is_final_block:
            self.upsamplers = Upsample2D(channel=out_channels, scale_factor=2.0, rngs=rngs, shd=shd)
        else:
            self.upsamplers = None

    def __call__(self, x):
        for resnet in self.resnets:
            x = resnet(x)

        if self.upsamplers is not None:
            x = self.upsamplers(x)

        return x


class Decoder(nnx.Module):
    def __init__(self, block_out_channels, latent_channels, groups, rngs: nnx.Rngs, shd: ShardConfig | None = None):
        self.shd = shd or ShardConfig.no_sharding()
        self.conv_in = nnx.Conv(
            in_features=latent_channels,
            out_features=block_out_channels[-1],
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="SAME",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )
        self.mid_block = UNetMidBlock2D(channels=block_out_channels[-1], groups=groups, num_res_blocks=2, rngs=rngs, shd=shd)
        self.up_blocks = nnx.List([])

        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]

        for i, out_channels in enumerate(block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1

            self.up_blocks.append(
                UpDecoderBlock2D(
                    in_channels=prev_output_channel,
                    out_channels=output_channel,
                    groups=groups,
                    is_final_block=is_final_block,
                    rngs=rngs,
                    shd=shd,
                )
            )

            prev_output_channel = output_channel

        from functools import partial
        self.conv_norm_out = nnx.GroupNorm(
            num_groups=groups, num_features=block_out_channels[0], epsilon=1e-6, scale_init=partial(jax.nn.initializers.ones, out_sharding=self.shd.group_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=self.shd.group_norm), rngs=rngs
        )

        self.conv_out = nnx.Conv(block_out_channels[0], 3, kernel_size=(3, 3), strides=1, padding=1, kernel_metadata={"out_sharding": self.shd.conv_kernel}, rngs=rngs)

    def __call__(self, x):
        x = shard(self.conv_in(x), self.shd.activation)
        x = self.mid_block(x)
        for up_block in self.up_blocks:
            x = up_block(x)
        x = self.conv_norm_out(x)
        x = nnx.silu(x)
        x = shard(self.conv_out(x), self.shd.activation)

        return x


class VAE(nnx.Module):
    def __init__(self, cfg: ModelConfig, rngs: nnx.Rngs):
        self.shd = cfg.shd_cfg
        self.encoder = Encoder(cfg.block_out_channels, cfg.latent_channels, cfg.norm_num_groups, rngs, shd=self.shd)

        self.quant_conv = nnx.Conv(
            in_features=2 * cfg.latent_channels,
            out_features=2 * cfg.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

        self.post_quant_conv = nnx.Conv(
            in_features=cfg.latent_channels,
            out_features=cfg.latent_channels,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="VALID",
            kernel_metadata={"out_sharding": self.shd.conv_kernel},
            rngs=rngs,
        )

        self.decoder = Decoder(cfg.block_out_channels, cfg.latent_channels, cfg.norm_num_groups, rngs, shd=self.shd)

    def __call__(self, x):
        x = self.encoder(x)
        x = shard(self.quant_conv(x), self.shd.activation)
        mean, _ = jnp.split(x, 2, axis=-1)
        x = shard(self.post_quant_conv(mean), self.shd.activation)
        x = self.decoder(x)

        return x

    @classmethod
    def from_pretrained(cls, model_name: str, config: ModelConfig | None = None):
        """model_name the *model id* of a pretrained model hosted inside
        a model repo on huggingface.co. For example, "stabilityai/sd-vae-ft-mse"
        """
        from huggingface_hub import snapshot_download
        from bonsai.models.vae import params

        if config is None:
            config_map = {
                "stabilityai/sd-vae-ft-mse": ModelConfig.stable_diffusion_v1_5,
            }
            if model_name not in config_map:
                raise ValueError(f"Model name '{model_name}' is unknown, please provide config argument")
            config = config_map[model_name]()

        model_ckpt_path = snapshot_download(repo_id=model_name, allow_patterns="*.safetensors")
        return params.create_model_from_safe_tensors(model_ckpt_path, config)


@jax.jit
def forward(model, x):
    return model(x)
