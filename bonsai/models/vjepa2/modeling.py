import dataclasses

from enum import Enum
import jax
import jax.numpy as jnp
from flax import nnx
from jax import Array, P
from jax.sharding import PartitionSpec, reshard

class ShardMode(Enum):
    FSDP = "fsdp"
    TP = "tp"

@dataclasses.dataclass(slots=True, frozen=True)
class ShardConfig:
    patch_embed_weight: PartitionSpec | None = None
    linear_weight: PartitionSpec | None = None
    qkv_weight: PartitionSpec | None = None
    mlp_weight: PartitionSpec | None = None
    layer_norm: PartitionSpec | None = None
    activation: PartitionSpec | None = None
    attn_qk_activation: PartitionSpec | None = None

    @staticmethod
    def no_sharding():
        return ShardConfig()

    @staticmethod
    def default(use_fsdp: bool, use_tp: bool):
        fsdp = ShardMode.FSDP.value if use_fsdp else None
        tp = ShardMode.TP.value if use_tp else None
        return ShardConfig(
            patch_embed_weight=P(None, None, None, tp, fsdp),
            linear_weight=P(tp, fsdp),
            qkv_weight=P(tp, fsdp),
            mlp_weight=P(tp, fsdp),
            layer_norm=P(tp),
            activation=P(fsdp, None, tp),
            attn_qk_activation=P(fsdp, None, tp, None),
        )

def shard(x: jnp.ndarray, s: PartitionSpec | None):
    if s is None:
        return x
    else:
        return reshard(x, s)


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    model_type: str = "vjepa2"
    patch_size: int = 16
    tubelet_size: int = 2
    in_chans: int = 3
    crop_size: int = 256
    frames_per_clip: int = 64
    hidden_size: int = 1024
    num_attention_heads: int = 16
    num_hidden_layers: int = 24
    mlp_ratio: float = 4.0
    layer_norm_eps: float = 1e-6
    qkv_bias: bool = True
    hidden_act: str = "gelu"
    pred_hidden_size: int = 384
    pred_num_attention_heads: int = 12
    pred_num_hidden_layers: int = 12
    pred_num_mask_tokens: int = 10
    pred_zero_init_mask_tokens: bool = True
    pred_mlp_ratio: float = 4.0
    num_pooler_layers: int = 3
    num_labels: int = 174
    shd_cfg: ShardConfig = dataclasses.field(default_factory=ShardConfig.no_sharding)

    @classmethod
    def vitl_fpc64_256(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(shd_cfg=shd_cfg)

    @classmethod
    def vith_fpc64_256(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(hidden_size=2180, num_attention_heads=16, num_hidden_layers=32, shd_cfg=shd_cfg)

    @classmethod
    def vitg_fpc64_256(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(hidden_size=1408, num_attention_heads=22, num_hidden_layers=40, mlp_ratio=4.363636363636363, shd_cfg=shd_cfg)

    @classmethod
    def vitg_fpc64_384(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(
            crop_size=384, hidden_size=1408, num_attention_heads=22, num_hidden_layers=40, mlp_ratio=4.363636363636363, shd_cfg=shd_cfg
        )

    @classmethod
    def vitl_fpc16_256(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(frames_per_clip=16, shd_cfg=shd_cfg)

    @classmethod
    def vitl_fpc32_256(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(frames_per_clip=32, num_labels=48, shd_cfg=shd_cfg)

    @classmethod
    def vitg_fpc32_384(cls, use_fsdp: bool = False, use_tp: bool = False):
        shd_cfg = ShardConfig.default(use_fsdp, use_tp) if (use_fsdp or use_tp) else ShardConfig.no_sharding()
        return cls(
            frames_per_clip=32,
            crop_size=384,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            mlp_ratio=4.363636363636363,
            shd_cfg=shd_cfg,
        )

    @classmethod
    def standard_test(cls):
        return cls(
            hidden_size=64,
            num_attention_heads=2,
            num_hidden_layers=2,
            pred_hidden_size=32,
            pred_num_attention_heads=2,
            pred_num_hidden_layers=2,
            crop_size=64,
            frames_per_clip=4,
            num_pooler_layers=1,
        )


def gelu_exact(x):
    return jax.nn.gelu(x, approximate=False)


ACT2FN = {"gelu": gelu_exact, "silu": nnx.silu, "relu": nnx.relu}


class VJEPA2PatchEmbeddings3D(nnx.Module):
    def __init__(self, config: ModelConfig, hidden_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.shd = config.shd_cfg
        self.hidden_size = hidden_size
        kernel = (config.tubelet_size, config.patch_size, config.patch_size)
        self.proj = nnx.Conv(
            in_features=config.in_chans, out_features=hidden_size, kernel_size=kernel, strides=kernel, kernel_metadata={"out_sharding": self.shd.patch_embed_weight}, rngs=rngs
        )

    def __call__(self, pixel_values_videos: Array) -> Array:
        batch_size = pixel_values_videos.shape[0]
        x = shard(self.proj(pixel_values_videos), self.shd.activation)
        x = x.reshape(batch_size, -1, self.hidden_size)
        return x


class VJEPA2Embeddings(nnx.Module):
    def __init__(self, config: ModelConfig, hidden_size: int, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.patch_embeddings = VJEPA2PatchEmbeddings3D(config, hidden_size=hidden_size, rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> Array:
        num_frames = pixel_values_videos.shape[1]
        if num_frames < self.config.tubelet_size:
            repeats = self.config.tubelet_size
            pixel_values_videos = jnp.tile(pixel_values_videos, (1, repeats, 1, 1, 1))
        embeddings = self.patch_embeddings(pixel_values_videos)
        return embeddings


def rotate_queries_or_keys(x: Array, pos: Array, dim: int) -> Array:
    """Apply rotary position embeddings to queries or keys."""
    _, _, _, D = x.shape

    omega = jnp.arange(D // 2, dtype=x.dtype)
    omega = omega / (D / 2.0)
    omega = 1.0 / (10000**omega)

    freq = pos[..., None] * omega

    emb_sin = jnp.sin(freq)
    emb_cos = jnp.cos(freq)

    emb_sin = jnp.tile(emb_sin, (1, 1, 1, 2))
    emb_cos = jnp.tile(emb_cos, (1, 1, 1, 2))

    y = x.reshape(*x.shape[:-1], -1, 2)
    y1, y2 = y[..., 0], y[..., 1]

    y_rotated = jnp.stack((-y2, y1), axis=-1)
    y_rotated = y_rotated.reshape(x.shape)

    rotated = (x * emb_cos) + (y_rotated * emb_sin)
    return rotated


class VJEPA2RopeAttention(nnx.Module):
    def __init__(self, config: ModelConfig, hidden_size: int, num_attention_heads: int, rngs: nnx.Rngs):
        super().__init__()
        self.shd = config.shd_cfg
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if hidden_size % num_attention_heads != 0:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by num_attention_heads {num_attention_heads}")

        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = num_attention_heads * self.attention_head_size

        self.query = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.key = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.value = nnx.Linear(hidden_size, self.all_head_size, use_bias=config.qkv_bias, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.proj = nnx.Linear(hidden_size, hidden_size, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)

        self.grid_size = config.crop_size // config.patch_size
        self.grid_depth = config.frames_per_clip // config.tubelet_size

        # RoPE dimension splits (divide head_dim into 3 parts for d, h, w)
        self.rope_dim = int(2 * ((self.attention_head_size // 3) // 2))
        self.d_dim = self.rope_dim
        self.h_dim = self.rope_dim
        self.w_dim = self.rope_dim

        self.scaling = self.attention_head_size**-0.5

    def _get_frame_pos(self, ids: Array) -> Array:
        tokens_per_frame = self.grid_size * self.grid_size
        return ids // tokens_per_frame

    def _get_height_pos(self, ids: Array) -> Array:
        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)
        ids = ids - tokens_per_frame * frame_ids
        tokens_per_row = self.grid_size
        return ids // tokens_per_row

    def get_position_ids(self, x: Array, masks: Array | None = None) -> tuple[Array, Array, Array]:
        token_size = x.shape[1]

        if masks is not None:
            ids = jnp.broadcast_to(masks[:, None, :], (masks.shape[0], self.num_attention_heads, masks.shape[1]))
        else:
            ids = jnp.arange(token_size)

        tokens_per_frame = self.grid_size * self.grid_size
        frame_ids = self._get_frame_pos(ids)

        tokens_per_row = self.grid_size
        height_ids = self._get_height_pos(ids)

        width_ids = (ids - tokens_per_frame * frame_ids) - tokens_per_row * height_ids

        return frame_ids, height_ids, width_ids

    def apply_rotary_embeddings(self, qk: Array, pos_ids: tuple[Array, Array, Array]) -> Array:
        d_mask, h_mask, w_mask = pos_ids

        s = 0
        qkd = rotate_queries_or_keys(qk[..., s : s + self.d_dim], pos=d_mask, dim=self.d_dim)
        s += self.d_dim

        qkh = rotate_queries_or_keys(qk[..., s : s + self.h_dim], pos=h_mask, dim=self.h_dim)
        s += self.h_dim

        qkw = rotate_queries_or_keys(qk[..., s : s + self.w_dim], pos=w_mask, dim=self.w_dim)
        s += self.w_dim

        if s < self.attention_head_size:
            qkr = qk[..., s:]
            qk = jnp.concatenate([qkd, qkh, qkw, qkr], axis=-1)
        else:
            qk = jnp.concatenate([qkd, qkh, qkw], axis=-1)

        return qk

    def __call__(self, hidden_states: Array, position_mask: Array | None = None) -> Array:
        batch_size, seq_length, _ = hidden_states.shape

        query_layer = shard(self.query(hidden_states), self.shd.activation)
        key_layer = shard(self.key(hidden_states), self.shd.activation)
        value_layer = shard(self.value(hidden_states), self.shd.activation)

        query_layer = query_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        query_layer = query_layer.transpose(0, 2, 1, 3)

        key_layer = key_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        key_layer = key_layer.transpose(0, 2, 1, 3)

        value_layer = value_layer.reshape(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        value_layer = value_layer.transpose(0, 2, 1, 3)

        pos_ids = self.get_position_ids(hidden_states, masks=position_mask)
        query_layer = self.apply_rotary_embeddings(query_layer, pos_ids)
        key_layer = self.apply_rotary_embeddings(key_layer, pos_ids)

        from bonsai.utils.attention import flex_attention
        context_layer = flex_attention(
            query_layer.transpose(0, 2, 1, 3),
            key_layer.transpose(0, 2, 1, 3),
            value_layer.transpose(0, 2, 1, 3),
            scale=self.scaling,
            is_causal=False
        )
        context_layer = shard(context_layer, self.shd.attn_qk_activation)
        context_layer = context_layer.reshape(batch_size, seq_length, self.all_head_size)

        output = shard(self.proj(context_layer), self.shd.activation)
        return output


class VJEPA2MLP(nnx.Module):
    def __init__(self, config: ModelConfig, hidden_size: int, mlp_ratio: float, rngs: nnx.Rngs):
        super().__init__()
        self.shd = config.shd_cfg
        hidden_features = int(hidden_size * mlp_ratio)
        self.fc1 = nnx.Linear(hidden_size, hidden_features, kernel_metadata={"out_sharding": self.shd.mlp_weight}, rngs=rngs)
        self.fc2 = nnx.Linear(hidden_features, hidden_size, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)
        self.activation = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Array) -> Array:
        hidden_states = shard(self.fc1(hidden_states), self.shd.activation)
        hidden_states = self.activation(hidden_states)
        hidden_states = shard(self.fc2(hidden_states), self.shd.activation)
        return hidden_states


class VJEPA2Layer(nnx.Module):
    def __init__(
        self, config: ModelConfig, hidden_size: int, num_attention_heads: int, mlp_ratio: float, rngs: nnx.Rngs
    ):
        super().__init__()
        from functools import partial
        self.norm1 = nnx.LayerNorm(hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.attention = VJEPA2RopeAttention(
            config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, rngs=rngs
        )
        self.norm2 = nnx.LayerNorm(hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.mlp = VJEPA2MLP(config, hidden_size=hidden_size, mlp_ratio=mlp_ratio, rngs=rngs)

    def __call__(self, hidden_states: Array, position_mask: Array | None = None) -> Array:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, position_mask=position_mask)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual

        return hidden_states


class VJEPA2Encoder(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.embeddings = VJEPA2Embeddings(config, hidden_size=config.hidden_size, rngs=rngs)
        self.layer = nnx.List(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.hidden_size,
                    num_attention_heads=config.num_attention_heads,
                    mlp_ratio=config.mlp_ratio,
                    rngs=rngs,
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        from functools import partial
        self.layernorm = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> Array:
        hidden_states = self.embeddings(pixel_values_videos)
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_mask=None)
        hidden_states = self.layernorm(hidden_states)
        return hidden_states


def apply_masks(tensor: Array, masks: list) -> Array:
    all_masked = []
    for mask in masks:
        batch_size = tensor.shape[0]
        feature_dim = tensor.shape[-1]
        mask_expanded = jnp.broadcast_to(mask[..., None], (batch_size, mask.shape[1], feature_dim))
        gathered = jnp.take_along_axis(tensor, mask_expanded.astype(jnp.int32), axis=1)
        all_masked.append(gathered)
    return jnp.concatenate(all_masked, axis=0)


class VJEPA2PredictorEmbeddings(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.predictor_embeddings = nnx.Linear(config.hidden_size, config.pred_hidden_size, kernel_metadata={"out_sharding": config.shd_cfg.linear_weight}, rngs=rngs)

        if config.pred_zero_init_mask_tokens:
            mask_tokens = jnp.zeros((config.pred_num_mask_tokens, 1, 1, config.pred_hidden_size))
        else:
            mask_tokens = (
                jax.random.normal(rngs.params(), (config.pred_num_mask_tokens, 1, 1, config.pred_hidden_size)) * 0.02
            )
        self.mask_tokens = nnx.Param(mask_tokens)

    def __call__(
        self, hidden_states: Array, context_mask: list, target_mask: list, mask_index: int = 1
    ) -> tuple[Array, Array]:
        batch_size = hidden_states.shape[0]
        context = shard(self.predictor_embeddings(hidden_states), self.config.shd_cfg.activation)

        mask_index = mask_index % self.config.pred_num_mask_tokens
        target_token = self.mask_tokens[mask_index]

        max_patch_num = int(jnp.max(target_mask[0])) + 1
        target = jnp.tile(target_token, (batch_size, max_patch_num, 1))
        target = apply_masks(target, target_mask)

        context = jnp.tile(context, (len(context_mask), 1, 1))
        embeddings = jnp.concatenate([context, target], axis=1)

        cm = jnp.concatenate(context_mask, axis=0)
        tm = jnp.concatenate(target_mask, axis=0)
        masks = jnp.concatenate([cm, tm], axis=1)

        return embeddings, masks


class VJEPA2Predictor(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.embeddings = VJEPA2PredictorEmbeddings(config, rngs=rngs)
        self.layer = nnx.List(
            [
                VJEPA2Layer(
                    config,
                    hidden_size=config.pred_hidden_size,
                    num_attention_heads=config.pred_num_attention_heads,
                    mlp_ratio=config.pred_mlp_ratio,
                    rngs=rngs,
                )
                for _ in range(config.pred_num_hidden_layers)
            ]
        )
        from functools import partial
        self.layernorm = nnx.LayerNorm(config.pred_hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.proj = nnx.Linear(config.pred_hidden_size, config.hidden_size, kernel_metadata={"out_sharding": config.shd_cfg.linear_weight}, rngs=rngs)

    def sort_tokens(self, hidden_states: Array, position_masks: Array, argsort: Array) -> tuple[Array, Array]:
        position_masks = jnp.take_along_axis(position_masks, argsort, axis=1)
        hidden_states_argsort = jnp.broadcast_to(argsort[..., None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, hidden_states_argsort, axis=1)
        return hidden_states, position_masks

    def unsort_tokens(self, hidden_states: Array, argsort: Array) -> Array:
        reverse_argsort = jnp.argsort(argsort, axis=1)
        reverse_argsort = jnp.broadcast_to(reverse_argsort[..., None], hidden_states.shape)
        hidden_states = jnp.take_along_axis(hidden_states, reverse_argsort, axis=1)
        return hidden_states

    def __call__(self, encoder_hidden_states: Array, context_mask: list, target_mask: list) -> Array:
        encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask)
        _, n_ctxt, _ = encoder_hidden_states.shape

        hidden_states, position_masks = self.embeddings(encoder_hidden_states, context_mask, target_mask)

        argsort = jnp.argsort(position_masks, axis=1)
        hidden_states, position_masks = self.sort_tokens(hidden_states, position_masks, argsort)

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, position_mask=position_masks)

        hidden_states = self.layernorm(hidden_states)
        hidden_states = self.unsort_tokens(hidden_states, argsort)
        hidden_states = hidden_states[:, n_ctxt:]
        hidden_states = shard(self.proj(hidden_states), self.config.shd_cfg.activation)

        return hidden_states


class VJEPA2PoolerSelfAttention(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.shd = config.shd_cfg
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.k_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.v_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.out_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.linear_weight}, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        batch_size, seq_length, _ = hidden_states.shape

        queries = shard(self.q_proj(hidden_states), self.shd.activation)
        keys = shard(self.k_proj(hidden_states), self.shd.activation)
        values = shard(self.v_proj(hidden_states), self.shd.activation)

        from bonsai.utils.attention import flex_attention
        attn_output = flex_attention(
            queries.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3),
            keys.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3),
            values.reshape(batch_size, seq_length, self.num_heads, self.head_dim).transpose(0, 2, 1, 3),
            scale=self.scale,
            is_causal=False
        )
        attn_output = shard(attn_output, self.shd.attn_qk_activation)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_length, self.embed_dim)

        attn_output = shard(self.out_proj(attn_output), self.shd.activation)

        return attn_output


class VJEPA2PoolerCrossAttention(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.shd = config.shd_cfg
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.k_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)
        self.v_proj = nnx.Linear(self.embed_dim, self.embed_dim, kernel_metadata={"out_sharding": self.shd.qkv_weight}, rngs=rngs)

    def __call__(self, queries: Array, keys: Array, values: Array) -> Array:
        batch_size, q_seq_length, _ = queries.shape
        kv_seq_length = keys.shape[1]

        queries = shard(self.q_proj(queries), self.shd.activation)
        keys = shard(self.k_proj(keys), self.shd.activation)
        values = shard(self.v_proj(values), self.shd.activation)

        from bonsai.utils.attention import flex_attention
        attn_output = flex_attention(
            queries.reshape(batch_size, q_seq_length, self.num_heads, self.head_dim),
            keys.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim),
            values.reshape(batch_size, kv_seq_length, self.num_heads, self.head_dim),
            scale=self.scale,
            is_causal=False
        )
        attn_output = shard(attn_output, self.shd.attn_qk_activation)
        attn_output = attn_output.reshape(batch_size, q_seq_length, self.embed_dim)

        return attn_output


class VJEPA2PoolerSelfAttentionLayer(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        from functools import partial
        self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.self_attn = VJEPA2PoolerSelfAttention(config, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio, rngs=rngs)

    def __call__(self, hidden_states: Array) -> Array:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VJEPA2PoolerCrossAttentionLayer(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        from functools import partial
        self.layer_norm1 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.cross_attn = VJEPA2PoolerCrossAttention(config, rngs=rngs)
        self.layer_norm2 = nnx.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps, scale_init=partial(jax.nn.initializers.ones, out_sharding=config.shd_cfg.layer_norm), bias_init=partial(jax.nn.initializers.zeros, out_sharding=config.shd_cfg.layer_norm), rngs=rngs)
        self.mlp = VJEPA2MLP(config, hidden_size=config.hidden_size, mlp_ratio=config.mlp_ratio, rngs=rngs)

    def __call__(self, queries: Array, hidden_state: Array) -> Array:
        residual = queries
        hidden_state_normed = self.layer_norm1(hidden_state)
        hidden_state = self.cross_attn(queries, hidden_state_normed, hidden_state_normed)
        hidden_state = residual + hidden_state

        residual = hidden_state
        hidden_state = self.layer_norm2(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = residual + hidden_state

        return hidden_state


class VJEPA2AttentivePooler(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.query_tokens = nnx.Param(jnp.zeros((1, 1, config.hidden_size)))
        self.cross_attention_layer = VJEPA2PoolerCrossAttentionLayer(config, rngs=rngs)
        self.self_attention_layers = nnx.List(
            [VJEPA2PoolerSelfAttentionLayer(config, rngs=rngs) for _ in range(config.num_pooler_layers)]
        )

    def __call__(self, hidden_state: Array) -> Array:
        batch_size = hidden_state.shape[0]

        for layer in self.self_attention_layers:
            hidden_state = layer(hidden_state)

        queries = jnp.tile(self.query_tokens[...], (batch_size, 1, 1))
        hidden_state = self.cross_attention_layer(queries, hidden_state)

        return hidden_state.squeeze(1)


class VJEPA2Model(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.encoder = VJEPA2Encoder(config, rngs=rngs)
        self.predictor = VJEPA2Predictor(config, rngs=rngs)

    def __call__(
        self,
        pixel_values_videos: Array,
        context_mask: list | None = None,
        target_mask: list | None = None,
        skip_predictor: bool = False,
    ) -> dict:
        sequence_output = self.encoder(pixel_values_videos)

        batch_size = pixel_values_videos.shape[0]
        num_patches = sequence_output.shape[1]

        if context_mask is None and target_mask is None:
            default_mask = jnp.broadcast_to(jnp.arange(num_patches)[None, :], (batch_size, num_patches))
            context_mask = [default_mask]
            target_mask = [default_mask]

        predictor_last_hidden_state = None
        predictor_target_hidden_state = None
        if not skip_predictor:
            predictor_last_hidden_state = self.predictor(
                encoder_hidden_states=sequence_output, context_mask=context_mask, target_mask=target_mask
            )
            predictor_target_hidden_state = apply_masks(sequence_output, target_mask)

        masked_hidden_state = apply_masks(sequence_output, context_mask) if context_mask else None
        return {
            "last_hidden_state": sequence_output,
            "masked_hidden_state": masked_hidden_state,
            "predictor_last_hidden_state": predictor_last_hidden_state,
            "predictor_target_hidden_state": predictor_target_hidden_state,
        }

    def get_vision_features(self, pixel_values_videos: Array) -> Array:
        outputs = self(pixel_values_videos, skip_predictor=True)
        return outputs["last_hidden_state"]


class VJEPA2ForVideoClassification(nnx.Module):
    def __init__(self, config: ModelConfig, rngs: nnx.Rngs):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.vjepa2 = VJEPA2Model(config, rngs=rngs)
        self.pooler = VJEPA2AttentivePooler(config, rngs=rngs)
        self.classifier = nnx.Linear(config.hidden_size, config.num_labels, kernel_metadata={"out_sharding": config.shd_cfg.linear_weight}, rngs=rngs)

    def __call__(self, pixel_values_videos: Array) -> dict[str, Array]:
        outputs = self.vjepa2(pixel_values_videos, skip_predictor=True)
        last_hidden_state = outputs["last_hidden_state"]
        pooler_output = self.pooler(last_hidden_state)
        logits = shard(self.classifier(pooler_output), self.config.shd_cfg.activation)
        return {"logits": logits, "last_hidden_state": last_hidden_state}


@jax.jit
def forward(model, inputs):
    return model(inputs)
