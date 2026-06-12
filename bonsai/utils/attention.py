# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified attention backend for bonsai models.

Provides a single ``flex_attention`` entry-point that dispatches to
the best available implementation depending on the JAX backend:

* **TPU** – Pallas splash-attention (``make_splash_mha``) with
  optional ``shard_map`` mesh parallelism.
* **GPU / CPU** – ``jax.nn.dot_product_attention`` (XLA custom-call
  on GPU, reference impl on CPU).

All callers should pass Q / K / V in **BTNH** layout
(batch, time/seq, num_heads, head_dim) and will receive output in the
same layout.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import shard_map
from jax.sharding import PartitionSpec as PS

try:
    from jax.experimental.pallas.ops.tpu.splash_attention import (
        splash_attention_kernel,
        splash_attention_mask,
    )
    _HAS_SPLASH = True
except ImportError:
    _HAS_SPLASH = False


# ---------------------------------------------------------------------------
# TPU splash-attention helper (internal)
# ---------------------------------------------------------------------------

def _splash_attention(
    query,
    key,
    value,
    *,
    decoder_segment_ids=None,
    is_causal=True,
    head_shards=1,
    q_seq_shards=1,
    attn_logits_soft_cap=None,
    mesh=None,
    batch_axis_name="batch",
    model_axis_name="model",
):
    """Run splash-attention on TPU.

    Q / K / V must be in **BNTH** layout (batch, heads, time, head_dim).
    """
    if not is_causal:
        seq_len = query.shape[2]
        mask = splash_attention_mask.FullMask(shape=(seq_len, seq_len))
    else:
        seq_len = query.shape[2]
        mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))

    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1]
    )

    seq_len = query.shape[2]
    block_size = min(512, seq_len)

    # Splash attention expects pre-scaled queries.
    dim_per_head = query.shape[-1]
    query = query * jnp.array(1.0 / jnp.sqrt(dim_per_head), dtype=query.dtype)

    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_size,
        block_kv_compute=block_size,
        block_kv=block_size,
        block_q_dkv=block_size,
        block_kv_dkv=block_size,
        block_kv_dkv_compute=block_size,
        block_q_dq=block_size,
        block_kv_dq=block_size,
    )
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        attn_logits_soft_cap=attn_logits_soft_cap,
        block_sizes=block_sizes,
    )

    def _impl(q, k, v, seg_ids):
        if seg_ids is not None:
            return jax.vmap(splash_kernel)(q, k, v, segment_ids=seg_ids)
        return jax.vmap(splash_kernel)(q, k, v)

    if mesh is None:
        return _impl(query, key, value, decoder_segment_ids)

    qkv_spec = PS(batch_axis_name, model_axis_name, None, None)
    seg_spec = PS(batch_axis_name, None) if decoder_segment_ids is not None else None

    sharded_fn = shard_map.shard_map(
        _impl,
        mesh=mesh,
        in_specs=(qkv_spec, qkv_spec, qkv_spec, seg_spec),
        out_specs=qkv_spec,
        check_rep=False,
    )
    return sharded_fn(query, key, value, decoder_segment_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def flex_attention(
    query,
    key,
    value,
    *,
    is_causal=False,
    scale=None,
    bias=None,
    mask=None,
    # TPU-only knobs (ignored on GPU / CPU)
    mesh=None,
    decoder_segment_ids=None,
):
    """Backend-agnostic scaled dot-product attention.

    Args:
        query:  (B, T, N, H)  – queries.
        key:    (B, S, K, H)  – keys  (K may differ from N for GQA).
        value:  (B, S, K, H)  – values.
        is_causal: If ``True``, apply a causal (lower-triangular) mask.
        scale:  Custom scale factor; defaults to ``1 / sqrt(H)``.
        bias:   Additive attention bias, broadcastable to (B, N, T, S).
        mask:   Boolean attention mask, broadcastable to (B, 1, T, S).
                ``True`` means *attend*, ``False`` means *mask out*.
        mesh:   ``jax.sharding.Mesh`` for TPU shard-map parallelism.
        decoder_segment_ids: Segment ids for splash-attention on TPU.

    Returns:
        Output tensor of shape (B, T, N, H).
    """
    backend = jax.default_backend()

    if (
        backend == "tpu"
        and _HAS_SPLASH
        and bias is None
        and mask is None
        and query.shape[1] == key.shape[1]
    ):
        # Splash attention uses BNTH layout.
        q = query.transpose(0, 2, 1, 3)
        k = key.transpose(0, 2, 1, 3)
        v = value.transpose(0, 2, 1, 3)

        # Broadcast GQA → MHA (splash requires equal head counts).
        B, N, T, H = q.shape
        _, K, S, _ = k.shape
        if K != N:
            assert N % K == 0
            k = jnp.repeat(k, N // K, axis=1)
            v = jnp.repeat(v, N // K, axis=1)

        out = _splash_attention(
            q, k, v,
            decoder_segment_ids=decoder_segment_ids,
            is_causal=is_causal,
            mesh=mesh,
        )
        return out.transpose(0, 2, 1, 3)  # back to BTNH

    # GPU / CPU path: jax.nn.dot_product_attention expects (B, T, N, H).
    return jax.nn.dot_product_attention(
        query, key, value,
        bias=bias,
        mask=mask,
        is_causal=is_causal,
        scale=scale,
    )
