import functools
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
except ImportError:
    pass

def patched_wrap_flash_attention(
    query,
    key,
    value,
    decoder_segment_ids=None,
    custom_mask=None,
    attn_logits_soft_cap=None,
    head_shards=1,
    q_seq_shards=1,
    jax_mesh=None,
):
    # Overrides that works for this script. DON'T modify them unless you know what you ARE doing.
    mesh = jax_mesh
    batch_axis_name = "batch"
    model_axis_name = "model"
    
    if custom_mask is not None:
        mask = splash_attention_mask.NumpyMask(array=custom_mask)
    else:
        seq_len = query.shape[2]
        mask = splash_attention_mask.CausalMask(shape=(seq_len, seq_len))

    multi_head_mask = splash_attention_mask.MultiHeadMask(
        masks=(mask,) * query.shape[1]
    )

    seq_len = query.shape[2]
    block_size_val = min(512, seq_len)

    dim_per_head = query.shape[-1]
    query = query * (1.0 / jnp.sqrt(dim_per_head)).astype(query.dtype)
    block_sizes = splash_attention_kernel.BlockSizes(
        block_q=block_size_val,
        block_kv_compute=block_size_val,
        block_kv=block_size_val,
        block_q_dkv=block_size_val,
        block_kv_dkv=block_size_val,
        block_kv_dkv_compute=block_size_val,
        block_q_dq=block_size_val,
        block_kv_dq=block_size_val,
    )
    splash_kernel = splash_attention_kernel.make_splash_mha(
        mask=multi_head_mask,
        head_shards=head_shards,
        q_seq_shards=q_seq_shards,
        attn_logits_soft_cap=attn_logits_soft_cap,
        block_sizes=block_sizes 
    )

    def inner_impl(q, k, v, seg_ids):
        if seg_ids is not None:
            return jax.vmap(splash_kernel)(q, k, v, segment_ids=seg_ids)
        else:
            return jax.vmap(splash_kernel)(q, k, v)

    if mesh is None:
        return inner_impl(query, key, value, decoder_segment_ids)

    qkv_spec = PS(batch_axis_name, model_axis_name, None, None)
    
    if decoder_segment_ids is not None:
        seg_spec = PS(batch_axis_name, None)
    else:
        seg_spec = None

    in_specs = (qkv_spec, qkv_spec, qkv_spec, seg_spec) # (query, key, value, decoder_segment_ids)
    out_specs = qkv_spec # (B, H, S, D)

    sharded_attention_fn = shard_map.shard_map(
        inner_impl,
        mesh=mesh,
        in_specs=in_specs,
        out_specs=out_specs,
        check_rep=False
    )

    return sharded_attention_fn(query, key, value, decoder_segment_ids)

def flex_attention(query, key, value, is_causal=False, jax_mesh=None, decoder_segment_ids=None, custom_mask=None, scale=None, bias=None):
    """
    Standardized flex attention wrapper.
    Args:
        query: shape (B, T, N, H)
        key: shape (B, S, K, H)
        value: shape (B, S, K, H)
    Returns:
        output: shape (B, T, N, H)
    """
    backend = jax.default_backend()
    if backend == "tpu" and query.shape[1] == key.shape[1] and bias is None:
        # TPU expects (B, N, T, H)
        q = query.transpose(0, 2, 1, 3)
        k = key.transpose(0, 2, 1, 3)
        v = value.transpose(0, 2, 1, 3)
        
        # Broadcast GQA to MHA for TPU as make_splash_mha expects MHA
        B, N, T, H = q.shape
        _, K, S, _ = k.shape
        if K != N:
            assert N % K == 0
            k = jnp.repeat(k, N // K, axis=1)
            v = jnp.repeat(v, N // K, axis=1)
            
        tpu_mask = None
        if not is_causal:
            tpu_mask = np.ones((T, S), dtype=np.bool_)
            
        out = patched_wrap_flash_attention(
            q, k, v, 
            decoder_segment_ids=decoder_segment_ids, 
            custom_mask=tpu_mask, 
            jax_mesh=jax_mesh
        )
        return out.transpose(0, 2, 1, 3)
    else:
        # GPU / CPU - use jax.nn.dot_product_attention
        # shape expected is (B, T, N, H)
        return jax.nn.dot_product_attention(
            query, key, value,
            bias=bias,
            mask=custom_mask,
            is_causal=is_causal,
            scale=scale
        )
