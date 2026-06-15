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

"""Training demo for Qwen3-4B on TPU v5e-8.

Demonstrates how to train Qwen3 using the bonsai library with:
  - FSDP + TP sharding across 8 TPU chips
  - Splash attention (automatic on TPU, native GQA support)
  - bf16 training for peak MXU throughput
  - JIT-compiled train step with buffer donation
  - Throughput measurement (tokens/sec)

Usage (TPU v5e-8):
    python train_qwen3.py

Usage (CPU / GPU, for testing):
    python train_qwen3.py --small
"""

import argparse
import time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jax import P
from jax.sharding import AxisType

from bonsai.models.qwen3 import modeling
from bonsai.utils.attention import flex_attention


# ---------------------------------------------------------------------------
# Training forward pass (cache-free)
# ---------------------------------------------------------------------------

def train_attention(attn: modeling.Attention, x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Cache-free attention for training. Uses causal masking via flex_attention.

    Args:
        attn: Attention module (with proj weights and norms).
        x: Input hidden states, shape [B, T, D].
        positions: Position ids, shape [B, T].

    Returns:
        Output hidden states, shape [B, T, D].
    """
    shd = attn.shd_cfg.act_btnh
    query = attn.q_norm(attn.q_proj(x, out_sharding=shd))   # [B, T, N, H]
    key = attn.k_norm(attn.k_proj(x, out_sharding=shd))     # [B, T, K, H]
    value = attn.v_proj(x, out_sharding=shd)                 # [B, T, K, H]

    # RoPE
    sin, cos = modeling._generate_pos_embeddings(positions, attn.head_dim)
    query = modeling.apply_rope(query, sin, cos)
    key = modeling.apply_rope(key, sin, cos)

    # Causal attention — on TPU this dispatches to splash attention (no mask/bias).
    # flex_attention handles GQA natively (no K/V repeat needed).
    out = flex_attention(query, key, value, is_causal=True, scale=attn.scale)
    return attn.o_proj(out, out_sharding=attn.shd_cfg.act_btd)


def train_decoder_layer(layer: modeling.DecoderLayer, x: jnp.ndarray, positions: jnp.ndarray) -> jnp.ndarray:
    """Cache-free decoder layer forward."""
    normed = layer.input_layernorm(x)
    attn_out = x + train_attention(layer.attn, normed, positions)
    return attn_out + layer.mlp(layer.post_attention_layernorm(attn_out))


def train_forward(model: modeling.Qwen3, tokens: jnp.ndarray) -> jnp.ndarray:
    """Cache-free forward pass for training. Returns logits [B, T, V].

    Args:
        model: Qwen3 model.
        tokens: Input token ids, shape [B, T].

    Returns:
        Logits of shape [B, T, V].
    """
    B, T = tokens.shape
    positions = jnp.broadcast_to(jnp.arange(T, dtype=jnp.int32)[None, :], (B, T))
    x = model.embedder.embedding[...].at[(tokens,)].get(out_sharding=model.out_emb_shd)
    for layer in model.layers:
        x = train_decoder_layer(layer, x, positions)
    return model.lm_head(model.final_norm(x), out_sharding=model.out_emb_shd)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def compute_loss(model: modeling.Qwen3, tokens: jnp.ndarray) -> jnp.ndarray:
    """Causal LM loss: predict next token.

    Uses teacher-forcing: input = tokens[:, :-1], target = tokens[:, 1:].

    Args:
        model: Qwen3 model.
        tokens: Token ids of shape [B, T].

    Returns:
        Scalar mean cross-entropy loss.
    """
    inputs = tokens[:, :-1]     # [B, T-1]
    targets = tokens[:, 1:]     # [B, T-1]

    logits = train_forward(model, inputs)  # [B, T-1, V]

    # Cross-entropy loss
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Gather log-probs at target indices
    target_log_probs = jnp.take_along_axis(
        log_probs, targets[:, :, None], axis=-1
    ).squeeze(-1)  # [B, T-1]

    return -jnp.mean(target_log_probs)

# ---------------------------------------------------------------------------
# Train step
# ---------------------------------------------------------------------------

@nnx.jit
def train_step(optimizer: nnx.Optimizer, tokens: jnp.ndarray) -> jnp.ndarray:
    """Single training step: forward + backward + optimizer update.

    Uses Flax NNX reference semantics — optimizer.update(grads) mutates
    the model parameters in-place. No need to return updated state.
    """
    def loss_fn(model):
        return compute_loss(model, tokens)

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    num_steps: int = 50,
    seq_len: int = 1024,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 5,
    weight_decay: float = 0.01,
    seed: int = 42,
    small: bool = False,
    use_pretrained: bool = False,
):
    """Main training loop with throughput measurement.

    Args:
        num_steps: Number of training steps.
        seq_len: Sequence length (tokens per sample).
        batch_size: Global batch size.
        learning_rate: Peak learning rate.
        warmup_steps: Linear warmup steps.
        weight_decay: AdamW weight decay.
        seed: Random seed.
        small: If True, use Qwen3-0.6B for CPU/GPU testing.
        use_pretrained: If True, load pretrained weights from HuggingFace.
    """
    num_devices = jax.device_count()
    backend = jax.default_backend()
    print(f"Backend: {backend}, Devices: {num_devices}")

    # ---- Mesh & sharding ----
    if small:
        # CPU/GPU testing: no sharding
        print(">>> Small mode (qwen3_0_6b, no sharding)")
        config = modeling.ModelConfig.qwen3_0_6b()
        seq_len = min(seq_len, 128)
        batch_size = 2
    elif num_devices >= 8:
        # TPU v5e-8: FSDP=4, TP=2
        fsdp_size = num_devices // 2
        tp_size = 2
        print(f">>> Mesh: FSDP={fsdp_size}, TP={tp_size}")
        mesh = jax.make_mesh(
            (fsdp_size, tp_size),
            ("fsdp", "tp"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        jax.set_mesh(mesh)
        config = modeling.ModelConfig.qwen3_4b(use_fsdp=True, use_tp=True)
    elif num_devices >= 4:
        # 4 devices: FSDP=2, TP=2
        print(">>> Mesh: FSDP=2, TP=2")
        mesh = jax.make_mesh(
            (2, 2),
            ("fsdp", "tp"),
            axis_types=(AxisType.Explicit, AxisType.Explicit),
        )
        jax.set_mesh(mesh)
        config = modeling.ModelConfig.qwen3_4b(use_fsdp=True, use_tp=True)
    else:
        # Single device
        print(">>> Single device, no sharding")
        config = modeling.ModelConfig.qwen3_4b()

    # ---- Model ----
    print(f"Model: Qwen3 ({config.num_layers}L, d={config.emb_dim}, "
          f"h={config.num_heads}, kv_h={config.num_kv_heads})")
    print(f"Seq len: {seq_len}, Batch size: {batch_size}")

    if use_pretrained:
        model_name = "Qwen/Qwen3-0.6B" if small else "Qwen/Qwen3-4B"
        print(f"Loading pretrained: {model_name} ...")
        model = modeling.Qwen3.from_pretrained(model_name, config)
    else:
        print("Random initialization ...")
        model = modeling.Qwen3(config, rngs=nnx.Rngs(params=seed))

    # ---- Optimizer ----
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=max(num_steps, warmup_steps + 1),
    )
    tx = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    optimizer = nnx.Optimizer(model, tx)

    # ---- Synthetic data ----
    rng = jax.random.key(seed)

    # Batch sharding: shard along FSDP axis if available
    fsdp_name = modeling.ShardMode.FSDP.value
    batch_shd = P(fsdp_name) if config.shd_cfg.act_btd is not None else None

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Starting training for {num_steps} steps ...")
    print(f"{'='*60}\n")

    jit_warmup_steps = 2  # First steps include JIT compilation
    total_tokens = 0
    timing_start = None
    losses = []

    for step in range(num_steps):
        # Generate random tokens for this step
        rng, data_rng = jax.random.split(rng)
        tokens = jax.random.randint(
            data_rng,
            shape=(batch_size, seq_len),
            minval=0,
            maxval=config.vocab_size,
            dtype=jnp.int32,
        )
        if batch_shd is not None:
            tokens = jax.device_put(tokens, batch_shd)

        # Train step
        step_start = time.perf_counter()
        loss = train_step(optimizer, tokens)
        loss_val = float(loss)
        step_time = time.perf_counter() - step_start


        # Start timing after JIT warmup
        if step == jit_warmup_steps:
            timing_start = time.perf_counter()
            total_tokens = 0

        if step >= jit_warmup_steps:
            step_tokens = batch_size * (seq_len - 1)  # teacher-forcing: T-1 target tokens
            total_tokens += step_tokens
            elapsed = time.perf_counter() - timing_start
            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            samples_per_sec = (step - jit_warmup_steps + 1) * batch_size / elapsed if elapsed > 0 else 0

            if step % 5 == 0 or step < jit_warmup_steps + 3:
                print(
                    f"Step {step:4d} | loss={loss_val:.4f} | "
                    f"step_time={step_time:.3f}s | "
                    f"throughput={tokens_per_sec:.0f} tok/s | "
                    f"{samples_per_sec:.1f} samples/s"
                )
        else:
            print(f"Step {step:4d} | loss={loss_val:.4f} | step_time={step_time:.3f}s (JIT warmup)")

        losses.append(loss_val)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("Training complete!")
    if timing_start is not None:
        measured_steps = num_steps - jit_warmup_steps
        total_elapsed = time.perf_counter() - timing_start
        avg_step_time = total_elapsed / measured_steps if measured_steps > 0 else 0
        avg_throughput = total_tokens / total_elapsed if total_elapsed > 0 else 0
        print(f"  Measured steps: {measured_steps}")
        print(f"  Avg step time: {avg_step_time:.3f}s")
        print(f"  Avg throughput: {avg_throughput:.0f} tokens/sec")
        print(f"  Total tokens processed: {total_tokens:,}")
    if len(losses) >= 10:
        print(f"  First 5 losses: {[f'{l:.4f}' for l in losses[:5]]}")
        print(f"  Last 5 losses:  {[f'{l:.4f}' for l in losses[-5:]]}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Qwen3 with bonsai (JAX)")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Global batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=5, help="LR warmup steps")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--small", action="store_true", help="Use qwen3-0.6B for CPU/GPU testing")
    parser.add_argument("--pretrained", action="store_true", help="Load pretrained weights")
    args = parser.parse_args()

    train(
        num_steps=args.num_steps,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        seed=args.seed,
        small=args.small,
        use_pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
