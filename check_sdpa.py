import jax
try:
    print("sdpa exists:", hasattr(jax.nn, "sdpa"))
    import inspect
    print(inspect.signature(jax.nn.sdpa))
except Exception as e:
    print(e)
