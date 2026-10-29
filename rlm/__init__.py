"""Phase 1: reasoning model (SFT + GRPO). See rlm/README.md."""


def disable_cudnn_attention() -> None:
    """Keep PyTorch from using cuDNN for attention (see EXPERIMENTS.md).

    On the DGX's H200, PyTorch picks cuDNN for ``scaled_dot_product_attention``, which
    rebuilds its plan on the CPU for every new sequence length: when generating, that is
    every token. Qwen3-1.7B went from 288 to 15 ms per token without it. Flash or
    memory-efficient attention take over; elsewhere (e.g. a laptop GPU) nothing changes.
    """
    import torch

    torch.backends.cuda.enable_cudnn_sdp(False)
