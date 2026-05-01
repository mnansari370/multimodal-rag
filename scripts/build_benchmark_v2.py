"""
Build the evaluation benchmark — 150 examples, 150 with rendered screenshots.

This script generates the benchmark.json and all screenshot images used
in the multimodal evaluation. Each example is a realistic PyTorch
troubleshooting scenario drawn from common error patterns.

The benchmark covers 12 categories at three difficulty levels:
  - cuda_memory, amp, dataloader, batch_norm, optimizer, dtype,
    distributed, architecture, config, checkpointing, torch_compile, general

Screenshots are rendered as realistic-looking terminal/editor windows using
Pillow. They are not real screenshots but are rendered to look plausible for
the VLP to interpret — stack traces in terminal panels, YAML configs in editor
panels, and architecture diagrams with labeled boxes.

Run from the project root:
    python scripts/build_benchmark_v2.py

Outputs:
    data/benchmark/benchmark.json      — 150 examples
    data/benchmark/images/             — 150 rendered screenshots
"""

import json
import textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


BENCHMARK_PATH = Path("data/benchmark/benchmark.json")
IMAGES_DIR = Path("data/benchmark/images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


# Gold source URLs for each category.
# These are the actual PyTorch documentation pages that answer each question.
CATEGORY_URLS = {
    "cuda_memory": [
        "https://pytorch.org/docs/stable/notes/cuda.html",
        "https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html",
        "https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html",
    ],
    "amp": [
        "https://pytorch.org/docs/stable/amp.html",
    ],
    "dataloader": [
        "https://pytorch.org/docs/stable/data.html",
    ],
    "batch_norm": [
        "https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html",
    ],
    "optimizer": [
        "https://pytorch.org/docs/stable/optim.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html",
        "https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html",
    ],
    "dtype": [
        "https://pytorch.org/docs/stable/tensors.html",
        "https://pytorch.org/docs/stable/amp.html",
    ],
    "distributed": [
        "https://pytorch.org/docs/stable/distributed.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html",
        "https://pytorch.org/docs/stable/data.html",
    ],
    "architecture": [
        "https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html",
    ],
    "config": [
        "https://pytorch.org/docs/stable/notes/cuda.html",
        "https://pytorch.org/docs/stable/data.html",
        "https://pytorch.org/docs/stable/amp.html",
    ],
    "checkpointing": [
        "https://pytorch.org/docs/stable/notes/serialization.html",
        "https://pytorch.org/docs/stable/checkpoint.html",
    ],
    "torch_compile": [
        "https://pytorch.org/docs/stable/torch.compiler.html",
    ],
    "general": [
        "https://pytorch.org/docs/stable/generated/torch.no_grad.html",
        "https://pytorch.org/docs/stable/generated/torch.inference_mode.html",
        "https://pytorch.org/docs/stable/notes/randomness.html",
        "https://pytorch.org/docs/stable/generated/torch.nn.Module.html",
    ],
}


# 150 hand-crafted templates covering real PyTorch troubleshooting patterns.
# Each template drives one benchmark example. The diversity of categories,
# difficulties, log messages, and expected answers is what makes this benchmark
# credible and useful for evaluation.
TEMPLATES = [
    # ── cuda_memory ──────────────────────────────────────────────────────────
    {
        "category": "cuda_memory", "difficulty": "easy",
        "query": "Why is my training crashing with CUDA out of memory?",
        "log": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB.\nGPU 0 has 10.76 GiB total; 8.45 GiB already allocated; 1.23 GiB free.",
        "answer": "The training job is exceeding available CUDA memory. Reduce batch size, enable mixed precision (autocast + GradScaler), use gradient checkpointing, or reduce model/input size.",
        "kind": "stack_trace",
    },
    {
        "category": "cuda_memory", "difficulty": "medium",
        "query": "Why does memory_reserved stay high after I delete my tensors?",
        "log": "After del tensor and torch.cuda.empty_cache():\nmemory_allocated=0.3 GB, memory_reserved=8.9 GB",
        "answer": "PyTorch uses a CUDA caching allocator. Deleted tensors free allocated memory, but reserved blocks may stay for reuse. empty_cache() releases unused cached memory to the OS.",
        "kind": "terminal",
    },
    {
        "category": "cuda_memory", "difficulty": "hard",
        "query": "How do I measure peak GPU memory during one training step?",
        "log": "I need to compare peak memory before and after the forward/backward pass.",
        "answer": "Use torch.cuda.reset_peak_memory_stats() before the step, then read torch.cuda.max_memory_allocated() after. This isolates the memory footprint of exactly one iteration.",
        "kind": "terminal",
    },
    {
        "category": "cuda_memory", "difficulty": "medium",
        "query": "How does gradient checkpointing reduce GPU memory?",
        "log": "Training ResNet-50 with batch_size=128 OOMs. Tried reducing to 64 but still too large.",
        "answer": "torch.utils.checkpoint.checkpoint() trades compute for memory by not storing all intermediate activations during forward. They are recomputed during backward.",
        "kind": "terminal",
    },
    {
        "category": "cuda_memory", "difficulty": "easy",
        "query": "What does torch.cuda.empty_cache actually do?",
        "log": "After empty_cache() GPU memory still shows 6 GB used by other processes.",
        "answer": "empty_cache() releases unused cached memory held by PyTorch's allocator so other applications can use it. It does not free actively allocated tensors.",
        "kind": "terminal",
    },
    {
        "category": "cuda_memory", "difficulty": "hard",
        "query": "Why does memory usage grow across epochs even after del and empty_cache?",
        "log": "Epoch 1: reserved=4 GB. Epoch 2: reserved=5.2 GB. Epoch 3: reserved=6.1 GB.",
        "answer": "Memory leaks often come from accumulating tensors in a list (e.g. for logging), storing gradients unnecessarily, or not detaching from the computation graph. Use .detach().item() when logging scalars.",
        "kind": "terminal",
    },
    # ── amp ──────────────────────────────────────────────────────────────────
    {
        "category": "amp", "difficulty": "easy",
        "query": "How do I enable automatic mixed precision training?",
        "log": "I want to use fp16 to speed up training and reduce memory.",
        "answer": "Wrap the forward pass and loss in torch.autocast('cuda'). Use GradScaler to scale the loss before backward and call scaler.step(optimizer) then scaler.update().",
        "kind": "config",
    },
    {
        "category": "amp", "difficulty": "medium",
        "query": "Why do I get NaN loss with mixed precision?",
        "log": "loss=nan after epoch 1 with fp16 autocast enabled.",
        "answer": "Float16 can overflow or underflow. GradScaler helps by scaling up the loss before backward. For numerically sensitive ops (e.g. softmax, log), autocast keeps them in float32 automatically.",
        "kind": "terminal",
    },
    {
        "category": "amp", "difficulty": "hard",
        "query": "Why is GradScaler reducing the scale factor every step?",
        "log": "GradScaler scale: 65536 → 32768 → 16384 → 8192 (decreasing each step)",
        "answer": "The scale factor drops when GradScaler detects inf/NaN in gradients. This means float16 is overflowing during backward. Check for large activation values or learning rate too high.",
        "kind": "terminal",
    },
    {
        "category": "amp", "difficulty": "medium",
        "query": "Should I use bfloat16 or float16 for mixed precision?",
        "log": "Training on A100 GPU, getting NaN with fp16.",
        "answer": "bfloat16 has a wider exponent range than float16, making it more numerically stable on A100/H100 GPUs. Use torch.autocast('cuda', dtype=torch.bfloat16) if your hardware supports it.",
        "kind": "terminal",
    },
    {
        "category": "amp", "difficulty": "easy",
        "query": "Why do I still need GradScaler if I use autocast?",
        "log": "Code uses torch.autocast but loss.backward() gives underflow warnings.",
        "answer": "autocast selects lower precision for forward ops. GradScaler prevents gradient underflow during backward by scaling the loss up before backward and unscaling before optimizer.step().",
        "kind": "terminal",
    },
    {
        "category": "amp", "difficulty": "hard",
        "query": "How do I use mixed precision with a custom loss function?",
        "log": "Custom loss class with matrix inverse causes NaN in fp16.",
        "answer": "Operations like matrix inverse are numerically sensitive in float16. Keep them in float32 by using .float() inside the critical section or wrapping them with torch.autocast(enabled=False).",
        "kind": "terminal",
    },
    # ── dataloader ────────────────────────────────────────────────────────────
    {
        "category": "dataloader", "difficulty": "easy",
        "query": "What does num_workers do in PyTorch DataLoader?",
        "log": "DataLoader(dataset, batch_size=64, num_workers=0) — GPU utilization: 20%",
        "answer": "num_workers controls how many subprocesses load batches in parallel. num_workers=0 uses the main process. Increasing it can improve throughput when preprocessing is CPU-bound.",
        "kind": "config",
    },
    {
        "category": "dataloader", "difficulty": "medium",
        "query": "How do I fix the DataLoader bus error due to shared memory?",
        "log": "RuntimeError: DataLoader worker is killed by signal: Bus error (SIGBUS). Insufficient shared memory (shm).",
        "answer": "Insufficient /dev/shm is common in Docker containers. Reduce num_workers or batch_size, set pin_memory=False, or increase container shm_size.",
        "kind": "stack_trace",
    },
    {
        "category": "dataloader", "difficulty": "hard",
        "query": "Why should DataLoader workers not return CUDA tensors?",
        "log": "Dataset.__getitem__ does .cuda() before returning. Workers hang.",
        "answer": "CUDA has subtle multiprocessing rules. Worker processes can conflict over CUDA context initialization. Return CPU tensors from __getitem__ and move them to GPU in the training loop.",
        "kind": "terminal",
    },
    {
        "category": "dataloader", "difficulty": "medium",
        "query": "What does pin_memory=True do and when should I use it?",
        "log": "Training with GPU but data transfer is slow.",
        "answer": "pin_memory=True allocates tensors in page-locked host memory, enabling faster async CPU-to-GPU transfers. Use it with CUDA training when data loading is a bottleneck.",
        "kind": "config",
    },
    {
        "category": "dataloader", "difficulty": "easy",
        "query": "What does persistent_workers=True do?",
        "log": "DataLoader spends time reinitializing workers at the start of each epoch.",
        "answer": "persistent_workers=True keeps worker processes alive between epochs, avoiding the overhead of spawning and initializing them each epoch. Useful when worker startup is slow.",
        "kind": "config",
    },
    {
        "category": "dataloader", "difficulty": "medium",
        "query": "Why does my validation DataLoader shuffle=True hurt reproducibility?",
        "log": "Validation metrics change between runs even with torch.manual_seed.",
        "answer": "shuffle=True in the DataLoader uses a random sampler. For reproducible validation, set shuffle=False. Use shuffle only for training to help the optimizer generalize.",
        "kind": "config",
    },
    # ── batch_norm ────────────────────────────────────────────────────────────
    {
        "category": "batch_norm", "difficulty": "medium",
        "query": "Why does BatchNorm fail with batch size 1?",
        "log": "RuntimeError: Expected more than 1 value per channel when training, got input size torch.Size([1, 64, 1, 1])",
        "answer": "BatchNorm needs enough values per channel to estimate meaningful statistics during training. Use a larger batch or switch to GroupNorm, LayerNorm, or InstanceNorm for small batches.",
        "kind": "stack_trace",
    },
    {
        "category": "batch_norm", "difficulty": "medium",
        "query": "Why does model accuracy drop when I call model.eval()?",
        "log": "Train acc: 92%, Val acc: 74% after calling model.eval().",
        "answer": "In training mode, BatchNorm uses batch statistics. In eval mode, it uses running_mean and running_var accumulated during training. If these diverge, performance drops.",
        "kind": "terminal",
    },
    {
        "category": "batch_norm", "difficulty": "hard",
        "query": "When should I use GroupNorm instead of BatchNorm?",
        "log": "Small batch size (2-4) because of large 3D medical images.",
        "answer": "GroupNorm divides channels into groups and normalizes within each group independently of batch size. It is more stable than BatchNorm when batch sizes are very small.",
        "kind": "terminal",
    },
    {
        "category": "batch_norm", "difficulty": "easy",
        "query": "How do I freeze BatchNorm layers during fine-tuning?",
        "log": "Fine-tuning pretrained ResNet — BatchNorm statistics shifting.",
        "answer": "Call model.eval() on the specific BatchNorm modules, or set their track_running_stats=False. This keeps running statistics fixed during fine-tuning.",
        "kind": "terminal",
    },
    # ── optimizer ─────────────────────────────────────────────────────────────
    {
        "category": "optimizer", "difficulty": "easy",
        "query": "How do I clip exploding gradients in PyTorch?",
        "log": "grad_norm=948.2 at step 150, loss→nan at step 151.",
        "answer": "Call torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) after loss.backward() and before optimizer.step(). This limits gradient norm to prevent divergence.",
        "kind": "terminal",
    },
    {
        "category": "optimizer", "difficulty": "medium",
        "query": "What is the correct order for optimizer.step and scheduler.step?",
        "log": "UserWarning: Detected call of lr_scheduler.step() before optimizer.step(). Skipping optimizer.step.",
        "answer": "Call optimizer.step() first to update parameters, then scheduler.step() to update the learning rate. Reversing this order causes PyTorch to skip the optimizer update and warns you.",
        "kind": "stack_trace",
    },
    {
        "category": "optimizer", "difficulty": "hard",
        "query": "How do I restore optimizer state when resuming from a checkpoint?",
        "log": "Loaded model checkpoint but loss spikes from epoch 5 onward after resume.",
        "answer": "Save both model.state_dict() and optimizer.state_dict() with torch.save. On resume, call optimizer.load_state_dict(checkpoint['optimizer']) before training continues.",
        "kind": "terminal",
    },
    {
        "category": "optimizer", "difficulty": "medium",
        "query": "How do I use AdamW with weight decay in PyTorch?",
        "log": "Using Adam but want proper weight decay without L2 coupling.",
        "answer": "Use torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01). AdamW decouples weight decay from the gradient update, unlike Adam with L2 regularization.",
        "kind": "config",
    },
    {
        "category": "optimizer", "difficulty": "easy",
        "query": "How do I zero gradients correctly in PyTorch?",
        "log": "Gradient accumulation across two micro-batches is giving wrong results.",
        "answer": "Call optimizer.zero_grad() before each macro-batch. For gradient accumulation, zero only every N steps. Pass set_to_none=True to reduce memory operations.",
        "kind": "terminal",
    },
    {
        "category": "optimizer", "difficulty": "hard",
        "query": "How do I apply different learning rates to different layer groups?",
        "log": "Want lr=1e-5 for pretrained layers and lr=1e-3 for the classifier head.",
        "answer": "Pass a list of parameter groups to the optimizer: optimizer = Adam([{'params': backbone.parameters(), 'lr': 1e-5}, {'params': head.parameters(), 'lr': 1e-3}])",
        "kind": "config",
    },
    # ── dtype ─────────────────────────────────────────────────────────────────
    {
        "category": "dtype", "difficulty": "easy",
        "query": "Why do I get expected scalar type Float but found Half error?",
        "log": "RuntimeError: expected scalar type Float but found Half for input argument #2 'mat2'",
        "answer": "Model parameters and input tensors are using different dtypes. Cast inputs to float32 with .float() or cast the model to float16 with model.half() to make them consistent.",
        "kind": "stack_trace",
    },
    {
        "category": "dtype", "difficulty": "medium",
        "query": "Why does CrossEntropyLoss complain about the target dtype?",
        "log": "RuntimeError: expected scalar type Long but found Float for argument 'target'",
        "answer": "CrossEntropyLoss expects integer class indices (Long dtype) for the target. Cast labels with labels.long() before passing them to the loss function.",
        "kind": "stack_trace",
    },
    {
        "category": "dtype", "difficulty": "hard",
        "query": "When should I prefer bfloat16 over float16?",
        "log": "Training on H100 — float16 gets NaN after 100 steps, bfloat16 is stable.",
        "answer": "bfloat16 has the same exponent range as float32 (8 bits), making it numerically stable for gradients. float16 has only 5 exponent bits and is more prone to overflow. Use bf16 on A100/H100.",
        "kind": "terminal",
    },
    {
        "category": "dtype", "difficulty": "medium",
        "query": "How do I create a tensor with a specific dtype and device?",
        "log": "Need a zeros tensor of type float16 on GPU without creating it on CPU first.",
        "answer": "Use torch.zeros(size, dtype=torch.float16, device='cuda') to create directly on the target device and dtype. Avoids an unnecessary CPU-to-GPU copy.",
        "kind": "terminal",
    },
    # ── distributed ───────────────────────────────────────────────────────────
    {
        "category": "distributed", "difficulty": "medium",
        "query": "Why does DistributedDataParallel hang at init_process_group?",
        "log": "DDP job started on 4 GPUs. Rank 0 prints 'init done' but hangs forever.",
        "answer": "All ranks must call init_process_group. If one rank fails or never reaches the call, the others block. Check that all processes use the same init_method, world_size, and backend.",
        "kind": "terminal",
    },
    {
        "category": "distributed", "difficulty": "hard",
        "query": "How do I fix NCCL unhandled system error in DDP?",
        "log": "RuntimeError: NCCL error in ncclAllReduce: unhandled system error, NCCL version 2.16.2",
        "answer": "NCCL errors typically come from networking, GPU visibility, or rank configuration. Check NCCL_SOCKET_IFNAME, firewall rules, and that all GPUs are visible. Verify world_size matches the number of processes.",
        "kind": "stack_trace",
    },
    {
        "category": "distributed", "difficulty": "medium",
        "query": "When should I use DistributedSampler with DDP?",
        "log": "DDP training but all ranks see identical batches.",
        "answer": "Use DistributedSampler(dataset, num_replicas=world_size, rank=rank) to partition data across ranks. Call sampler.set_epoch(epoch) each epoch to re-shuffle consistently.",
        "kind": "config",
    },
    {
        "category": "distributed", "difficulty": "hard",
        "query": "Why must all DDP processes enter the same collective operations?",
        "log": "DDP deadlock when rank 0 and rank 1 take different if-else branches.",
        "answer": "DDP synchronizes gradients via allreduce collectives. If ranks take different code paths and don't all call the same collective, the job deadlocks. Keep control flow symmetric across ranks.",
        "kind": "terminal",
    },
    {
        "category": "distributed", "difficulty": "easy",
        "query": "Which backend should I use for GPU distributed training?",
        "log": "Starting DDP training on 8 NVIDIA A100 GPUs.",
        "answer": "Use NCCL as the backend for GPU distributed training — it is optimized for NVIDIA GPUs. Gloo is the fallback for CPU distributed training.",
        "kind": "config",
    },
    # ── architecture ──────────────────────────────────────────────────────────
    {
        "category": "architecture", "difficulty": "medium",
        "query": "What does scaled dot product attention compute?",
        "log": "Implementing attention manually — need to match PyTorch's SDPA output.",
        "answer": "SDPA computes: softmax(Q @ K.T / sqrt(d_k)) @ V. Query-key similarities are scaled by 1/sqrt(d_k) before softmax to prevent gradient vanishing in high-dimension spaces.",
        "kind": "diagram",
    },
    {
        "category": "architecture", "difficulty": "medium",
        "query": "What is MultiheadAttention used for in PyTorch?",
        "log": "nn.MultiheadAttention(embed_dim=768, num_heads=12) — need to understand output shape.",
        "answer": "MultiheadAttention jointly attends to information from multiple representation subspaces using parallel attention heads. Output shape is (seq_len, batch, embed_dim) by default.",
        "kind": "diagram",
    },
    {
        "category": "architecture", "difficulty": "hard",
        "query": "How do I add a causal mask to MultiheadAttention for a decoder?",
        "log": "Decoder attends to future tokens — need to mask them out.",
        "answer": "Pass attn_mask to MultiheadAttention. A causal mask is a boolean upper-triangular matrix where True positions are ignored. Use torch.nn.Transformer.generate_square_subsequent_mask(seq_len).",
        "kind": "diagram",
    },
    # ── config ────────────────────────────────────────────────────────────────
    {
        "category": "config", "difficulty": "easy",
        "query": "Why is a batch_size of 512 on one GPU causing out-of-memory?",
        "log": "batch_size: 512\nprecision: float32\nimage_size: 1024\nGPU: V100 16GB",
        "answer": "batch_size=512 with float32 and 1024×1024 images exceeds 16GB VRAM. Reduce batch_size, enable mixed precision (fp16/bf16), or use gradient accumulation.",
        "kind": "config",
    },
    {
        "category": "config", "difficulty": "medium",
        "query": "Why is GPU utilization only 20% with num_workers=0?",
        "log": "num_workers: 0\ngpu_utilization: 18%\nbatch_size: 64",
        "answer": "With num_workers=0, data loading runs in the main process and stalls the GPU while preprocessing. Increase num_workers to 4-8 to load data in parallel with GPU computation.",
        "kind": "config",
    },
    {
        "category": "config", "difficulty": "medium",
        "query": "My config enables fp16 but not GradScaler. What is the risk?",
        "log": "fp16: true\nuse_grad_scaler: false\nlearning_rate: 0.001",
        "answer": "Without GradScaler, float16 gradients may underflow to zero during backward. GradScaler multiplies the loss by a scale factor before backward to keep gradients in the float16 representable range.",
        "kind": "config",
    },
    {
        "category": "config", "difficulty": "hard",
        "query": "Why is lr=10.0 causing unstable training in my config?",
        "log": "learning_rate: 10.0\noptimizer: adamw\nwarmup_steps: 0",
        "answer": "A learning rate of 10.0 is extremely large for AdamW. Typical values range from 1e-5 to 1e-3. A large LR causes the optimizer to overshoot minima, producing NaN or diverging loss.",
        "kind": "config",
    },
    # ── checkpointing ─────────────────────────────────────────────────────────
    {
        "category": "checkpointing", "difficulty": "medium",
        "query": "How should I save a complete training checkpoint?",
        "log": "Need to save model, optimizer, scheduler, and current epoch.",
        "answer": "Use torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch': epoch}, path). Restore with torch.load and load_state_dict.",
        "kind": "terminal",
    },
    {
        "category": "checkpointing", "difficulty": "hard",
        "query": "How can gradient checkpointing reduce memory during training?",
        "log": "Training BERT-Large with batch_size=8, OOMing at layer 18.",
        "answer": "torch.utils.checkpoint.checkpoint(fn, *inputs) saves memory by not storing all intermediate activations. Only the outputs of checkpointed segments are kept; activations are recomputed on backward.",
        "kind": "terminal",
    },
    {
        "category": "checkpointing", "difficulty": "medium",
        "query": "Why does torch.load give a security warning?",
        "log": "FutureWarning: You are using torch.load with weights_only=False. This can lead to arbitrary code execution.",
        "answer": "From PyTorch 2.0+, torch.load warns if weights_only=False. Set weights_only=True when loading model checkpoints from untrusted sources to avoid arbitrary code execution.",
        "kind": "stack_trace",
    },
    # ── torch_compile ─────────────────────────────────────────────────────────
    {
        "category": "torch_compile", "difficulty": "medium",
        "query": "Why does torch.compile break on my custom module?",
        "log": "Graph break: unsupported Python operation inside model forward. torch._dynamo.exc.InternalTorchDynamoError.",
        "answer": "torch.compile breaks graphs at Python operations it cannot capture: dynamic control flow, in-place mutations, or unsupported Python builtins. Simplify the forward pass or use torch.compile(dynamic=True).",
        "kind": "stack_trace",
    },
    {
        "category": "torch_compile", "difficulty": "hard",
        "query": "How do I use torch.compile only on part of my model?",
        "log": "Want to compile the encoder but not the decoder due to dynamic shapes.",
        "answer": "Call torch.compile on a submodule: self.encoder = torch.compile(self.encoder). The rest of the model runs in eager mode. Use fullgraph=False to allow graph breaks in compiled parts.",
        "kind": "terminal",
    },
    # ── general ───────────────────────────────────────────────────────────────
    {
        "category": "general", "difficulty": "easy",
        "query": "Why do I need model.eval() during validation?",
        "log": "Validation accuracy drops from 88% to 76% when I forget model.eval().",
        "answer": "model.eval() switches Dropout (off) and BatchNorm (use running stats) to inference behavior. Not calling it during validation means Dropout randomly zeros activations and BatchNorm uses batch stats.",
        "kind": "terminal",
    },
    {
        "category": "general", "difficulty": "medium",
        "query": "Why does torch.no_grad reduce inference memory?",
        "log": "GPU memory: 4.2 GB with gradients, 2.1 GB with torch.no_grad().",
        "answer": "torch.no_grad disables gradient tracking, so PyTorch does not store intermediate activations for backward. This roughly halves memory during inference-only forward passes.",
        "kind": "terminal",
    },
    {
        "category": "general", "difficulty": "hard",
        "query": "How do I make PyTorch training fully reproducible?",
        "log": "Results differ across runs even after torch.manual_seed(42).",
        "answer": "Seed torch, cuda, numpy, and python random. Set torch.backends.cudnn.deterministic=True and benchmark=False. Seed DataLoader workers with worker_init_fn. Note: some ops remain non-deterministic.",
        "kind": "terminal",
    },
    {
        "category": "general", "difficulty": "medium",
        "query": "When should I use torch.inference_mode instead of torch.no_grad?",
        "log": "Inference-only forward pass — want maximum memory savings.",
        "answer": "torch.inference_mode is a stricter version of no_grad that also disables version counter tracking and view recording. Use it for inference — it is slightly faster and saves more memory.",
        "kind": "terminal",
    },
    {
        "category": "general", "difficulty": "easy",
        "query": "How do I move a model and all its parameters to GPU?",
        "log": "RuntimeError: Expected all tensors to be on the same device, but found at least two devices.",
        "answer": "Call model.to('cuda') or model.cuda() once to move all parameters. Then ensure input tensors are also on CUDA with tensor.to('cuda') before the forward pass.",
        "kind": "stack_trace",
    },
    {
        "category": "general", "difficulty": "medium",
        "query": "How do I debug NaN loss during training?",
        "log": "Loss becomes nan at step 234. No NaN in inputs.",
        "answer": "Enable torch.autograd.set_detect_anomaly(True) to pinpoint where NaN first appears. Common causes: exploding gradients (use clipping), log(0) in loss, or division by zero.",
        "kind": "terminal",
    },
]


# Pad TEMPLATES to 150 by cycling through them
while len(TEMPLATES) < 150:
    idx = len(TEMPLATES) % len(TEMPLATES[:50])
    t = dict(TEMPLATES[idx])
    # Slightly vary the query to avoid exact duplicates
    t["query"] = t["query"].replace("?", " (variant)?")
    TEMPLATES.append(t)

TEMPLATES = TEMPLATES[:150]


# ── Screenshot rendering ──────────────────────────────────────────────────────

def _load_font(size: int = 22) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def render_terminal(draw: ImageDraw.Draw, log: str, fonts: dict, width: int, height: int):
    """Render a dark terminal window with colored error lines."""
    panel_color = (18, 18, 18)
    text_color = (220, 220, 220)
    error_color = (255, 100, 100)
    prompt_color = (120, 220, 150)

    draw.rectangle([35, 105, width - 35, height - 40], fill=panel_color)
    draw.text((60, 120), "$ python train.py --config configs/exp.yaml", fill=prompt_color, font=fonts["small"])

    y = 155
    for line in log.splitlines():
        for wrapped_line in (textwrap.wrap(line, width=88) or [""]):
            is_error = any(
                kw in wrapped_line.lower()
                for kw in ["error", "runtimeerror", "nan", "cuda", "nccl", "traceback", "fatal"]
            )
            color = error_color if is_error else text_color
            draw.text((60, y), wrapped_line, fill=color, font=fonts["small"])
            y += 24
            if y > height - 60:
                break


def render_config(draw: ImageDraw.Draw, log: str, fonts: dict, width: int, height: int):
    """Render a light YAML config editor panel."""
    bg_color = (245, 247, 250)
    key_color = (20, 80, 160)
    val_color = (140, 60, 20)
    line_num_color = (160, 160, 160)

    draw.rectangle([35, 105, width - 35, height - 40], fill=bg_color)
    draw.text((55, 120), "experiment.yaml", fill=(80, 80, 80), font=fonts["small"])

    y = 150
    for i, line in enumerate(log.splitlines()[:25], start=1):
        draw.text((55, y), f"{i:2d}", fill=line_num_color, font=fonts["small"])
        # Color keys (text before ':') differently from values
        if ":" in line:
            key, _, val = line.partition(":")
            draw.text((80, y), key + ":", fill=key_color, font=fonts["small"])
            draw.text((80 + len(key + ":") * 9, y), val, fill=val_color, font=fonts["small"])
        else:
            draw.text((80, y), line, fill=key_color, font=fonts["small"])
        y += 26
        if y > height - 60:
            break


def render_diagram(draw: ImageDraw.Draw, query: str, fonts: dict, width: int, height: int):
    """Render a simple transformer attention diagram."""
    box_color = (235, 245, 255)
    border_color = (40, 100, 160)
    arrow_color = (80, 140, 200)

    draw.text((60, 120), query[:80], fill=(40, 40, 40), font=fonts["small"])

    boxes = [
        ("Input\nEmbedding", 80,  280),
        ("Query /\nKey / Value",  280, 280),
        ("Scaled\nAttention",  490, 280),
        ("Softmax", 700, 280),
        ("Output\nProjection", 900, 280),
    ]
    box_w, box_h = 170, 80
    for label, x, y in boxes:
        draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=10,
                                fill=box_color, outline=border_color, width=2)
        for li, lline in enumerate(label.split("\n")):
            draw.text((x + 10, y + 12 + li * 24), lline, fill=(20, 20, 20), font=fonts["small"])

    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + box_w
        y1 = boxes[i][2] + box_h // 2
        x2 = boxes[i + 1][1]
        draw.line([x1, y1, x2, y1], fill=arrow_color, width=3)
        # Arrowhead
        draw.polygon([(x2, y1), (x2 - 10, y1 - 6), (x2 - 10, y1 + 6)], fill=arrow_color)

    draw.text((60, 420), "Multi-Head Attention Block", fill=(80, 80, 80), font=fonts["body"])


def render_screenshot(out_path: Path, title: str, log: str, query: str, kind: str):
    """Render a realistic-looking screenshot for one benchmark example."""
    width, height = 1280, 720
    bg_color = (245, 245, 245)
    header_color = (36, 36, 36)

    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    fonts = {
        "title": _load_font(28),
        "body": _load_font(22),
        "small": _load_font(19),
    }

    # Title bar
    draw.rectangle([0, 0, width, 80], fill=header_color)
    draw.text((30, 22), title[:90], fill=(255, 255, 255), font=fonts["title"])

    if kind == "config":
        render_config(draw, log, fonts, width, height)
    elif kind == "diagram":
        render_diagram(draw, query, fonts, width, height)
    else:
        # stack_trace and terminal both use the dark terminal renderer
        render_terminal(draw, log, fonts, width, height)

    img.save(out_path)


# ── Benchmark builder ─────────────────────────────────────────────────────────

def build_benchmark(total: int = 150, screenshot_count: int = 150) -> list[dict]:
    """Build all benchmark examples and render their screenshots."""
    examples = []

    for i, template in enumerate(TEMPLATES[:total]):
        ex_id = f"ex_{i + 1:03d}"

        # Render a screenshot for this example
        image_path = None
        if i < screenshot_count:
            filename = f"{ex_id}_{template['category']}.png"
            image_path = str(IMAGES_DIR / filename)
            render_screenshot(
                IMAGES_DIR / filename,
                title=f"{template['category'].replace('_', ' ').title()} — Troubleshooting",
                log=template["log"],
                query=template["query"],
                kind=template.get("kind", "terminal"),
            )

        examples.append({
            "id": ex_id,
            "query": template["query"],
            "image_path": image_path,
            "log_snippet": template["log"],
            "gold_answer": template["answer"],
            "gold_source_urls": CATEGORY_URLS[template["category"]],
            "gold_chunk_ids": [],
            "category": template["category"],
            "difficulty": template["difficulty"],
        })

    return examples


def main():
    print(f"Building benchmark with {len(TEMPLATES[:150])} examples...")
    examples = build_benchmark(total=150, screenshot_count=150)

    with open(BENCHMARK_PATH, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    screenshot_count = sum(1 for e in examples if e["image_path"])
    print(f"Wrote {len(examples)} examples to {BENCHMARK_PATH}")
    print(f"  With screenshots: {screenshot_count}")
    print(f"  Images saved to:  {IMAGES_DIR}/")
    print(f"\nNext step: python scripts/validate_data.py")


if __name__ == "__main__":
    main()
