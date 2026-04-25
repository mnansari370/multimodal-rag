import json
from pathlib import Path

path = Path("data/benchmark/benchmark.json")
data = json.load(open(path, encoding="utf-8"))

new_examples = []

templates = [
    ("cuda_memory", "easy", "Why does torch.cuda.empty_cache not increase available GPU memory for my tensors?", "torch.cuda.empty_cache() does not increase the amount of GPU memory available to PyTorch tensors; it releases unoccupied cached memory held by the caching allocator so it can be used by other GPU applications.", ["https://pytorch.org/docs/stable/generated/torch.cuda.empty_cache.html", "https://pytorch.org/docs/stable/notes/cuda.html"]),
    ("cuda_memory", "medium", "How can I check peak CUDA memory usage during training?", "Use torch.cuda.memory_allocated, torch.cuda.max_memory_allocated, memory_reserved, and max_memory_reserved to inspect allocated and reserved GPU memory. Reset peak statistics before measuring a region.", ["https://pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html", "https://pytorch.org/docs/stable/notes/cuda.html"]),
    ("amp", "easy", "How do I use autocast for mixed precision training in PyTorch?", "Use torch.autocast or torch.cuda.amp.autocast around the forward pass and loss computation so selected operations run in lower precision automatically.", ["https://pytorch.org/docs/stable/amp.html"]),
    ("amp", "medium", "Why should I use GradScaler with float16 mixed precision?", "GradScaler helps prevent gradient underflow in float16 training by scaling the loss before backward and unscaling before optimizer updates.", ["https://pytorch.org/docs/stable/amp.html"]),
    ("dataloader", "easy", "What does num_workers do in a PyTorch DataLoader?", "num_workers controls how many subprocesses are used for data loading. num_workers=0 loads data in the main process; higher values use worker processes.", ["https://pytorch.org/docs/stable/data.html"]),
    ("dataloader", "medium", "When should I set pin_memory=True in DataLoader?", "pin_memory=True copies tensors into page-locked host memory, which can speed up host-to-GPU transfers when using CUDA.", ["https://pytorch.org/docs/stable/data.html"]),
    ("optimizer", "easy", "How do I clip gradients in PyTorch?", "Use torch.nn.utils.clip_grad_norm_ or clip_grad_value_ after loss.backward() and before optimizer.step() to limit gradient magnitude.", ["https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html"]),
    ("optimizer", "medium", "What is the correct order for optimizer.step and scheduler.step?", "For most PyTorch schedulers, call optimizer.step() before scheduler.step() during training unless a scheduler's documentation states otherwise.", ["https://pytorch.org/docs/stable/optim.html"]),
    ("dtype", "easy", "Why do I get expected scalar type Float but found Half?", "This usually means model parameters and input tensors use different dtypes. Move both model and input to compatible dtypes using .float(), .half(), or autocast.", ["https://pytorch.org/docs/stable/tensors.html", "https://pytorch.org/docs/stable/amp.html"]),
    ("dtype", "medium", "How do I create tensors with a specific dtype and device?", "Pass dtype= and device= when constructing tensors or use tensor.to(device=..., dtype=...) to move and cast existing tensors.", ["https://pytorch.org/docs/stable/tensors.html"]),
    ("distributed", "medium", "Why does DistributedDataParallel hang at startup?", "DDP can hang if ranks, world size, backend, or initialization method are misconfigured, or if not all processes enter the same collective operations.", ["https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html", "https://pytorch.org/docs/stable/distributed.html"]),
    ("distributed", "hard", "When should I use DistributedSampler with DDP?", "Use DistributedSampler so each process receives a distinct subset of the dataset. Call sampler.set_epoch(epoch) each epoch to shuffle consistently.", ["https://pytorch.org/docs/stable/data.html", "https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"]),
    ("batch_norm", "medium", "Why does BatchNorm behave differently in train and eval mode?", "In training mode BatchNorm uses batch statistics and updates running estimates. In eval mode it uses running_mean and running_var.", ["https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html"]),
    ("batch_norm", "medium", "What normalization layer should I use for very small batch sizes?", "GroupNorm is often suitable for small batches because it computes statistics over groups of channels and does not depend on batch dimension in the same way as BatchNorm.", ["https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html"]),
    ("config", "easy", "What is wrong if my YAML config sets batch_size to 512 on one GPU?", "A very large batch size can exceed GPU memory. Reduce batch_size, enable mixed precision, or use gradient accumulation.", ["https://pytorch.org/docs/stable/notes/cuda.html", "https://pytorch.org/docs/stable/amp.html"]),
    ("config", "medium", "Why is my model slow when num_workers is 0?", "With num_workers=0, data loading runs in the main training process and may bottleneck GPU training. Increase num_workers if the dataset pipeline is CPU-bound.", ["https://pytorch.org/docs/stable/data.html"]),
    ("general", "easy", "How do I move a model to GPU in PyTorch?", "Use model.to('cuda') or model.cuda(), and move input tensors to the same device before the forward pass.", ["https://pytorch.org/docs/stable/generated/torch.nn.Module.html", "https://pytorch.org/docs/stable/tensors.html"]),
    ("general", "easy", "Why do I get tensors on different devices error?", "The model and tensors must be on the same device. Move inputs, labels, and model parameters to the same CPU or CUDA device.", ["https://pytorch.org/docs/stable/tensors.html"]),
    ("general", "medium", "How do I save and load a PyTorch model checkpoint?", "Use torch.save to save state_dicts and torch.load plus load_state_dict to restore model and optimizer states.", ["https://pytorch.org/docs/stable/notes/serialization.html"]),
    ("general", "medium", "What is the difference between model.train() and model.eval()?", "model.train() enables training behavior such as Dropout and BatchNorm batch statistics. model.eval() switches modules to inference behavior.", ["https://pytorch.org/docs/stable/generated/torch.nn.Module.html"]),
    ("cuda_memory", "hard", "Why does memory_reserved stay high after tensors are deleted?", "PyTorch uses a CUDA caching allocator. Freed tensor memory may remain reserved by PyTorch for reuse even though it is no longer allocated to active tensors.", ["https://pytorch.org/docs/stable/notes/cuda.html"]),
    ("amp", "hard", "Why can mixed precision training produce NaNs?", "Some operations or losses can overflow or underflow in lower precision. GradScaler and autocast help, but numerically sensitive operations may need float32.", ["https://pytorch.org/docs/stable/amp.html"]),
    ("dataloader", "hard", "Why should DataLoader workers not return CUDA tensors?", "The PyTorch docs generally recommend not returning CUDA tensors from multi-process DataLoader workers because CUDA multiprocessing has subtleties. Use pinned memory instead.", ["https://pytorch.org/docs/stable/data.html"]),
    ("optimizer", "medium", "How do I use AdamW in PyTorch?", "Use torch.optim.AdamW with model parameters, learning rate, and weight_decay. AdamW decouples weight decay from gradient-based parameter updates.", ["https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html"]),
    ("optimizer", "hard", "How do I resume optimizer state from a checkpoint?", "Save optimizer.state_dict() with torch.save and restore it with optimizer.load_state_dict() when resuming training.", ["https://pytorch.org/docs/stable/optim.html", "https://pytorch.org/docs/stable/notes/serialization.html"]),
    ("dtype", "hard", "When should I use bfloat16 instead of float16?", "bfloat16 has a wider exponent range than float16 and can be more numerically stable on supported hardware, while still reducing memory and bandwidth.", ["https://pytorch.org/docs/stable/amp.html"]),
    ("distributed", "hard", "Why must all DDP processes call backward consistently?", "DistributedDataParallel synchronizes gradients across processes. If processes take different control-flow paths and do not participate in expected collectives, training may hang or error.", ["https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html"]),
    ("architecture", "medium", "What does scaled dot product attention compute?", "Scaled dot product attention computes attention weights from query-key similarity, applies softmax, and uses those weights to combine value vectors.", ["https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html"]),
    ("architecture", "medium", "What is MultiheadAttention in PyTorch?", "MultiheadAttention jointly attends to information from different representation subspaces using multiple attention heads.", ["https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html"]),
    ("config", "medium", "My config uses lr=10.0 for AdamW. Why is training unstable?", "An excessively large learning rate can cause unstable optimization. AdamW still requires an appropriate learning rate and scheduler for the task.", ["https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html", "https://pytorch.org/docs/stable/optim.html"]),
    ("config", "hard", "My config enables fp16 but disables GradScaler. What is the risk?", "Float16 gradients may underflow without gradient scaling. GradScaler is recommended for typical float16 mixed precision training.", ["https://pytorch.org/docs/stable/amp.html"]),
    ("general", "hard", "Why does torch.no_grad reduce memory during inference?", "torch.no_grad disables gradient calculation, reducing memory consumption because intermediate activations do not need to be saved for backward.", ["https://pytorch.org/docs/stable/generated/torch.no_grad.html"]),
    ("general", "medium", "When should I use torch.inference_mode instead of no_grad?", "torch.inference_mode disables autograd and additional view/version tracking overhead, making it useful for inference-only workloads.", ["https://pytorch.org/docs/stable/generated/torch.inference_mode.html"]),
    ("cuda_memory", "medium", "How can gradient checkpointing reduce memory usage?", "Gradient checkpointing trades compute for memory by not storing all intermediate activations during forward and recomputing them during backward.", ["https://pytorch.org/docs/stable/checkpoint.html"]),
    ("dataloader", "medium", "What does persistent_workers=True do?", "persistent_workers keeps DataLoader worker processes alive after one epoch, avoiding worker startup cost in later epochs.", ["https://pytorch.org/docs/stable/data.html"]),
    ("optimizer", "medium", "How do I zero gradients correctly?", "Call optimizer.zero_grad() before backpropagating the next batch. set_to_none=True can reduce memory operations and may improve performance.", ["https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html"]),
    ("dtype", "medium", "How do I avoid dtype mismatch between labels and CrossEntropyLoss?", "CrossEntropyLoss expects class indices as long tensors for the target in the common classification case, while logits are floating point.", ["https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html"]),
    ("general", "medium", "How do I debug exploding gradients?", "Inspect gradient norms and use gradient clipping such as torch.nn.utils.clip_grad_norm_ before optimizer.step().", ["https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html"]),
    ("general", "easy", "How do I set random seeds in PyTorch?", "Use torch.manual_seed and related random seeding utilities. For reproducibility with CUDA and DataLoader workers, additional deterministic settings may be needed.", ["https://pytorch.org/docs/stable/notes/randomness.html"]),
    ("distributed", "medium", "Which backend should I use for GPU distributed training?", "For CUDA GPU distributed training, NCCL is commonly the recommended backend; Gloo is often used for CPU training.", ["https://pytorch.org/docs/stable/distributed.html"]),
]

start = len(data) + 1
for i, (category, difficulty, query, answer, urls) in enumerate(templates, start=start):
    new_examples.append({
        "id": f"ex_{i:03d}",
        "query": query,
        "image_path": None,
        "log_snippet": query,
        "gold_answer": answer,
        "gold_source_urls": urls,
        "gold_chunk_ids": [],
        "category": category,
        "difficulty": difficulty,
    })

data.extend(new_examples)

with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"benchmark size: {len(data)}")
