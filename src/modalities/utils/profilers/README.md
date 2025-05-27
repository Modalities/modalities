# Modalities Profiling

## Activation Checkpointing

# Example 1:  Selective Activation Checkpointing

With **selective OP activation checkpointing (SAC)**, we reduce the memory footprint at the expense of increased compute by saving **only the activations of selected ATen ops**.  
To make SAC effective, we focus on ATen ops that are **memory-intensive yet fast to recompute**.

In Modalities, we follow an iterative process to determine which ops to save:

1. **Establish a baseline**  
   Run a training step with the maximum batch size that fits in memory *without* any activation checkpointing. Record the total memory footprint and runtime.

2. **Profile the forward pass**  
   Use the PyTorch profiler to identify compute-heavy ops. These are typically associated with attention (e.g., flash attention, matmuls, etc.).

3. **Estimate op memory cost indirectly**  
   Because PyTorch's profiler doesn't always attribute memory allocations to the corresponding ATen ops, we rely on **step-level peak memory tracking** (`torch.cuda.max_memory_allocated`) instead.  
   We **experiment with different save lists in isolation**, and compare the resulting memory footprint and runtime against the baseline from step 1.
