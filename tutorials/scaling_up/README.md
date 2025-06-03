# Conducting Scaling Experiments with Modalities
In this tutorial, we showcase how the profiling and benchmark tooling within Modalities can be used to determine the copnfigurations that maximize throughput for a varying number of
ranks (i.e., scaling benchmarks).

Generally, there are two types of scaling benchmarkings. In the first case, one wants demonstrate linear scalability of the framework on a given HPC cluster. By varying parameters of the configuration 
(e.g., batch size and activation checkpointing variant), we determine the maximum throughput for different number of ranks. Plotting the maximum throughput against the number of ranks should yield a linear relationship.

In the second case, which can be regarded a simplification of the first one, we only want to determine the optimal settings for a fixed number of ranks. This is typically the case, when you received
a GPU allocation on an HPC cluster and want to maximize the efficiency of your training run. 


In this tutorial, we cover both cases in the jupyter notebook `scaling_tutorial.ipynb`.

