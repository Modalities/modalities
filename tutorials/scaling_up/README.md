# Conducting Scaling Experiments with Modalities
In this tutorial, we explain how to conduct scaling experiments with Modalities, i.e., how to scale up the number of ranks while varying hyperparameters to maximize throughput. The goals are twofold. Firstly, we want to showcase linear scalability up to a targeted number of ranks and secondly, we want to find the optimal hyperparameters that maximize the throughput for a given number of ranks. The latter one is the configuration we typically select for the final model training.

This tutorial follows the file structure below.
The `scripts` folder contains scripts to run the benchmarks on a single node or on a cluster, while the `configs` folder contains the respective sweep configurations. The `data` folder contains the  training data used for the experiments, and the `analysis` folder contains Jupyter notebooks for analyzing the results. The `experiments` folder contains the configs and respective results of the experiments. The folder follows a nested file structure with the convention `YYYY-MM-DD__HH-MM-SS_<sweep_hash>/<rank>/<experiment_hash__YYYY-MM-DD__HH-MM-SS>`, where `<sweep_hash>` is the hash of the sweep configuration, `<rank>` is the number of ranks used in the experiment, and `<experiment_hash>` is the config file hash of the individual experiment.
The config file hash is also used as the name for the experiment config. 

```txt
├── README.md
├── analysis
│   └── throughput_analysis.ipynb
├── configs
│   ├── sweep_8B_fsdp2.yaml
│   └── sweep_config.yaml
├── data
│   └── lorem_ipsum_long.pbin
├── experiments
│   ├── 2025-07-16__17-28-03_970fedec
│   │   ├── 2
│   │   │   ├── abcd1234_2025-07-16__17-30-00
│   │   │   │   ├── abcd1234.yaml
│   │   │   │   ├── abcd1234.yaml.resolved
│   │   │   │   └── evaluation_results.jsonl
│   │   │   ├── abcd1234_2025-07-24__21-23-04
│   │   │   │   ├── abcd1234.yaml
│   │   │   │   ├── abcd1234.yaml.resolved
│   │   │   │   └── evaluation_results.jsonl
│   │   │   └── abcd1234_2025-07-24__21-23-37
│   │   │       ├── abcd1234.yaml
│   │   │       ├── abcd1234.yaml.resolved
│   │   │       └── evaluation_results.jsonl
│   │   ├── 4
│   │   │   ├── abcd1234_2025-07-16__17-30-00
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   └── ...
└── scripts
    ├── create_sweep_configs.sh
    ├── hpc
    │   └── leonardo
    │       ├── job.sbatch
    │       └── submit_sweep.sh
    └── single_node
        └── run_single_node.sh
```

## Running the Experiments
The experiments can be run either on a single node or on a cluster.
Before running experiments, we need to create the experiment configs from the sweep configuration. The sweep config contains the typical training config but is enriched with a sweep configuration that defines the hyperparameters to sweep over.
A minimal example could be to sweep over the local batch size and the model's sequence length, as shown below.

```yaml
sweep:
  local_train_micro_batch_size: [1, 2, 4]
  sequence_length: [256, 512]

...

<training config>
```

It is important that these hyperparameters are referred to within the training config. This is achieved by resolving references via omega conf. 
For instance the `step_profile` in the `settings` component can refer to these hyperarameters as follows
```yaml
settings:
  step_profile:
    gradient_accumulation_steps: 1
    local_train_micro_batch_size: ${sweep.local_train_micro_batch_size}
    sequence_length: ${sweep.sequence_length}
```
effectively ingesting the hyperparameters into the training config.

**IMPORTANT:** It is the user's responsibility to ensure that the hyperparameters are correctly referenced in the training config and there are no automatic consistency checks.

To create the experiment configs from the sweep configuration, we execute the `create_sweep_config.sh`. 

```sh
sh scripts/create_sweep_configs.sh
```
Note that the script has the sweep config hardcoded and if you want to use a different sweep config, you need to modify the script accordingly.
Alternatively you can run the experiment config creation via modalities directly, e.g., 

```sh
modalities benchmark prepare_sweep_configs --sweep_config_path ../configs/sweep_config.yaml --output_dir ../experiments --world_sizes 2,4,8
```

### Single Node Benchmarking
To run the experiments on a single node, you can use the script `scripts/single_node/run_single_node.sh`.
This script will run the experiments on a single node with the specified number of ranks and the specified sweep configuration.
You can specify the number of ranks and the experiments folder in the script, prior to running it.

```sh
sh scripts/single_node/run_single_node.sh <path to sweep folder>
```

This script will run all configs sequentially on the specified number of ranks and collect all the results data (e.g., throughput, potential errors) in the `experiments` folder.

For an end to end script, also see `scripts/run_scaling_up_single_node.sh`. 

### HPC Cluster Benchmarking
To run the experiments on a SLURM cluster, you can use the script `scripts/hpc/leonardo/submit_sweep.sh`, which submits an sbatch job to the cluster for each node configuration. Each sbatch job will run all the experiments on the specified number of ranks and the specified sweep configuration.

The sbatch job is defined in the file `scripts/hpc/leonardo/job.sbatch`. Note that both the `submit_sweep.sh` and the `job.sbatch` are dedicated to the Leonardo HPC cluster, so you will need to adapt environmental setttings (e.g., python env path  or slurm partition) to your specific HPC cluster. Respective hints are given by "TODO" comments in both scripts. 

## Evaluation 
To evaluate the results of the experiments, you can use the Jupyter notebook `analysis/throughput_analysis.ipynb`.
By specifying the `experiments` folder, the notebook will load all the results and plot the throughput for each rank and hyperparameter configuration.