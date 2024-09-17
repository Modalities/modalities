# Getting started with Modalities in 15 minutes

Throughout the tutorial, we will use the Jupyter Notebook `modalities_demo.ipynb` to guide us through the process of getting started with Modalities. The notebook is located in the root directory of the tutorial, along with the `configs` and `data` directories. The `configs` directory contains configuration files for the model pretraining and tokenization, while the `data` directory contains subdirectories for storing checkpoints, preprocessed data, raw data, and tokenizer-related files.

```text
└── getting_started_15mins                 # Root directory for the tutorial
    ├── modalities_demo.ipynb              # Jupyter Notebook which we will be using for the tutorial.
    ├── configs                      
    │   ├── pretraining_config.yaml        # Config file for the model pretraining
    │   └── tokenization_config.yaml       # Config file for tokenization
    └── data                         
        ├── checkpoints                    # Dir where model and optimizer checkpoints  are stored.
        │   └── <checkpoints>        
        ├── preprocessed                   # Dir containing preprocessed training and evaluation data.
        │   └── <files>              
        ├── raw                      
        │   └── fineweb_edu_num_docs_483606.jsonl   # JSONL file containing raw data for training and evaluation.
        └── tokenizer                
            ├── tokenizer.json             # JSON file defining the tokenizer model, including token mappings.
            └── tokenizer_config.json      # Config file specifying all tokenizer settings
```


To start the tutorial check out the Jupyter Notebook `modalities_demo.ipynb` and follow the instructions provided in the notebook.
If you don't have Jupyter Notebook installed in your python environment, you can install it by running the following command:

```bash
pip install jupyterlab
```