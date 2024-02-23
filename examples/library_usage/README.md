# Library Usage

Modalities can be used in a library fashion by installing the package via `pip`, as described in the [README](https://github.com/Modalities/modalities?tab=readme-ov-file#installation) and adding custom components at runtime to the registry without necessitating any code changes to modalities.

A  typical use case would be to have a custom model implemented in a repository parallel to modalities. To train the model, we would register the model class and its config class  within Modalities' registry and additionally provide the typical training config (see here, for an example) that also references the new model. Since modalities is aware of the model and config class, the model can be built from the config YAML file and used for training.

## Concrete Example

Given the explanation above, we now provide a minimal example of the process of implementing, registering and instantiating a custom component via the example of a custom collate function. 

The `main.py` script below shows a minimal example 


https://github.com/Modalities/modalities/blob/23960207055613fcf2f402e9d2519e3f5fe5f40f/examples/library_usage/main.py
