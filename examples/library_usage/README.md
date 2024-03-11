# Running Modalities like a package

Modalities can be used in a library fashion by installing the package via `pip`, as described in the [README](https://github.com/Modalities/modalities?tab=readme-ov-file#installation). The framework allows for the addition of custom components to the registry at runtime without necessitating any code changes to modalities. This functionality is achieved in Modalities with the introduction of a component registry, containing all the internal components (e.g., Dataloader, Loss function etc.). To support the addition of custom components (e.g., new model architectures) at runtime, Modalities exposes a function endpoint adding custom components to the internal registry. 

A  typical use case for running Modalities in package-like fashion would be to have a custom model implemented in a repository parallel to modalities. To train the model, we would register the model class and its config class  within Modalities' registry and additionally provide the typical training config (see [here](https://github.com/Modalities/modalities/blob/main/examples/getting_started/example_config.yaml) for an example) that also references the new model. Since modalities is aware of the model and config class, the model can be built from the config YAML file and used for training.

## Concrete Example

Given the explanation above, we now provide a minimal dummy example of the process of implementing, registering and instantiating a custom component via the example of a custom collate function. 
The full example code can be found [here](https://github.com/Modalities/modalities/tree/hierarchical_instantiation/examples/library_usage).

The code for the custom collate function, its config and registering is implemented in 
[main.py](https://github.com/Modalities/modalities/blob/hierarchical_instantiation/examples/library_usage/main.py). Firstly, the script implements the custom collate function by first defining the config that parameterizes the collate function. Here, we took the two attributes from the original [GPT2LLMCollateFnConfig]() and added the custom field `custom_attribute`.

```python
 class CustomGPT2LLMCollateFnConfig(BaseModel): 
     sample_key: str 
     target_key: str 
     custom_attribute: str 
```

The collate function implements the `CollateFnIF` interface. Its constructor expects the attributes from the previously defined `CustomGPT2LLMCollateFnConfig`. Since this is only a minimal example to demonstrate the registering of custom components, we just print the custom attribute without adding any senseful functionality. 

```python
class CustomGPT2LLMCollateFn(CollateFnIF): 
     def __init__(self, sample_key: str, target_key: str, custom_attribute: str): 
         self.sample_key = sample_key 
         self.target_key = target_key 
         self.custom_attribute = custom_attribute 
  
     def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> DatasetBatch: 
         sample_tensor = torch.stack([torch.tensor(d[self.sample_key]) for d in batch]) 
         samples = {self.sample_key: sample_tensor[:, :-1]} 
         targets = {self.target_key: sample_tensor[:, 1:]} 
  
         print(f"Custom attribute: {self.custom_attribute}") 
  
         return DatasetBatch(targets=targets, samples=samples)
```

Given `CustomGPT2LLMCollateFnConfig` and `CustomGPT2LLMCollateFnConfig`, we register the new component via `add_custom_component(...)` by providing the respective component key and variant key together with the two previously defined classes. Note that even though the `component_key` and `variant_key` are in principle arbitrary, it is good practice to follow the patterns used for the internal components, as defined in [components.py](https://github.com/Modalities/modalities/blob/hierarchical_instantiation/src/modalities/registry/components.py#L64).

```python
def main(): 
     # load and parse the config file 
     config_file_path = Path("config_lorem_ipsum.yaml") 
     config_dict = load_app_config_dict(config_file_path) 
  
     # instantiate the Main entrypoint of modalities by passing in the config 
     modalities_main = Main(config_dict=config_dict) 
  
     # add the custom component to modalities 
     modalities_main.add_custom_component( 
         component_key="collate_fn", 
         variant_key="custom_gpt_2_llm_collator", 
         custom_component=CustomGPT2LLMCollateFn, 
         custom_config=CustomGPT2LLMCollateFnConfig, 
     ) 
     # run the experiment 
     modalities_main.run() 
```

Lastly, we add the `collate_fn` to the [example YAML config](https://github.com/Modalities/modalities/blob/hierarchical_instantiation/examples/library_usage/config_lorem_ipsum.yaml) with the the new collator.
```yaml
collate_fn:  
  component_key: collate_fn
  variant_key: custom_gpt_2_llm_collator
  config:
    sample_key: ${settings.referencing_keys.sample_key}
    target_key: ${settings.referencing_keys.target_key}
    custom_attribute: "custom_value"
```

Given the changes above, we are now ready to run the training by executing the following bash command in the example directory.
```sh
CUDA_VISIBLE_DEVICES=0,1 torchrun --rdzv-endpoint localhost:29504 --nnodes 1 --nproc_per_node 2 main.py
```


