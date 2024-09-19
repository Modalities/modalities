.. role:: python(code)
   :language: python

Configuration
========================================================================

Training config is defined in yaml formatted files. See :file:`data/config_lorem_ipsum.yaml`. These configs are very explicit specifying all training parameters to keep model trainings as transparent and reproducible as possible. Each config setting is reflected in pydantic classes in :file:`src/modalities/config/*.py`. In the config you need to define which config classes to load in field type_hint. This specifies the concrete class. A second parameter, config, then takes all the constructor arguments for that config class. This way it is easy to change i.e. DataLoaders while still having input validation in place.

Pydantic and ClassResolver
------------------------------------------------------------------------

The mechanismn introduced to instantiate classes via :python:`type_hint` in the :file:`config.yaml`, utilizes 

1) Omegaconf to load the config yaml file
2) Pydantic for the validation of the config
3) ClassResolver to instantiate the correct, concrete class of a class hierarchy.

Firstly, Omegaconf loads the config yaml file and resolves internal refrences such as `${subconfig.attribue}`. 

Then, Pydantic validates the whole config as is and checks that each of the sub-configs are :python:`pydantic.BaseModel` classes.
For configs, which allow different concrete classes to be instantiated by :python:`ClassResolver`, the special member names :python:`type_hint` and :python:`config` are introduced.
With this we utilize Pydantics feature to auto-select a fitting type based on the keys in the config yaml file.

:python:`ClassResolver` replaces large if-else control structures to infer the correct concrete type with a :python:`type_hint` used for correct class selection:

.. code-block:: python

  activation_resolver = ClassResolver(
    [nn.ReLU, nn.Tanh, nn.Hardtanh],
    base=nn.Module,
    default=nn.ReLU,
  )
  type_hint="ReLU"
  activation_kwargs={...}
  activation_resolver.make(type_hint, activation_kwargs),


In our implmentation we go a step further, as both,

* a :python:`type_hint` in a :python:`BaseModel` config must be of type :python:`modalities.config.lookup_types.LookupEnum` and 
* :python:`config` is a union of allowed concrete configs of base type :python:`BaseModel`. 

:python:`config` hereby replaces :python:`activation_kwargs` in the example above, and replaces it with pydantic-validated :python:`BaseModel` configs.

With this, a mapping between type hint strings needed for `class-resolver`, and the concrete class is introduced, while allowing pydantic to select the correct concrete config:

.. code-block:: python

  from enum import Enum
  from typing import Annotated
  from pydantic import BaseModel, PositiveInt, PositiveFloat, Field
  
  class LookupEnum(Enum):
      @classmethod
      def _missing_(cls, value: str) -> type:
          """constructs Enum by member name, if not constructable by value"""
          return cls.__dict__[value]
  
  class SchedulerTypes(LookupEnum):
      StepLR = torch.optim.lr_scheduler.StepLR
      ConstantLR = torch.optim.lr_scheduler.ConstantLR
  
  class StepLRConfig(BaseModel):
      step_size: Annotated[int, Field(strict=True, ge=1)]
      gamma: Annotated[float, Field(strict=True, ge=0.0)]
  
  
  class ConstantLRConfig(BaseModel):
      factor: PositiveFloat
      total_iters: PositiveInt
  
  
  class SchedulerConfig(BaseModel):
      type_hint: SchedulerTypes
      config: StepLRConfig | ConstantLRConfig

To allow a user-friendly instantiation, all class resolvers are defined in the :python:`ResolverRegistry` and :python:`build_component_by_config` as convenience function is introduced. Dependecies can be passed-through with the :python:`extra_kwargs` argument:

.. code-block:: python

  resolvers = ResolverRegister(config=config)
  optimizer = ...  # our example dependency
  scheduler = resolvers.build_component_by_config(config=config.scheduler, extra_kwargs=dict(optimizer=optimizer))

To add a new resolver use :python:`add_resolver`, and the corresponding added resolver will be accessible by the register_key given during adding.

For access use the :python:`build_component_by_key_query` function of the :python:`ResolverRegistry`.



