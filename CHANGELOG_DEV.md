# Changelog

| PR               | Type       | Ref. Issue(s) | Breaking Changes |PR Description|                                                                                  
|------------------|------------|---------------|------------------|------------------------------------------------------------------------------------------------|
| [#154](pr-154-manual-swiglu-implementation)  | Bug Fix    |  [#14](https://github.com/Modalities/modalities/issues/14)         | **Yes**              | Towards stable modalities version                                                               |
|    |   |           |        |                                                                |



## PR #154 Manual SwiGLU implementation

This [PR](https://github.com/Modalities/modalities/pull/154) adds a manual SwiGLU implementation. The original one from xops was imcompatible with activation checkpointing (see issue [#14](https://github.com/Modalities/modalities/issues/14)) 

**General changes:**
* replaces xops swiglu imlementation with custom reimplementation

**Breaking changes:** 
* renaming of `fused_swiglu` to `swiglu` in `ActivationType` (see [here](https://github.com/Modalities/modalities/pull/154/commits/90fb3bd06a407333423cffeab486711e26ef8ddf) for the respective config changes)
