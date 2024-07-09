# Changelog

| PR               | Type       | Ref. Issue(s) | Breaking Changes |PR Description|                                                                                  
|------------------|------------|---------------|------------------|------------------------------------------------------------------------------------------------|
| [#141](#pr-141-towards-stable-modalities-version)  | Bug Fix    |  [#129](https://github.com/Modalities/modalities/issues/129)         | **Yes**              | Towards stable modalities version                                                               |
| [#154](pr-154-manual-swiglu-implementation)  | Bug Fix    |  [#14](https://github.com/Modalities/modalities/issues/14)         | **Yes**              | Towards stable modalities version                                                               |
|    |   |           |        |                                                                |



## PR #141 Towards stable modalities version

This PR further stabilise the codebase and makes training more robust also w.r.t. loss spikes, which we fixed via scaled weight initialisation and an increased batch size in our experiments.
The PR also fixes all failing tests and adds a simple entrypoint for running cpu, single-gpu and multi-gpu tests. The PR contains multiple sub PRs. 

**General changes:**
* Bug fix: the model evaluation mode is now properly deactivated after evaluation (see PR [#131](https://github.com/Modalities/modalities/pull/131))
* Bug fix: Fixed the implementation of Pre-LN for GPT2 model (see PR [#136](https://github.com/Modalities/modalities/pull/136))
* Enhancement: Further mixed precision strategies; also added one matching MegatronLM's.
* Enhancement: Single, unified entrypoint for running cpu, single-gpu and multi-gpu tests. All tests fixed. (PR [#155](https://github.com/Modalities/modalities/pull/155))

**Breaking changes:** 
* Enhancement: Logging is now always based on #training steps and #consumed tokens (PR [#137](https://github.com/Modalities/modalities/pull/137))
   This change is a breaking change and the experiment configs need to adapated as shown [here](https://github.com/Modalities/modalities/pull/137/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4). 
* Enhancement: The model parameters are now grouped within the respective model. The optimizer can leverage these groups to e.g., only apply weight decay to non-layer-norm weights. See [here](https://github.com/Modalities/modalities/pull/139/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the necessary config changes. (PR [#139](https://github.com/Modalities/modalities/pull/139))
* Enhancement: We support now different attention implementations (manual, pytorch flash, DAO flash) See [here](https://github.com/Modalities/modalities/pull/138/files#diff-2bea5a6678ec91ea603cc2e80d17847360af5e9f7624c8e710f329ee1eb9b4f4) for the respective config changes. (PR [#138](https://github.com/Modalities/modalities/pull/138))
## PR #154 Manual SwiGLU implementation

This [PR](https://github.com/Modalities/modalities/pull/154) adds a manual SwiGLU implementation. The original one from xops was imcompatible with activation checkpointing (see issue [#14](https://github.com/Modalities/modalities/issues/14)) 

**General changes:**
* replaces xops swiglu imlementation with custom reimplementation

**Breaking changes:** 
* renaming of `fused_swiglu` to `swiglu` in `ActivationType` (see [here](https://github.com/Modalities/modalities/pull/154/commits/90fb3bd06a407333423cffeab486711e26ef8ddf) for the respective config changes)
