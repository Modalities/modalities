Welcome to Modalities' documentation!
======================================================================

We propose a novel training framework for Multimodal Large Language Models (LLMs) that prioritizes code readability and efficiency. 
The codebase adheres to the principles of "clean code," minimizing Lines of Code (LoC) while maintaining extensibility.
A single, comprehensive configuration file enables easy customization of various model and training parameters.

A key innovation is the adoption of a PyTorch-native training loop integrated with the Fully Sharded Data Parallelism (FSDP) technique.
FSDP optimizes memory usage and training speed, enhancing scalability for large-scale multimodal models.
By leveraging PyTorch's native capabilities, our framework simplifies the development process and promotes ease of maintenance.

The framework's modular design facilitates experimentation with different multimodal architectures and training strategies.
Users can seamlessly integrate diverse datasets and model components, allowing for comprehensive exploration of multimodal learning tasks. 
The combination of clean code, minimal configuration, and PyTorch-native training with FSDP contributes to a user-friendly and efficient platform for developing state-of-the-art multimodal language models.

.. note::

   This project is under active development.

.. toctree::
   :caption: Getting Started

   quickstart
   configuration
   model_cards
   benchmarking
   known_issues

.. toctree::
   :caption: Datasets

   memmap

.. toctree::
   :caption: Entrypoints

   entrypoints

.. toctree::
   :caption: VSCode Setup

   vs_code_setup


.. toctree::
   :caption: Future Work

   future_work

.. toctree::
   :caption: API

   api/modules