VSCode Setup
===========

**EDIT "docs/source/vs_code_setup.rst" IN ORDER TO MAKE CHANGES HERE**

In VSCode, add this to your :file:`launch.json`:

.. code-block:: json

  {
      "name": "Torchrun Main",
      "type": "python",
      "request": "launch",
      "module": "torch.distributed.run",
      "env": {
          "CUDA_VISIBLE_DEVICES": "0"
      },
      "args": [
          "--nnodes",
          "1",
          "--nproc_per_node",
          "2",
          "--rdzv-endpoint=0.0.0.0:29503",
          "src/modalities/__main__.py",
          "run",
          "--config_file_path",
          "config_files/config.yaml",
      ],
      "console": "integratedTerminal",
      "justMyCode": true,
      "envFile": "${workspaceFolder}/.env"
  }

