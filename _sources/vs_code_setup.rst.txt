VSCode Setup
====================================================



We recommend a docker environment based on the most recent pytorch e.g.:

.. code-block:: bash

    FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
    RUN apt-get update && apt-get install -y wget openssh-client git-core bash-completion
    RUN wget -O /tmp/git-lfs.deb https://packagecloud.io/github/git-lfs/packages/ubuntu/focal/git-lfs_2.13.3_amd64.deb/download.deb && \
        dpkg -i /tmp/git-lfs.deb && \
        rm /tmp/git-lfs.deb
    RUN echo  'source /usr/share/bash-completion/completions/git' >> ~/.bashrc 
    CMD ["/bin/bash"]

This works seamlessly in combination with the VSCode DevContainer extention:

.. code-block:: json

    {
        "name": "Dev Container",
        "dockerFile": "Dockerfile",
        "runArgs": [
            "--network",
            "host",
            "--gpus",
            "all"
        ],
        "customizations": {
            "vscode": {
                "settings": {
                    "terminal.integrated.shell.linux": "/bin/bash"
                },
                "extensions": [
                    "ms-python.python"
                ]
            }
        }
    }

In VSCode, add this to your :file:`launch.json`:

.. code-block:: json

    {
        "name": "Torchrun Train and Eval",
        "type": "python",
        "request": "launch",
        "module": "torch.distributed.run",
        "env": {
            "CUDA_VISIBLE_DEVICES": "4,5"
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
            "config_files/config_lorem_ipsum.yaml",
        ],
        "console": "integratedTerminal",
        "justMyCode": true,
        "envFile": "${workspaceFolder}/.env",
        "cwd": "${workspaceFolder}/modalities"
    }

