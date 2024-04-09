FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime
RUN apt update && apt install git -y
RUN conda install nvidia/label/cuda-12.1.0::cuda-toolkit -y
RUN pip install ninja  # actually extra "install_helper", but running "pip install .[install_helper] causes flash-attn installation
RUN pip install flash-attn --no-build-isolation  # own step for docker build caching
COPY ./ /app
RUN cd /app && pip install .[tests]

WORKDIR /app
