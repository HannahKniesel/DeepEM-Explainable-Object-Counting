FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
ADD requirements.txt /reproducability/
ADD Dockerfile /reproducability/
RUN pip install -r /reproducability/requirements.txt
WORKDIR /workspace/
ENV CUBLAS_WORKSPACE_CONFIG=:16:8 