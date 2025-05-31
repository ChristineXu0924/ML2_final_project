FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies
RUN yum install -y gcc gcc-c++ make ffmpeg libsndfile wget unzip git

# Set work directory
WORKDIR /opt

# Create layer structure
RUN mkdir -p python

# Install Python packages into /opt/python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.0.1 \
        torchaudio==2.0.2 \
        transformers \
        boto3 \
        git+https://github.com/openai/whisper.git@v20231117 \
        -t python/

# Package the layer
RUN cd /opt && zip -r /whisper_layer.zip python
