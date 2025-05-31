FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies including git
RUN yum install -y git gcc gcc-c++ make openssl-devel --setopt=install_weak_deps=False && \
    yum clean all && \
    rm -rf /var/cache/yum

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_DISABLE_JIT=1

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install torch==2.0.1 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers soundfile boto3 && \
    pip install --no-deps git+https://github.com/openai/whisper.git@v20231117

COPY app/ ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
