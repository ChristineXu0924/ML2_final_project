FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies including tar (for ffmpeg install)
RUN yum install -y git gcc gcc-c++ make openssl-devel tar xz --setopt=install_weak_deps=False && \
    yum clean all && \
    rm -rf /var/cache/yum

# Install static ffmpeg manually
RUN curl -L https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
    | tar -xJ && \
    cp ffmpeg-*-static/ffmpeg /usr/local/bin/ && \
    chmod +x /usr/local/bin/ffmpeg && \
    rm -rf ffmpeg-*-static

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/root/.cache \
    TRANSFORMERS_CACHE=/root/.cache \
    TORCH_HOME=/root/.cache \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    NUMBA_DISABLE_JIT=1

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install numba==0.57.1 && \
    pip install transformers boto3 && \
    pip install --no-deps git+https://github.com/openai/whisper.git@v20231117 && \
    pip install tiktoken && \
    python3 -c "import whisper; whisper.load_model('base')" && \
    rm -rf ~/.cache/pip

COPY app/ ${LAMBDA_TASK_ROOT}

CMD ["lambda_function.lambda_handler"]
