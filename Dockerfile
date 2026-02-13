FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r requirements.txt

ARG INSTALL_DETECTRON2=1
RUN if [ "$INSTALL_DETECTRON2" = "1" ]; then \
        pip install --no-cache-dir "git+https://github.com/facebookresearch/detectron2.git"; \
    fi

COPY . .

EXPOSE 8501

CMD ["sh", "scripts/start_server.sh"]
