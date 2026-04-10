FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip install Flask --break-system-packages

RUN mkdir -p out \
    && curl -fsSL "https://github.com/Azimkin/LayoutDetectC1/releases/download/C2.1/LayoutDetectC2_1.pth" \
        -o out/LayoutDetectC1.pth

COPY api.py model.py ./

EXPOSE 5000

CMD ["python", "api.py"]
