FROM pytorch/pytorch:2.11.0-cuda13.0-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASK_HOST=0.0.0.0 \
    FLASK_PORT=5000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libgomp1 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv venv

RUN venv/bin/pip3 install Flask

RUN mkdir -p out \
    && curl -fsSL "https://github.com/Azimkin/LayoutDetectC1/releases/download/%D0%A11/LayoutDetectC1.pth" \
        -o out/LayoutDetectC1.pth

COPY api.py model.py ./

EXPOSE 5000

CMD ["python", "api.py"]
