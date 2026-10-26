FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/deepocr \
    HF_HOME=/home/deepocr/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates libgomp1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin deepocr \
    && mkdir -p /home/deepocr/.paddleocr /home/deepocr/.cache/paddle /home/deepocr/.cache/huggingface \
    && chown -R deepocr:deepocr /home/deepocr

COPY web-requirements.txt ./web-requirements.txt
# CPU-only PyTorch keeps the image small; the trained CTC models run on CPU.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r web-requirements.txt

# The API reuses the research packages (recognisers, preprocessing, layout, post-processing).
COPY api ./api
COPY configs ./configs
COPY datasets ./datasets
COPY detection ./detection
COPY evaluation ./evaluation
COPY layout ./layout
COPY models ./models
COPY postprocessing ./postprocessing
COPY preprocessing ./preprocessing
COPY utils ./utils
COPY checkpoints ./checkpoints
COPY results ./results
COPY samples ./samples

USER 10001:10001
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
