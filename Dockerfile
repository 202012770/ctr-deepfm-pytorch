FROM python:3.10-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
# Override for restricted networks, e.g. in mainland China:
#   docker build \
#     --build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
#     --build-arg TORCH_FIND_LINKS=https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
#     --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn .
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ARG TORCH_FIND_LINKS=""

COPY pyproject.toml ./
COPY src ./src
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip --retries 10 --timeout 120 \
    && if [ -n "${TORCH_FIND_LINKS}" ]; then \
         /opt/venv/bin/pip install torch --no-deps --no-index \
           --find-links "${TORCH_FIND_LINKS}" --retries 10 --timeout 120; \
       else \
         /opt/venv/bin/pip install torch --index-url ${TORCH_INDEX_URL} \
           --retries 10 --timeout 120; \
       fi \
    && /opt/venv/bin/pip install filelock typing-extensions sympy networkx jinja2 fsspec \
         --index-url ${PIP_INDEX_URL} --retries 10 --timeout 120 \
    && /opt/venv/bin/pip install . --index-url ${PIP_INDEX_URL} --retries 10 --timeout 120

FROM python:3.10-slim

ARG APT_MIRROR=deb.debian.org

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PORT=7860 \
    CTR_PROJECT_ROOT=/app

RUN sed -i "s|deb.debian.org|${APT_MIRROR}|g" /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY configs ./configs
COPY models/bundle ./models/bundle

RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "uvicorn recsys.serving.api:app --host 0.0.0.0 --port ${PORT}"]
