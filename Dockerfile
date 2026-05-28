FROM python:3.13-slim-bookworm

EXPOSE 8501

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY . /app

RUN uv sync --frozen --no-dev

ENTRYPOINT ["uv", "run", "streamlit", "run", "transition_finder.py", "--server.port=8501", "--server.address=0.0.0.0"]