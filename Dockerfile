FROM python:3.13-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git curl gcc g++ libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV PATH="/opt/poetry/bin:${PATH}"

RUN curl -sSL https://install.python-poetry.org | python3 -

# Python path
ENV PYTHONPATH=/app/src

# Copy project
COPY . /app

# ✅ CORRECT (same as BA / QE)
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi  --no-root

# Runtime
RUN pip install gunicorn uvicorn

# ✅ FIX for hyphen folder
WORKDIR /app/src/success-stories-retrival-system

EXPOSE ${PORT:-8000}

CMD ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker main:http_app --bind 0.0.0.0:${PORT:-8000} --timeout 120"]
