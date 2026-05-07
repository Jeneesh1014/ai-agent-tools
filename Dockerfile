FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

ENV PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p chroma_db logs data/documents

EXPOSE 8000 7860

CMD ["python", "-m", "app.run"]