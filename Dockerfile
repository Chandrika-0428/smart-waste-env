FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --no-cache-dir

COPY pyproject.toml ./
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]