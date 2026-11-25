FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN useradd --system --uid 10001 gateway
USER 10001

EXPOSE 8080

# A streamed answer can take a minute or two: give in-flight requests time to finish on shutdown.
CMD ["uvicorn", "app.main:build", "--factory", "--host", "0.0.0.0", "--port", "8080", \
     "--no-server-header", "--proxy-headers", "--forwarded-allow-ips", "*", \
     "--timeout-keep-alive", "75", "--timeout-graceful-shutdown", "110"]
