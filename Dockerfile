FROM python:3.11

WORKDIR /app

COPY requirements.txt .

ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN pip --no-cache-dir install -r requirements.txt
