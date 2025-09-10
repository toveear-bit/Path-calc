# Dockerfile for Hugging Face Spaces (Flask app)
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860
RUN pip install gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]

