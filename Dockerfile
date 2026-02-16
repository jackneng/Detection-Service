FROM python:3.10-slim
# FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install fastapi uvicorn nudenet pillow transformers python-multipart
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
