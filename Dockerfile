FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY heart_disease_model.pkl .
COPY scaler.pkl .

EXPOSE 5000

CMD ["python", "app.py"]