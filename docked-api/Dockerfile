FROM python:3.8.10-slim

WORKDIR /app

# System dependencies for LightGBM
RUN apt-get update && apt-get install -y libgomp1

# Copy and install Python dependencies
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app/ .

EXPOSE 8000

CMD ["uvicorn", "app_main:app", "--host", "0.0.0.0", "--port", "8000"]
