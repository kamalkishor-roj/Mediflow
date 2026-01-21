FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY src ./src
COPY mlruns ./mlruns

EXPOSE 8000
EXPOSE 8501
