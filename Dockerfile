FROM python:3.10-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
COPY setup_nltk.py .
RUN python setup_nltk.py

# Copy all project files
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
