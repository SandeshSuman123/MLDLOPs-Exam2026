FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir \
    transformers \
    torch \
    sentencepiece \
    sacrebleu \
    sacremoses

# Copy scripts
COPY translate.py .
COPY evaluate.py .

# Copy input files
COPY input.txt .
COPY output.rtf .


CMD ["sh", "-c", "python translate.py && python evaluate.py"]