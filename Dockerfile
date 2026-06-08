FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port 7860 (required by Hugging Face Spaces)
EXPOSE 7860

ENV TF_USE_LEGACY_KERAS=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=-1
ENV PORT=7860

CMD ["python", "app.py"]
