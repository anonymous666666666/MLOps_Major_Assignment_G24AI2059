# Use a lightweight Python base
FROM python:3.11-slim

WORKDIR /app

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app code and model
COPY app.py .
COPY templates/ ./templates/
# Copy your saved model file (ensure savedmodel.pth is in repo root)
COPY savedmodel.pth .

# Expose port
EXPOSE 5000

# default env
ENV MODEL_PATH=/app/savedmodel.pth
ENV PORT=5000

# Run
CMD ["python", "app.py"]