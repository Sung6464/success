FROM python:3.12-slim

WORKDIR /app

# Copy agent code
COPY agents/success-stories-retrival-system /app

# Copy shared package
COPY packages/download_response /packages/download_response

RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r /app/requirements.txt

# Install shared package properly
RUN pip install -e /packages/download_response

# Expose port used by Uvicorn
EXPOSE 8000

# Run the FastAPI app using uvicorn
CMD ["uvicorn", "main:http_app", "--host", "0.0.0.0", "--port", "8000", "--no-server-header"]
