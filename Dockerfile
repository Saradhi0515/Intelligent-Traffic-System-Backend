FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create necessary directories for data storage
# Note: In HF Spaces, only /tmp is writable usually, but we can write to app dir if not persistent?
# Actually, HF Spaces allows writing to the working dir, but it's ephemeral.
RUN mkdir -p Data/ANPR-ATCC/Results/Interpolated_Results
RUN mkdir -p Data/Accident-Detection/Results
RUN mkdir -p Models

# Create the user with UID 1000
RUN useradd -m -u 1000 user

# Set permissions for the non-root user (ID 1000) used by HF Spaces
RUN chown -R 1000:1000 /app

# Switch to non-root user
USER 1000

# Expose the port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Start the application with Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app", "--timeout", "600"]
