# Base image with GPU support
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

# Set the working directory
WORKDIR /app

# Set the geographical area
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Argentina/Buenos_Aires

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg

# Copy the requirements.txt file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (change it to the appropriate port your app uses)
EXPOSE 8501
