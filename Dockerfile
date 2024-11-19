# Use an official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install build dependencies for TensorFlow and other scientific packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libatlas-base-dev \
    libhdf5-dev \
    libeigen3-dev \
    python3-dev \
    build-essential \
    wget \
    curl \
    libomp-dev \
    && apt-get clean

# Upgrade pip and setuptools to ensure smooth package installation
RUN pip install --upgrade pip setuptools wheel

# Install TensorFlow (you can specify a version like tensorflow==2.11.0)
RUN pip install tensorflow

# Install other required dependencies from the requirements file
COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

# Copy the application source code
COPY src/ src/

# Copy the data folder
COPY data/ data/

# Set the default command to run your app
CMD ["python", "src/main.py"]
