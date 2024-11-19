# Use a lightweight Python base image
FROM python:3.11-alpine

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements/base.txt requirements/base.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/base.txt

# Copy the application source code
COPY src/ src/

# Copy the /data folder into the container
COPY data/ data/

# Specify the default command to run your Python application
CMD ["python", "src/main.py"]
