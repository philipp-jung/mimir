# Use the official Python 3.7 image as the base image
FROM python:3.7

# Copy the requirements.txt and main.py files into the container's working directory
COPY src/ /src/

# Set the working directory inside the container
WORKDIR /src

# Install the Python dependencies specified in requirements.txt
RUN pip install --upgrade pip setuptools
RUN pip install --no-cache-dir -r requirements.txt

# Run the Python script main.py
CMD ["python", "main.py"]
