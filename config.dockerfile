# Use the official Python 3.12 image
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy requirements first to leverage Docker caching
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files to the container
#COPY . .

# Expose the Flask app's port (default is 5000)
EXPOSE 5001

# Command to run the application
CMD ["python", "app.py"]
