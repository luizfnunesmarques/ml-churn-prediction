# Use an official Python runtime as a parent image
FROM python:3.8.18

# Set the working directory in the container
WORKDIR /app

# Mount a volume for local file synchronization
VOLUME /app

# Copy the current directory contents into the container at /app
COPY . /app

RUN pip3 install -r requirements_py3.8.txt

# Run your command or application
CMD ["python", "churn_script_logging_and_tests.py"]
