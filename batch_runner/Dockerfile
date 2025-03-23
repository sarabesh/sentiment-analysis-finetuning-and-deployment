FROM python:3.13-slim-bookworm

# Set the working directory inside the container
WORKDIR /code

# Copy the necessary files
COPY ./requirements.txt /code/requirements.txt
# Install dependencies
RUN pip install --no-cache-dir -r /code/requirements.txt

COPY ./app /code/app


# Define the command to run the script
CMD ["python", "-u","/code/app/batch_inference_job.py"]
