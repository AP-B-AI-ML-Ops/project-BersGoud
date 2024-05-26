FROM python:3.9-slim

WORKDIR /app

# Copy requirements.txt from the root directory to the working directory
COPY ../../requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app.py"]
