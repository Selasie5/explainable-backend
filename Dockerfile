# Use official Python base image
FROM 3.10.18

# Set working directory
WORKDIR /app

# Copy requirements first and install (better for caching)
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Make sure necessary folders exist
RUN mkdir -p cache explanations

# Expose FastAPI's default port
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
