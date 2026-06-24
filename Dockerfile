FROM python:3.10-slim

WORKDIR /app

COPY pyproject.toml .
# Install dependencies from pyproject.toml (using a method like pip install .)
# For simplicity with standard pip:
COPY requirements.txt . 
RUN pip install -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY prompts/ prompts/

EXPOSE 8501

# CMD ["streamlit", "run", "src/app.py", "--server.address=0.0.0.0"]
# Add flags for Azure compatibility
CMD ["streamlit", "run", "src/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false"]