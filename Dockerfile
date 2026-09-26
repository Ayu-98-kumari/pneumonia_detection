# Hugging Face Space (Docker SDK) running the Streamlit demo.
FROM python:3.11-slim

# HF Spaces run containers as a non-root user (uid 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Install Python deps first (better layer caching). requirements.txt pins
# CPU-only torch via its --extra-index-url line.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# App code, models (LFS), bundled test data, and .streamlit/ theme.
COPY --chown=user . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
