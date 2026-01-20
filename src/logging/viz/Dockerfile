FROM python:3.10-slim

WORKDIR /app

# First copy the entire src directory
COPY src/ /app/src/

# Copy the visualization app and requirements
COPY src/logging/viz/app.py .
COPY src/logging/viz/requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "app.py"] 