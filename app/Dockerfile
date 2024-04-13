# Use an official Python runtime as a parent image
FROM tiangolo/uvicorn-gunicorn:python3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY main.py /app/main.py
COPY model.py /app/model.py
COPY training_utils.py /app/training_utils.py
COPY data_processing.py /app/data_processing.py
COPY model_config.json /app/model_config.json
COPY L_size_model_weights.h5 /app/L_size_model_weights.h5
COPY tokenizer.keras /app/tokenizer.keras


RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8012

ENV MODULE_NAME="main"
ENV VARIABLE_NAME="app"
ENV PORT="8012"

 CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8012"]
