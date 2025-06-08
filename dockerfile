
FROM python:3.9-slim
#
WORKDIR /app
# 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
#
COPY . .
#
# Train the model
RUN python model_trainer.py
# 
EXPOSE 10000
# 
CMD ["python", "app.py"]