FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git && apt-get clean

# Copy file cần thiết vào container
COPY requirements.txt /app/
COPY train.py /app/

# Cài đặt thư viện cần thiết
RUN pip install -r requirements.txt

# Đảm bảo rằng script sử dụng volume chia sẻ
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

EXPOSE 5000

# Chạy script train
CMD ["python", "train.py"]
