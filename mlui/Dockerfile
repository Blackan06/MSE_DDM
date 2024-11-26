FROM bitnami/mlflow:latest

WORKDIR /app
COPY . /app

# Tạo thư mục mlruns mà không thay đổi quyền sở hữu
RUN mkdir -p /app/mlruns

EXPOSE 5000

ENTRYPOINT ["mlflow"]
CMD ["ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/app/mlruns"]
