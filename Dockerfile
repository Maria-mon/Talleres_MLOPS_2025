# Usar una imagen base de Python
FROM python:3.9-slim

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos necesarios
COPY main.py .
COPY logistic_regression_model.pkl .
COPY requirements.txt .

# Instalar las dependencias
RUN pip install -r requirements.txt

# Exponer el puerto 8989
EXPOSE 8989

# Comando para ejecutar la aplicación
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989"]
