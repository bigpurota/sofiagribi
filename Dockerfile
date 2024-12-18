# Используем официальный образ Python в качестве базового
FROM python:3.10-slim

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Копируем файл с зависимостями (например, requirements.txt)
COPY requirements.txt .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Открываем порт 8501 для Streamlit
EXPOSE 8501

# Запускаем Streamlit-приложение с параметрами
CMD ["streamlit", "run", "app/main.py", "--server.address", "0.0.0.0", "--server.port", "8501"]