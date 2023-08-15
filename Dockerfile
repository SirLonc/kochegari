
FROM python:3.8


RUN pip install -r requirements.txt

# Создайте рабочую директорию и скопируйте файлы проекта
WORKDIR /app
COPY . /app

# Запустите Streamlit приложение
CMD ["streamlit", "run", "app_v3.py"]