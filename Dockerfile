FROM python:3.10

WORKDIR /app

COPY . /app

## Installing the dependencies & removing the previous catched memories
RUN pip install --no-cache-dir -e .

EXPOSE 5000

ENV FASTAPI_APP=app.py

CMD ["python", "app.py"]