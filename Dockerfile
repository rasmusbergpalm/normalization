FROM python:3.6.1

WORKDIR /app
ADD . /app

RUN ["pip", "install", "-r", "requirements.txt"]

EXPOSE 5000
ENV FLASK_APP serve.py
CMD flask run --host=0.0.0.0