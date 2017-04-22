FROM python:3.6.1

WORKDIR /app
ADD requirements.txt requirements.txt

RUN ["pip", "install", "-r", "requirements.txt"]

ADD . /app

EXPOSE 8000
CMD gunicorn -b 0.0.0.0:8000 -w 2 --timeout 600 serve:app