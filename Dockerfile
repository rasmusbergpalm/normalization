FROM python:3.6.1

WORKDIR /app
ADD requirements.txt requirements.txt

RUN ["pip", "install", "-r", "requirements.txt"]

ADD . /app

EXPOSE 80
CMD gunicorn -b 0.0.0.0:80 -w 2 --timeout 600 serve:app