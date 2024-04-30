FROM python:3.10.12

COPY src/ /app
COPY datasets/ /datasets
COPY infrastructure/ /infrastructure

WORKDIR /app

COPY requirements.txt /app

RUN apt-get update && apt-get install -y openjdk-17-jdk

RUN pip install -U pip

RUN pip install -U setuptools wheel

RUN pip install -r requirements.txt

CMD ["python", "entrypoint.py", "--dedicated", "--saved-config=global-performance"]
