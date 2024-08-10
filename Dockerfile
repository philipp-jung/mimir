FROM continuumio/miniconda3

COPY src/ /app
COPY datasets/ /datasets
COPY infrastructure/ /infrastructure

WORKDIR /app

COPY environment.yml /app

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jdk

RUN conda env create -n mimir -f environment.yml

ENTRYPOINT ["conda", "run", "-n", "mimir", "python", "entrypoint.py"]
