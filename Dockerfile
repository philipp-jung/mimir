FROM continuumio/miniconda3:24.9.2-0

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jdk

COPY src/ /app
#COPY datasets/ /datasets
#COPY infrastructure/ /infrastructure

WORKDIR /app

COPY environment.yml /app

RUN conda env create -n mimir -f environment.yml

ENTRYPOINT ["conda", "run", "-n", "mimir", "python", "entrypoint.py"]
