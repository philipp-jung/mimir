FROM continuumio/miniconda3:24.9.2-0

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-17-jdk

COPY src/ /app

WORKDIR /app

COPY environment.yml /app

RUN conda env create -n mimir -f environment.yml
RUN conda clean --all --yes

ENTRYPOINT ["conda", "run", "-n", "mimir", "python", "entrypoint.py"]
