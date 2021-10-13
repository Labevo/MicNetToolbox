# IA Team Ixulabs

FROM continuumio/miniconda3:latest

WORKDIR /app
##Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN export DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
  build-essential \
  byobu \
  curl \
  git-core \
  htop \
  unzip \
  && \
  apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Conda 
RUN conda update -n base -c defaults conda
RUN conda clean --all

COPY environment.yml /app/environment.yml
RUN conda env create -f /app/environment.yml

RUN echo "source activate MicNet-env" > ~/.bashrc
ENV PATH /opt/conda/envs/MicNet-env/bin:$PATH

#Folders
COPY . /app

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]