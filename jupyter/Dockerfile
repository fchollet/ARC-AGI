FROM python:3.10-buster

RUN python3.10 -m pip install --upgrade pip
RUN python3.10 -m pip install jupyterlab

WORKDIR /jupyter/
ADD requirements.txt requirements.txt
RUN python3.10 -m pip install -r requirements.txt

# Jupyterlab wants to run with user-level permissions by default
# RUN useradd -ms /bin/bash jovyan
# USER jovyan
# WORKDIR /home/jovyan/
