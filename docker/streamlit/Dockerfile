FROM python:3.10-buster

RUN python3.10 -m pip install --upgrade pip

WORKDIR /streamlit/
ADD requirements.txt requirements.txt
RUN python3.10 -m pip install -r requirements.txt

RUN python3.10 -m pip install streamlit==1.2.0
RUN python3.10 -m pip install --upgrade numpy

ADD arc arc
ADD data data
ADD streamlit_app.py streamlit_app.py
