FROM python:3.8-slim-buster

COPY requirements.txt ./

RUN pip install -r requirements.txt

ENV BASE_DIR="/"

RUN mkdir ${BASE_DIR}/docker_app_data

COPY app.py ${BASE_DIR}/docker_app_data/
COPY score.py ${BASE_DIR}/docker_app_data/
COPY ./template ${BASE_DIR}/docker_app_data/template
COPY ./Data ${BASE_DIR}/docker_app_data/Data
COPY ./models ${BASE_DIR}/docker_app_data/models

WORKDIR ${BASE_DIR}/docker_app_data

ENTRYPOINT python ${BASE_DIR}/docker_app_data/app.py
# CMD ["/bin/bash"]