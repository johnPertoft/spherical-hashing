FROM python:3.7.9

RUN pip install \
    tensorflow==2.3.0 \
    pytest==6.0.1

WORKDIR /workspace
