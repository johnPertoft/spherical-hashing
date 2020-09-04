FROM python:3.7.9

RUN pip install \
    pytest==6.0.1 \
    tensorflow==2.3.0 \
    twine==3.2.0

WORKDIR /workspace
