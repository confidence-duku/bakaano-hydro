FROM python:3.10.4-slim
RUN apt-get update && \
    apt-get install -y g++
SHELL ["/bin/bash", "--login", "-c"]
COPY . /app_dir
WORKDIR /app_dir/
RUN pip install -r requirements.txt
EXPOSE 8000
ENTRYPOINT python ecoeffect.py
