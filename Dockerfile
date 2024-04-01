FROM python:3.10

RUN git clone https://github.com/lm-sys/FastChat.git

WORKDIR app/FastChat

RUN pip3 install "fschat[model_worker,webui]"

COPY . .
