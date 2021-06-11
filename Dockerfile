FROM pytorchlightning/pytorch_lightning:base-cuda-py3.8-torch1.8

COPY . /project
WORKDIR /project

RUN apt-get update && apt-get upgrade -y
RUN pip install -r requirements.txt
