FROM python:3.8-slim

COPY ./app ./app
WORKDIR ./app

RUN pip install --upgrade pip 
RUN pip install flask
RUN pip install numpy
RUN pip install transformers
RUN pip install pytorch-pretrained-bert
RUN pip3 install torch torchvision torchaudio

CMD ["python",  "app.py"] 

EXPOSE 5000