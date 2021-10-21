FROM python:3.8-slim

COPY ./app ./app
WORKDIR ./app

RUN pip install --upgrade pip 
RUN pip install flask
RUN pip install numpy
RUN pip install transformers==2.8.0
RUN pip3 install torch==1.9.0 
RUN pip install pytorch-pretrained-bert==0.6.2

CMD ["python",  "app.py"] 

EXPOSE 5000