FROM python:3.8-slim

COPY src/requirements.txt /root/FlaskServer/src/requirements.txt

RUN chown -R root:root /root/FlaskServer

WORKDIR /root/FlaskServer/src
RUN pip3 install -r requirements.txt

COPY FlaskServer/src/ ./
RUN chown -R root:root ./

ENV SECRET_KEY mmp
ENV FLASK_APP run.py

RUN chmod +x run.py
CMD ["python3", "run.py"]