FROM johntut/gbdx-base-image:latest
MAINTAINER Johnny T. <john@exogenesis.solutions>

ADD task /task
RUN pip install -r /task/requirements.txt

WORKDIR /task/madness

RUN make
WORKDIR /

ENTRYPOINT ["python", "/task/task.py"]
CMD []
