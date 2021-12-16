FROM ubuntu:18.04

RUN apt update && apt-get update

WORKDIR /app

# setup python environment
RUN apt install -y python3.7 python3.7-dev curl python3-distutils
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
# for opencv
RUN apt install -y libglib2.0-0 libsm6 libfontconfig1 libxrender1 libxext6 libgl1-mesa-dev
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN pip install pipenv
RUN pipenv --python 3.7

COPY ./ /app

# install python packages listed in Pipfile.
RUN pipenv lock && pipenv install

#install supervisor nginx
RUN apt install -y supervisor rsyslog

ADD ./docker/api/supervisor/supervisord.conf /etc/supervisord.conf
ADD ./docker/api/supervisor/supervisor.d/ /etc/supervisor.d/

# RUN ln -sf /dev/stdout /var/log/supervisor/daemon.log
# RUN ln -sf /dev/stdout /var/log/messages

CMD ["/usr/bin/supervisord"]

EXPOSE 8080