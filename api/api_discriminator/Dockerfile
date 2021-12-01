FROM nvcr.io/nvidia/pytorch:21.10-py3

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx
ENV DEBCONF_NOWARNINGS yes
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# setup python environment
RUN apt install -y curl
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# for opencv
RUN apt install -y libglib2.0-0 libsm6 libfontconfig1 libxrender1 libgl1-mesa-dev
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN pip install pipenv
RUN pipenv --python 3

COPY ./ /app/

# install python packages listed in Pipfile.
RUN pipenv install

# install supervisor nginx
RUN apt install -y supervisor rsyslog

ADD ./docker/api/supervisor/supervisord.conf /etc/supervisord.conf
ADD ./docker/api/supervisor/supervisor.d/ /etc/supervisor.d/

RUN ln -sf /dev/stdout /var/log/supervisor/daemon.log
RUN ln -sf /dev/stdout /var/log/messages

CMD ["sh", "/app/launch.sh"]

EXPOSE 8091