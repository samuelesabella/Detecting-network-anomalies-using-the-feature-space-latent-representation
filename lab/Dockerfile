FROM ubuntu:bionic
EXPOSE 3000

RUN apt-get update

# Ntop stable ..... #
RUN apt-get -y install software-properties-common wget curl
RUN add-apt-repository universe
RUN wget https://packages.ntop.org/apt-stable/18.04/all/apt-ntop-stable.deb
RUN apt install ./apt-ntop-stable.deb
RUN apt-get clean all && apt-get update && apt-get install -y ntopng

RUN apt-get install -y redis-server 

RUN wget https://dl.influxdata.com/influxdb/releases/influxdb_1.5.5_amd64.deb
RUN dpkg -i influxdb_1.5.5_amd64.deb 

RUN apt-get install -y tcpreplay iproute2

# Cloning private repo ..... #

COPY keys/ /app/keys
RUN apt-get install -y git
RUN eval $(ssh-agent) && \
    ssh-add /app/keys/github_key && \
    ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
    git clone git@github.com:samuelesabella/tesi_sabella /app/tesi_sabella
RUN chmod +rwx /app/tesi_sabella/src/data_generator.py

# Installing python3.7 ..... #
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.7 python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN pip3 install -r /app/tesi_sabella/lab.requirements.txt 

COPY ./pcaps /app/pcaps/
COPY ./container_init.sh /app/container_init.sh

ENTRYPOINT ["/bin/bash", "/app/container_init.sh"]
