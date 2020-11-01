FROM ubuntu:bionic
EXPOSE 3000

RUN apt-get update

RUN apt-get -y install software-properties-common wget curl
RUN add-apt-repository universe
RUN wget https://packages.ntop.org/apt-stable/18.04/all/apt-ntop-stable.deb
RUN apt install ./apt-ntop-stable.deb
# RUN wget https://packages.ntop.org/apt/18.04/all/apt-ntop.deb
# RUN apt install ./apt-ntop.deb
RUN apt-get clean all && apt-get update && apt-get install -y ntopng

RUN apt-get install -y redis-server 

RUN wget https://dl.influxdata.com/influxdb/releases/influxdb_1.5.5_amd64.deb
RUN dpkg -i influxdb_1.5.5_amd64.deb 

RUN apt-get install -y tcpreplay iproute2

# COPY ./scripts_configuration.json /app/scripts_configuration.json
# COPY ./runtimeprefs.json /var/lib/ntopng/runtimeprefs.json
COPY ./pcaps /app/pcaps/
COPY ./pcap_extractor.sh /app/pcap_extractor.sh

ENTRYPOINT ["/bin/bash", "/app/pcap_extractor.sh"]
