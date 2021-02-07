FROM ubuntu:bionic
EXPOSE 3000

RUN apt-get update

# Ntop 3.4 ..... #
RUN apt-get -y install build-essential git bison flex libxml2-dev libpcap-dev libtool \
                       libtool-bin rrdtool librrd-dev autoconf pkg-config automake autogen \
                       redis-server wget libsqlite3-dev libhiredis-dev libmaxminddb-dev \
                       libcurl4-openssl-dev libpango1.0-dev libcairo2-dev libnetfilter-queue-dev \ 
                       zlib1g-dev libssl-dev libcap-dev libnetfilter-conntrack-dev libreadline-dev \
                       libjson-c-dev libldap2-dev rename libsnmp-dev debhelper dpkg-sig dkms unzip \
                       libmysqlclient-dev libzmq3-dev iproute2 net-tools curl
RUN git clone https://github.com/ntop/nDPI.git && cd nDPI && git checkout 49857135 && cd ..
RUN cd nDPI && ./autogen.sh && make && cd ..
RUN git clone https://github.com/ntop/ntopng && cd ntopng && git checkout c5f2303 && cd ..
RUN mkdir /var/lib/ntopng && chmod -R 777 /var/lib/ntopng
RUN cd ntopng && ./autogen.sh && ./configure && make install && cd ..
RUN apt-get install -y redis-server 

RUN wget https://dl.influxdata.com/influxdb/releases/influxdb_1.8.3_amd64.deb
RUN dpkg -i influxdb_1.8.3_amd64.deb 

RUN apt-get install -y tcpreplay iproute2

# Cloning private repo ..... #
RUN apt-get install -y git
RUN git clone https://github.com/samuelesabella/Detecting-network-anomalies-using-the-feature-space-latent-representation/ /app/sabella
RUN chmod +rwx /app/sabella/src/data_generator.py

# Installing python3.7 ..... #
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install -y python3.7 python3-pip
RUN python3.7 -m pip install --upgrade pip
RUN pip3 install -r /app/sabella/lab.requirements.txt 
RUN pip3 install pyts 

COPY ./pcaps /app/pcaps/
COPY ./container_init.sh /app/container_init.sh

ENTRYPOINT ["/bin/bash", "/app/container_init.sh"]
