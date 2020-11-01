Laboratory
==========
This directory contains almost all what is needed to turn pcaps into datasets. 

To create influxd backups files store the pcaps of interest into the */pcaps/* directory, then build and run the environment:
```
$ docker build -t pcap_extractor_lab . && docker run -it -p=8080:3443 --cap-add=NET_ADMIN --rm --name=tcplab pcap_extractor_lab
```
***Note***: ntopng stores only the local network data. Please modify the file *container_init.sh* to include all the local network in interest

The container will store influxd backups into the directory */influx_backups*. Then InfluxDB needs to be restarted one backup at a time, with the command:
```
$ sh influx_docker_restore.sh /path/to/backup
```
After having InfluxDB running, follow the instructions into the Notebook *offline_generator.ipynb*

