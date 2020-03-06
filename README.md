# influxdb population
Create a dummy interface named _fake_nic_
```
$ sudo ip link add fake_nic type dummy
$ sudo ifconfig fake_nic up
$ sudo ifconfig fake_nic mtu 1400
```
Start **ntopng** and listen on the dummy interface
```
$ sudo ntopng -i fake_nic
```
Replay packets at native speed using **tcp_replay**
```
$ sudo tcpreplay -i fake_nic capture.pca
```

# Restore influx database
```
docker run --rm -it -p 8086:8086 --entrypoint /bin/bash -v /absolute/path/to/backup/:/backups influxdb -c "(influxd & influxd restore -portable /backups) && influx"
```

# Dataset
* Very small subsample, used for testing purpose: [CICIDS2017 Monday traffic from 15-16](http://bit.ly/CICIDS2017_Monday_from15to16_influx). The subsample have been extracted using **editcap**
```
$ editcap -A "2017-07-03 15:00:00" -B "2017-07-03 16:00:00" Monday-WorkingHours.pcap Monday_14_15.pcap
```
