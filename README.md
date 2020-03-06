# influxdb population with ntopng
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
