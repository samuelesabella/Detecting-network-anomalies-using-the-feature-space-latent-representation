# Main purpose
Turns pcaps file into a dataset usable to train/test our models

# Before going any further
Ntopng records time series data only for addresses specified via the [local-networks command line option](https://www.ntop.org/guides/ntopng/cli_options.html). Change the `LOCALNET` in `container_init.sh` according to your interests. A fast and simple way of listing the hosts IPs available in a pcap file is to use the following: 
```
$ tshark -Tfields -e eth.src_resolved -e eth.src -e ip.src -r pcaps/test_pcap.pcap | sort | uniq
```
Furthermore, the traffic in the pcap is replayed with tcpreplay in real time due to ntopng constraints. So, to generate a dataset from a 24h pcap you will need 24h.

# Producing datasets with docker
Store the pcaps of interests in the directory `pcaps/`, then run:
```
$ docker build -t pcap_extractor_lab . && docker run -it -p=8080:3900 --cap-add=NET_ADMIN --rm --name=tcplab -v $(pwd)/output:/app/ext pcap_extractor_lab
```
The script will store the exact time each pcap has been reproduced in the file `output/timeshift.txt`, influxDB backups will be stored in the directory `output/pcap_file_name.backup`, datasets will be available at path `output/pcap_file_name.pkl`.

