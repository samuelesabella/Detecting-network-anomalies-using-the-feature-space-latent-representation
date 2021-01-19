#!/bin/bash
HELLO="
----- ----- PCAP2FLOW LAB ----- -----
| environment ready, now configuring |
| then replaying packets...          |
----- ----- ------------- ----- -----"
POLLEVERY=2

echo "$HELLO"

TIMESHIFT_FILE="/app/ext/timeshift.txt"
LOCALNET="192.168.1.0/24"
# "192.168.10.0/24,205.174.165.0/24"

eval $(ssh-agent) && \
ssh-add /app/keys/github_key && \
ssh-keyscan -H github.com >> /etc/ssh/ssh_known_hosts && \
cd /app/tesi_sabella && git checkout . && git pull && cd -

redis-server 1>/dev/null &
openssl req -new -newkey rsa:4096 -nodes -keyout /app/keys/influxdb.key -out /app/keys/influxdb.csr \
  -subj "/C=US/ST=Denial/L=Springfield/O=Dis/CN=www.example.com"
openssl x509 -req -sha256 -days 365 -in /app/keys/influxdb.csr -signkey /app/keys/influxdb.key -out /app/keys/influxdb.pem
influxd -config /app/tesi_sabella/lab/influxd_config 1>/dev/null 2>/dev/null &


# Starting dummy interface ..... #
ip link add fake_nic type dummy && \
  ifconfig fake_nic up && \
  ifconfig fake_nic mtu 20000

# Starting ntopng ..... #
ntopng --disable-login=1 --interface=fake_nic --http-port=0.0.0.0:3900 -m $LOCALNET 1>/dev/null &
 
sleep 15
# This line extract cross-site-request-forgery token and use it to POST the configuration
csrf=$(curl -s http://127.0.0.1:3900/lua/admin/prefs.lua?tab=on_disk_ts | grep "name=\"csrf\"" | grep -oh "value=\"[a-zA-Z0-9]*\"" | tail -1 | sed -e "s/value=//g" -e "s/\"//g") && curl -s -X POST -d "timeseries_driver=influxdb&ts_post_data_url=http%3A%2F%2Flocalhost%3A8086&influx_dbname=ntopng&toggle_influx_auth=0&influx_username=&influx_password=&ts_high_resolution=10&influx_query_timeout=10&toggle_interface_traffic_rrd_creation=1&interfaces_ndpi_timeseries_creation=both&hosts_ts_creation=full&hosts_ndpi_timeseries_creation=both&toggle_l2_devices_traffic_rrd_creation=1&l2_devices_ndpi_timeseries_creation=per_category&toggle_system_probes_timeseries=1&toggle_vlan_rrds=1&toggle_asn_rrds=1&toggle_country_rrds=1&toggle_internals_rrds=1&&csrf=${csrf}" http://127.0.0.1:3900/lua/admin/prefs.lua

cat 

# echo "> Environment ready"
# # Formatting timeshift file ..... #
# echo "TCPREPLAY_START = {" > $TIMESHIFT_FILE
# tcpreplay -i fake_nic "$absfname" &
# 
# cat
# 
# # Starting tcpreplay
# for absfname in /app/pcaps/*.pcap; do
#     [ -f "$absfname" ] || break
#     fname=`basename "$absfname" .pcap`
#     echo "> Replaying \"$fname\""
# 
#     python3.7 /app/tesi_sabella/src/data_generator.py -b ntopng --credentials=admin:admin -o /app/ext/$fname -e $POLLEVERY &
#     data_gen_pid=$!
#     echo "Poller: $data_gen_pid"
# 
#     start=`date --utc +"%Y-%m-%dT%H:%M:%S.%N"`
#     echo "tcpreplay -i fake_nic \"$absfname\""
#     tcpreplay -i fake_nic "$absfname"
#     echo " \"$fname\": \"$start\", " >> $TIMESHIFT_FILE
# 
#     kill -INT $data_gen_pid
#     sleep $(( ($POLLEVERY * 60) + 1 ))
#     influxd backup -portable -database ntopng "/app/ext/${fname}.backup"
# done
# 
# echo "}" >> $TIMESHIFT_FILE
