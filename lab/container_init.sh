#!/bin/bash
TIMESHIFT_FILE="/app/timeshift.txt"
LOCALNET="192.168.10.0/24,205.174.165.0/24"

redis-server 1>/dev/null &
service influxdb start

# Starting dummy interface ..... #
ip link add fake_nic type dummy && \
  ifconfig fake_nic up && \
  ifconfig fake_nic mtu 1400

# Starging ntopng ..... #
ntopng --disable-login=1 --interface=fake_nic --https-port=0.0.0.0:3443 -m $LOCALNET 1>/dev/null &
sleep 10
curl -sX POST -d "timeseries_driver=influxdb&ts_post_data_url=http%3A%2F%2Flocalhost%3A8086&influx_dbname=ntopng&toggle_influx_auth=0&influx_username=&influx_password=&ts_high_resolution=10&influx_query_timeout=10&toggle_interface_traffic_rrd_creation=1&interfaces_ndpi_timeseries_creation=both&hosts_ts_creation=full&hosts_ndpi_timeseries_creation=both&toggle_l2_devices_traffic_rrd_creation=1&l2_devices_ndpi_timeseries_creation=per_category&toggle_system_probes_timeseries=1&toggle_flow_rrds=1&toggle_pools_rrds=1&toggle_vlan_rrds=1&toggle_asn_rrds=1&toggle_country_rrds=1&toggle_ndpi_flows_rrds=1&toggle_internals_rrds=0" http://localhost:3000/lua/admin/prefs.lua > /dev/null


echo "> Environment ready"
# Formatting timeshift file ..... #
echo "TCPREPLAY_START = {" > $TIMESHIFT_FILE
# 
# # Starting tcpreplay
# for absfname in /app/pcaps/*.pcap; do
#     [ -f "$fname" ] || break
# 
#     fname=`basename absfname .pcap`
#     start=`date --utc +"%Y-%m-%dT%H:%M:%S.%N"`
#     tcpreplay -i fake_nic $absfname
#     echo " \"$fname\": \"$start\", " >> $TIMESHIFT_FILE
# 
#     # Backing up and cleaning influxdb
#     influxd backup -portable -database ntopng
# 
# done
# 
# 
# echo "}" >> $TIMESHIFT_FILE