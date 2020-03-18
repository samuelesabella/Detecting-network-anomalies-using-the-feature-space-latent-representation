
if [ -z "$2" ]
then
  qlang="influxql"
else
  qlang="flux"
fi
config="   cache-max-memory-size = 0\\\n  cache-snapshot-memory-size = 0\\\n \\\n[http]\\\n  flux-enabled = true \\\n"
config="$config \\\n[coordinator]\\\n query-timeout = \\\"0\\\" \\\n log-queries-after = \\\"0\\\" \\\n max-select-point = 0\\\n max-select-series = 0\\\n max-select-buckets = 0 \\\n"

docker run --rm -it -p 8086:8086 --entrypoint /bin/bash \
  -v $1:/backups influxdb:latest \
  -c "(echo -e $config >> /etc/influxdb/influxdb.conf && cat /etc/influxdb/influxdb.conf && influxd) & 
      (sleep 2 && influxd restore -portable /backups && influx -type=$qlang)"


