
if [ -z "$2" ]
then
  qlang="influxql"
else
  qlang="flux"
fi
config="[http]\\\n  flux-enabled = true"

docker run --rm -it -p 8086:8086 --entrypoint /bin/bash \
  -v $1:/backups influxdb:latest \
  -c "(echo -e $config >> /etc/influxdb/influxdb.conf && influxd) & 
      (sleep 2 && influxd restore -portable /backups && influx -type=$qlang)"


