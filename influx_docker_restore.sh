
if [ -z "$2" ]
then
  qlang="influxql"
else
  qlang="flux"
fi

config=$(<influxd_config)

docker run --rm -it -p 8086:8086 --entrypoint /bin/bash \
  --name tesi_sabella_influx \
  -v $1:/backups influxdb:latest \
  -c "(echo '$config' > /etc/influxdb/influxdb.conf && cat /etc/influxdb/influxdb.conf && influxd) & 
      (sleep 2 && influxd restore -portable /backups && influx -type=$qlang)"


