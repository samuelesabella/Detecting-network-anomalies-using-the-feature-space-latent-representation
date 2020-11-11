HELLO="
----- ----- INFLUX RESTORED ----- -----
| The backup has been restored,       |
| database ready.                     |
----- ----- -------------- ----- ------
"
config=$(<influxd_config)

docker run --rm -it -p 8086:8086 --entrypoint /bin/bash \
  --name tesi_sabella_influx \
  -v $(pwd)/$1:/backups influxdb:latest \
  -c "(echo '$config' > /etc/influxdb/influxdb.conf && cat /etc/influxdb/influxdb.conf && influxd) & 
      (sleep 30 && influxd restore -portable -db \"ntopng\" -newdb \"ntopng\" /backups && echo \"$HELLO\" && tail -f /dev/null)"


