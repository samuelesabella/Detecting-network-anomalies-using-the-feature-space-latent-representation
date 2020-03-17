import csv
import pprint
import datetime
import requests
import pandas as pd


# ----- ----- RESULT ----- ----- #
# ----- ----- ------ ----- ----- #
FLUX_TYPE_MAP = {
    'string': str,
    'double': float,
    'long': int,
    'bool': bool,
    'uint': int,
    'int': int,
}


class FluxResponse():
    def __init__(self, res, query, groupby=True):
        self.query = query
        csv_data = self.parse_csv(res)
        # TODO: handle empty queries and errors!
        keys = csv_data[3][3:]
        dtypes = csv_data[0][3:]
        # Parsing results
        table = []
        for x in csv_data[4:-1]:
            if x==[]:
                break
            table.append(x[3:])
        dframe = pd.DataFrame(table, columns=keys)
        # Casting types ..... #
        for (k, dtype) in zip(keys, dtypes):
            dframe[k] = self.castFluxSeries(dframe[k], dtype)
        # Grouping ..... #
        if groupby:
            groups = csv_data[1][3:]
            group_keys = ([k for (i, k) in enumerate(keys) if groups[i] == 'true'])
            if len(group_keys) > 0:
                dframe = dframe.groupby(group_keys)

        self.dframe = dframe

    @staticmethod
    def rfc33392datatime(s):
        if len(s.split('.')) > 1:
            return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%S.%fZ')
        return pd.to_datetime(s, format='%Y-%m-%dT%H:%M:%SZ')

    @staticmethod
    def castFluxSeries(pd_series, str_fluxtype):
        if str_fluxtype in FLUX_TYPE_MAP:
            pd_series.loc[pd_series == ''] = None
            ptype = FLUX_TYPE_MAP[str_fluxtype]
            return pd_series.astype(ptype)
        if str_fluxtype == 'dateTime:RFC3339':
            return pd_series.apply(FluxResponse.rfc33392datatime)
        return pd_series

    @staticmethod
    def parse_csv(res):
        restrs = res.content.decode('utf-8')
        csv_data = csv.reader(restrs.splitlines())
        csv_data = list(csv_data)
        return csv_data

    def __str__(self):
        return pprint.pformat([self.keys, self.table])


# ----- ----- QUERY ----- ----- #
# ----- ----- ----- ----- ----- #
class FluxQueryFrom():
    def __init__(self, bucket):
        self.query = [f'from(bucket:"{bucket}")']

    def range(self, start, stop=None):
        if not stop:
            stop = 'now()'

        if isinstance(start, pd.Timestamp):
            start = start.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        if isinstance(stop, pd.Timestamp):
            stop = stop.strftime('%Y-%m-%dT%H:%M:%S.%fZ')

        self.query.append(f'range(start:{start}, stop:{stop})')
        return self

    def filter(self, fn):
        self.query.append(f'filter(fn:{fn})')
        return self

    def group(self, columns, mode='by'):
        cstr = str(columns).replace("\'", "\"")
        self.query.append(f'group(columns:{cstr}, mode:"{mode}")')
        return self

    def drop(self, columns=None, fn=None):
        if not (columns or fn):
            raise ValueError("No dropping method specified")

        if columns:
            self.query.append(f'drop(columns:{columns})')
        else:
            self.query.append(f'drop(fn:{fn})')
        return self

    def keys(self):
        self.query.append('keys()')
        return self

    def window(self, every):
        self.query.append(f'window(every:{every})')
        return self

    def aggregateWindow(self, every, fn):
        self.query.append(f'aggregateWindow(every:{every}, fn:{fn})')
        return self

    def mean(self):
        self.query.append('mean()')
        return self

    def distinct(self, column):
        self.query.append(f'distinct(column:"{column}")')
        return self

    def keep(self, columns):
        self.query.append(f'keep(columns:{columns})')
        return self
    
    def sort(self, columns, desc=False):
        self.query.append(f'sort(columns: {columns}, desc: {str(desc).lower()})')
        return self

    def last(self):
        self.query.append('last()')
        return self

    def first(self):
        self.query.append('first()')
        return self

    def limit(self, n=10, offset=0):
        self.query.append(f'limit(n: {n}, offset: {offset})')
        return self

    def __str__(self):
        s = " |> ".join(self.query)
        return s.replace("\'", "\"")


# ----- ----- CLIENT ----- ----- #
# ----- ----- ------ ----- ----- #
class Flux():
    def __init__(self, host='localhost', port=8086):
        self.session = requests.Session()
        url = f'http://{host}:{port}/api/v2/query'
        head = {
            'accept': 'application/csv',
            'content-type': 'application/vnd.flux'
        }
        self.preq = requests.Request('POST', url, headers=head)

    def __call__(self, q, grouby=True):
        self.preq.data = str(q)
        res = self.session.send(self.preq.prepare())

        return FluxResponse(res, q, grouby)

    def show_tag_keys(self, bucket, from_measurement, trange='-48h'):
        q = FluxQueryFrom(bucket).range(trange)
        q.filter(f'(r) => r._measurement == "{from_measurement}"')
        q.keep(["_field"]).keys().keep(["_field"])
        return self(q)

    def show_tag_values(self, bucket, from_measurement, with_key, trange='-48h'):
        q = FluxQueryFrom(bucket).range(trange)
        q.filter(f'(r) => r._measurement == "{from_measurement}"')
        q.group([with_key]).distinct(with_key)
        return self(q)

    def bucket_timerange(self, bucket):
        # Returns the oldest and the newest value in the bucket
        qfirst = FluxQueryFrom(bucket).range('1970-01-01T00:00:00Z').first() 
        qlast = FluxQueryFrom(bucket).range('1970-01-01T00:00:00Z').last()
        fs = self(qfirst, False).dframe.iloc[0]["_time"]
        ls = self(qlast, False).dframe.iloc[0]["_time"]
        return (fs, ls)

    def show_measurements(self, bucket, trange='-48h'):
        q = FluxQueryFrom(bucket)
        q.range(trange).group(["_measurement"])
        q.distinct("_measurement").keep(["_value"])
        return self(q)
