import pandas as pd
import os
import data_generator as generator


KNOWN_IP = { "192.168.1.158": "Soomfy Doorlock - IoT23",
             "192.168.1.132": "Phillips HUE - IoT23",
             "192.168.2.3": "Amazon Echo - IoT23", 
             "192.168.1.240": "Amazon Echo - IoT analytics",
             "192.168.1.197": "RaspberryPI", # "Trojan malware",
             "192.168.1.199": "RaspberryPI" } #"Mirai botnet"}

KNWON_MALICIUS = { "192.168.1.197": "Trojan",
                    "192.168.1.199": "Mirai botnet" }


class SimplePreprocessor(generator.Preprocessor):
    @staticmethod
    def label_host(df, host, anomaly):
        if (df.index.get_level_values("_host") == host).any():
            df.loc[(slice(None), host, slice(None)), "_isanomaly"] = anomaly
        return df
    
    def load_path(self, path_to_pkl):
        scenarios = [f for f in os.listdir(path_to_pkl) if f.endswith("pkl")]
        scenarios = [path_to_pkl / f for f in scenarios]

        df = pd.DataFrame()
        for sc in scenarios:
            iot_s = pd.read_pickle(sc)
            print(sc)
            scenario_df = self.preprocessing(iot_s)
            df = pd.concat([df, scenario_df])

        for host, label in KNWON_MALICIUS.items(): 
            df = self.label_host(df, host, label)

        return df

    def preprocessing(self, df):
        # Few changes to indexes ----- #
        df = df.reset_index()
        df["_time"] = df["_time"].dt.tz_localize(None)
        
        for ip, dev_cat in KNOWN_IP.items():
            df.loc[df["host"] == ip, "device_category"] = dev_cat
        df = df[df["device_category"] != "unknown"]

        df = df.set_index(["device_category", "host", "_time"])

        # Filtering hosts ..... #
        df = df.drop(["host_unreachable_flows:flows_as_client",
                      "dns_qry_sent_rsp_rcvd:replies_error_packets",
                      "dns_qry_rcvd_rsp_sent:replies_error_packets"], axis=1) # All zeros in training the dataset

        # Removing initial non zero traffic ..... #
        preproc_df = super().preprocessing(df)
        preproc_df.index.rename(["_device_category", "_host", "_time"], inplace=True)

        preproc_df["_isanomaly"] = "none"

        # Fix, in training we saw only morning and afternoon
        preproc_df[["time:evening", "time:night", "time:morning"]] = 0
        preproc_df["time:afternoon"] = 1

        return preproc_df
