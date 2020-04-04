import numpy as np
import copy
from collections import defaultdict


# ----- ----- L4 ----- ----- #
# ----- ----- -- ----- ----- #
SUPPORTED_L4 = set([
    "ip", "icmp", "igmp", "ggp",
    "ipencap", "st2", "tcp", "cbt",
    "egp", "igp", "bbn-rcc", "nvp",
    "pup", "argus", "emcon", "xnet",
    "chaos", "udp", "mux", "dcn",
    "hmp", "prm", "xns-idp", "trunk-1",
    "trunk-2", "leaf-1", "leaf-2", "rdp",
    "irtp", "iso-tp4", "netblt", "mfe-nsp",
    "merit-inp", "sep", "3pc", "idpr",
    "xtp", "ddp", "idpr-cmtp", "tp++",
    "il", "ipv6", "sdrp", "ipv6-route",
    "ipv6-frag", "idrp", "rsvp", "gre",
    "mhrp", "bna", "esp", "ah",
    "i-nlsp", "swipe", "narp", "mobile",
    "tlsp", "skip", "ipv6-icmp", "ipv6-nonxt",
    "ipv6-opts", "cftp", "sat-expak", "kryptolan",
    "rvd", "ippc", "sat-mon", "visa",
    "ipcv", "cpnx", "cphb", "wsn",
    "pvp", "br-sat-mon", "sun-nd", "wb-mon",
    "wb-expak", "iso-ip", "vmtp", "secure-vmtp",
    "vines", "ttp", "nsfnet-igp", "dgp",
    "tcf", "eigrp", "ospf", "sprite-rpc",
    "larp", "mtp", "ax.25", "ipip",
    "micp", "scc-sp", "etherip", "encap",
    "gmtp", "ifmp", "pnni", "pim",
    "aris", "scps", "qnx", "a/n",
    "ipcomp", "snp", "compaq-peer", "ipx-in-ip",
    "vrrp", "pgm", "l2tp", "ddx",
    "iatp", "st", "srp", "uti",
    "smp", "sm", "ptp", "isis",
    "fire", "crtp", "crdup", "sscopmce",
    "iplt", "sps", "pipe", "sctp",
    "fc", "divert"])
L4_BYTES_RCVD_COMPLETE = set([f"l4protos:bytes_rcvd__{x}" for x in SUPPORTED_L4])
L4_BYTES_SENT_COMPLETE = set([f"l4protos:bytes_sent__{x}" for x in SUPPORTED_L4])


# ----- ----- NDPI ----- ----- #
# ----- ----- ---- ----- ----- #
NDPI_CATEGORIES = { 
    "gaming": [
        "xbox", "battlefield", "quake",
        "steam", "halflife2", "world of warcraft",
        "armagetron", "crossfire", "fiesta",
        "florensia", "guildwars","maplestory", "warcraft3",
        "world of kung fu", "stracraft", "dofus"],
    "sysadmin": [
        "ftp", "ntp", "telnet", 
        "ssh", "rsync", "git", "tftp"],
    "mail_service": ["pop", "smtp", "imap", "gmail"],
    "file_sharing": ["directdownloadlink", "aimini", "applejuice"],
    "p2p_file_sharing": [
        "stealthnet", "filetopia", "kazaa/fasttrack",
        "gnutella", "edonkey", "bittorrent",
        "soulseek", "pando", "kontiki",
        "imesh", "openft", "directconnect",
        "feidian", "fiesta" ],
    "cloud storage": [
        "dropbox", "apple icloud", "microsoft cloud services", 
        "ubuntuone"],
    "search-engine": ["google", "yahoo"],
    "database": ["postgresql", "mysql", "mssql", "redis", "tds"],
    "social": ["facebook", "twitter", "snapchat", "instagram"],
    "update": ["windowsupdate"],
    "routing": [
        "bgp", "dhcp", "vrrp",
        "egp", "ospf", "megaco", "dhcpv6"],
    "video-media": ["youtube", "netflix", "twitch", "1kxun"],
    "chat": [
        "qq", "whatsapp", "viber", 
        "hangout", "kakaotalk voice and chat", "meebo"],
    "tor": ["tor" ],
    "instant-message-protocol": [ "oscar", "irc"],
    "message-broker": ["zeromq", "mqtt"],
    "vpn": ["openvpn", "ciscovpn", "hotspotshield vpn", "pptp"],
    "music-service": ["spotify", "apple itunes", "deezer"],
    "maps": ["google maps"],
    "online_encyclopedia": ["wikipedia"],
    "video-chat": [
        "skype", "citrixonline/gotomeeting", 
        "apple", "webex", "jabber"],
    "ridesharing": ["99taxi"],
    "monitoring": ["netflow_ipfix", "sflow", "collectd", "snmp", "syslog", "icmpv6", "icmp"],
    "audio_file": ["ogg", "mpeg"],
    "video_file": ["avi", "quicktime", "realmedia"],
    "dns": ["dns", "mdns", "llmnr"],
    "printing_scanners": ["ipp", "remotescan"],
    "device discovery": [ "ssdp", "upnp"],
    "remote-access": ["xdmcp", "rdp",  "vnc", "teamviewer", "pcanywhere"],
    "iptv": ["zattoo", "veohtv", "globotv"],
    "p2p-iptv": ["sopcast", "tvants", "tvuplayer"],
    "p2p-streaming": ["ppstream", "pplive", "qqlive"],
    "streaming": ["shoutcast", "icecast", "rtsp"],
    "aaa-protocol": ["radius", "diameter"],
    "l4-encr": ["ssl"],
    "http": ["http"],
    "network_file_system": ["nfs", "smb", "afp", "ldap", "http application activesync"],
    "voip": ["mgcp", "iax", "truphone", "teamspeak", "whatsapp voice"],
    "real-time-media-protocol": [ "h323", "rtcp", "rtp", "sip", ],
    "other-l4": ["quic", "sctp"],
    "rpc": ["rx", "dce rpc"],
    "osi-l5": ["netbios"],
    "ipsec": ["ipsec"],
    "tunneling": ["ip in ip", "gtp", "gre"],
    "other": [
        "teredo", "stun", "vmware",
        "citrix", "coap", "tuenti",
        "usenet", "kerberos", "simet",
        "opensignal", "apache jserv protocol", "i23v5", "socrates",
        "off", "flash", "windowsmedia",
        "thunder/webthunder", "msn", "igmp",
        "http connect (ssl over http)", 
        "http proxy", "lotusnotes",
        "sap" , "noe",
        "ciscoskinny", "oracle", "corba",
        "mms", "move", "cnn", "whois-das"]
}
NDPI_VALUE2CAT = defaultdict(lambda: "unknown", {v: key  for (key, values) in NDPI_CATEGORIES.items() for v in values})
SUPPORTED_NDPI = np.sum(list(NDPI_CATEGORIES.values()), initial=[])
NDPI_FLOWS_COMPLETE = set({f"ndpi_flows:num_flows__{x}" for x in NDPI_CATEGORIES.keys()})
NDPI_BYTES_RCVD_COMPLETE = set({f"ndpi:bytes_rcvd__{x}" for x in NDPI_CATEGORIES.keys()})
NDPI_BYTES_SENT_COMPLETE = set({f"ndpi:bytes_sent__{x}" for x in NDPI_CATEGORIES.keys()})


# ----- ----- FEATURES ----- ----- #
# ----- ----- -------- ----- ----- #
# To prevent new feature coming with new versions of ntopng
BASIC_FEATURES = set([
    "active_flows:flows_as_client",
    "active_flows:flows_as_server",
    "contacts:num_as_client",
    "contacts:num_as_server",
    "dns_qry_rcvd_rsp_sent:queries_packets",
    "dns_qry_rcvd_rsp_sent:replies_error_packets",
    "dns_qry_rcvd_rsp_sent:replies_ok_packets",
    "dns_qry_sent_rsp_rcvd:queries_packets",
    "dns_qry_sent_rsp_rcvd:replies_error_packets",
    "dns_qry_sent_rsp_rcvd:replies_ok_packets",
    "echo_packets:packets_rcvd",
    "echo_packets:packets_sent",
    "echo_reply_packets:packets_rcvd",
    "echo_reply_packets:packets_sent",
    #"engaged_alerts:alerts", (not used)
    "host_unreachable_flows:flows_as_client",
    "host_unreachable_flows:flows_as_server",
    #"l4protos:bytes_rcvd__x", with x an l4 protocol
    "misbehaving_flows:flows_as_client",
    "misbehaving_flows:flows_as_server",
    #"ndpi:bytes_rcvd__x", with x a supported application
    #"ndpi:bytes_sent__x",
    #"ndpi_flows:num_flows__x",
    "tcp_packets:packets_rcvd",
    "tcp_packets:packets_sent",
    "tcp_rx_stats:lost_packets",
    "tcp_rx_stats:out_of_order_packets",
    "tcp_rx_stats:retransmission_packets",
    "tcp_tx_stats:lost_packets",
    "tcp_tx_stats:out_of_order_packets",
    "tcp_tx_stats:retransmission_packets",
    #"total_alerts:alerts", (not used)
    #"total_flow_alerts:alerts", (not used)
    "total_flows:flows_as_client",
    "total_flows:flows_as_server",
    "traffic:bytes_rcvd",
    "traffic:bytes_sent",
    "udp_pkts:packets_rcvd",
    "udp_pkts:packets_sent",
    "udp_sent_unicast:bytes_sent_non_unicast",
    "udp_sent_unicast:bytes_sent_unicast",
    "unreachable_flows:flows_as_client",
    "unreachable_flows:flows_as_server"])
FEATURES_COMPLETE = copy.deepcopy(BASIC_FEATURES)
FEATURES_COMPLETE |= NDPI_FLOWS_COMPLETE | NDPI_BYTES_RCVD_COMPLETE | NDPI_BYTES_SENT_COMPLETE
FEATURES_COMPLETE |= L4_BYTES_RCVD_COMPLETE | L4_BYTES_SENT_COMPLETE
