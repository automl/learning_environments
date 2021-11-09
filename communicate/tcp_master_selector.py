import os
import time
import argparse
import selectors
import socket
import threading
import gzip
import os
import pickle
from datetime import datetime, time as datetime_time, timedelta
from copy import deepcopy

my_selector = selectors.DefaultSelector()
keep_running = True

connections_for_later = {}
lost_connections = {}
connections_response_time = {}


def time_diff(start, end):
    if isinstance(start, datetime_time):  # convert to datetime
        assert isinstance(end, datetime_time)
        start, end = [datetime.combine(datetime.min, t) for t in [start, end]]
    if start <= end:  # e.g., 10:33:26-11:15:49
        return end - start
    else:  # end < start e.g., 23:55:00-00:25:00
        end += timedelta(1)  # +day
        assert end > start
        return end - start


def work_with_sent_data(data, ip, port):
    key_for_connection = f"{ip}_{port}"
    connections_for_later[key_for_connection]["bohb_id"] = data["bohb_id"]
    connections_for_later[key_for_connection]["id"] = data["id"]
    connections_for_later[key_for_connection]["moab_id"] = data["moab_id"]


def read(connection, mask):
    """ Callback for read events """
    client_address = connection.getpeername()

    # Getting data sent to Master
    data = connection.recv(1024)

    if data:
        # A readable client socket has data
        data_loaded = pickle.loads(data)
        work_with_sent_data(data=data_loaded, ip=client_address[0], port=client_address[1])
        print('read({})  sent: {!r}'.format(client_address, data_loaded))
    else:
        # Interpret empty result as closed connection
        print('  closing for {}'.format(client_address))
        my_selector.unregister(connection)
        connection.close()

        key_lost_connection = f"{client_address[0]}_{client_address[1]}"
        lost_con = connections_for_later.pop(key_lost_connection)
        lost_connections[key_lost_connection] = lost_con
        print("lost_connections : ", lost_connections)
        print("connections_for_later : ", connections_for_later)


def accept(sock, mask):
    """ Callback for new connections: -> selector on: events=selectors.EVENT_READ
        Accepts a new connection and registers the read function above for this new connection in the event we receive data
    """

    new_connection, addr = sock.accept()  # addr := tuple of (IP , PORT)
    print('accept({})'.format(addr))

    key_for_connection = f"{addr[0]}_{addr[1]}"
    connections_for_later[key_for_connection] = {"ip": addr[0], "port": addr[0]}

    new_connection.setblocking(False)

    my_selector.register(fileobj=new_connection,  # register new connection on selector
                         events=selectors.EVENT_READ,  # listen to the event that data is sent
                         data=read)  # data := the callback function (name is misleading)


def get_ip_by_socket_module():
    h_name = socket.gethostname()
    IP_addres = socket.gethostbyname(h_name)
    return IP_addres


def write_server_file(ip, port):
    server_data = {"ip": ip, "port": port}
    datasets_dir = "./"
    data_file = "server.pkl.gzip"
    data_path_file = os.path.join(datasets_dir, data_file)
    f = gzip.open(data_path_file, 'wb')
    pickle.dump(server_data, f)


def get_server_file():
    datasets_dir = "./"
    data_file = "server.pkl.gzip"
    data_path_file = os.path.join(datasets_dir, data_file)
    f = gzip.open(data_path_file, 'wb')
    data = pickle.load(f)
    return data


def main_communcication(num_conn, ip, port):
    server_address = (ip, port)
    print('starting up on {} port {}'.format(*server_address))
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setblocking(False)
    server.bind(server_address)
    server.listen(num_conn)

    my_selector.register(fileobj=server,  # register new connection on selector
                         events=selectors.EVENT_READ,  # listen to the event that data is sent
                         data=accept)  # data := the callback function (name is misleading)

    """ 
    Double usage below, listens to new connections as well as handles sent data:
        - 1st: listens to new connections,
        - 2nd: uses new connection accept, to register a new callback for that connection
        - 3rd: now listens to all connections if they send something and in that case handles the data
    """
    while keep_running:
        for key, mask in my_selector.select(timeout=1):
            callback = key.data
            callback(key.fileobj, mask)

    print('shutting down')
    print("CONNECTION-LIST (connections_for_later): ", connections_for_later)
    my_selector.close()


def start_communication_thread(args):
    number_of_workers = args.number_workers
    ip = get_ip_by_socket_module()
    write_server_file(ip=ip, port=args.port)

    number_of_parallel_connections = number_of_workers + 1  # just to be sure plus one
    t_c = threading.Thread(target=main_communcication, args=(number_of_parallel_connections, ip, args.port))
    t_c.start()

