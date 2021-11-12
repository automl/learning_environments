import os
import time
import argparse
import selectors
import socket
import threading
import gzip
import os
import pickle
from datetime import datetime

mysel = selectors.DefaultSelector()
keep_running = True

connections_for_later = []
communication_list = []


def get_server_file():
    datasets_dir = "./"
    data_file = "server.pkl.gzip"
    data_path_file = os.path.join(datasets_dir, data_file)

    while not os.path.isfile(data_path_file):
        time.sleep(3)

    f = gzip.open(data_path_file, 'rb')
    data = pickle.load(f)
    return data


def my_worker_thread(server_data, worker_data):
    server_address = (server_data["ip"], server_data["port"])
    print('connecting to {} port {}'.format(*server_address))
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    sock.connect(server_address)
    sock.setblocking(False)  # Connecting is a blocking operation, so call setblocking() after it returns.

    # Set up the selector to watch for when the socket is ready
    # to send data as well as when there is data to read.
    mysel.register(fileobj=sock,  # listen on
                   events=selectors.EVENT_READ | selectors.EVENT_WRITE, )  # events: data receiving or data sending

    while keep_running:
        for key, mask in mysel.select(timeout=1):
            connection = key.fileobj
            client_address = connection.getpeername()
            # print('client({})'.format(client_address))

            if mask & selectors.EVENT_READ:
                pass

            if mask & selectors.EVENT_WRITE:
                # print('  ready to write')
                # Send the next message.
                if communication_list:  # sending finished iteration message
                    next_msg = pickle.dumps(communication_list.pop(), -1)
                    sock.sendall(next_msg)
                else:
                    next_msg = pickle.dumps(worker_data, -1)
                    # print('  sending {!r}'.format(next_msg))
                    sock.sendall(next_msg)
        time.sleep(30)  # in seconds -> 5 minutes

    print('shutting down')
    mysel.unregister(connection)
    connection.close()
    mysel.close()


def start_communication_thread(args):
    # Get IP and PORT OF Server from FILE
    server_data = get_server_file()
    print("server_data : ", server_data)
    worker_data = vars(args)
    print("worker_data: ", worker_data)

    # STRAT Thread for Communication
    t_c = threading.Thread(target=my_worker_thread, args=(server_data, worker_data))
    t_c.start()
