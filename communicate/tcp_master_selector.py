import selectors
import socket
import threading
import gzip
import os
import pickle

import logging

logger = logging.getLogger(__name__)

my_selector = selectors.DefaultSelector()
keep_running = True

connections_for_later = {}
lost_connections = {}
connections_response_time = {}
finished_for_iteration = []


def work_with_sent_data(data, ip, port):
    key_for_connection = f"{ip}_{port}"

    if data == "finished_iteration":
        key_for_connection = f"{ip}_{port}"
        finished_for_iteration.append(connections_for_later[key_for_connection]["id"])
    else:
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
        # logger.info('read({})  sent: {!r}'.format(client_address, data_loaded))
    else:
        # Interpret empty result as closed connection
        logger.info('  closing for {}'.format(client_address))
        my_selector.unregister(connection)
        connection.close()

        key_lost_connection = f"{client_address[0]}_{client_address[1]}"
        lost_con = connections_for_later.pop(key_lost_connection)
        lost_connections[key_lost_connection] = lost_con
        # logger.info("lost_connections : ", lost_connections)
        # logger.info("connections_for_later : ", connections_for_later)


def accept(sock, mask):
    """ Callback for new connections: -> selector on: events=selectors.EVENT_READ
        Accepts a new connection and registers the read function above for this new connection in the event we receive data
    """

    new_connection, addr = sock.accept()  # addr := tuple of (IP , PORT)
    logger.info('accept({})'.format(addr))

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
    logger.info('starting up on {} port {}'.format(*server_address))
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

    logger.info('shutting down')
    logger.info("CONNECTION-LIST (connections_for_later): ", connections_for_later)
    my_selector.close()


def start_communication_thread(args):
    number_of_workers = args.number_workers
    ip = get_ip_by_socket_module()
    write_server_file(ip=ip, port=args.port)

    number_of_parallel_connections = number_of_workers + 1  # just to be sure plus one
    t_c = threading.Thread(target=main_communcication, args=(number_of_parallel_connections, ip, args.port))
    t_c.start()
