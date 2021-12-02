#!/usr/bin/env python3

import logging
import sys




# http://stackoverflow.com/a/24956305/1076493
# filter messages lower than level (exclusive)
class MaxLevelFilter(logging.Filter):
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno < self.level


def set_logger_up(logger, name="logging"):
    kind_of_logs = {"std_out": False, "two_files": False, "single_file": True}
    # redirect messages to either stdout or stderr based on loglevel
    # stdout < logging.WARNING <= stderr
    formatter = logging.Formatter('%(asctime)s %(levelname)s [%(module)s]: %(message)s')

    if kind_of_logs["std_out"]:
        logging_out = logging.StreamHandler(sys.stdout)
        logging_err = logging.StreamHandler(sys.stderr)
        logging_out.setFormatter(formatter)
        logging_err.setFormatter(formatter)
        logging_out.addFilter(MaxLevelFilter(logging.WARNING))
        logging_out.setLevel(logging.DEBUG)
        logging_err.setLevel(logging.WARNING)

        # root logger, no __name__ as in submodules further down the hierarchy
        logger.addHandler(logging_out)
        logger.addHandler(logging_err)
        logger.setLevel(logging.DEBUG)

    elif kind_of_logs["two_files"]:
        logging_out = logging.FileHandler(f'{name}_out.log')
        logging_err = logging.FileHandler(f'{name}_err.log')

        logging_out.setFormatter(formatter)
        logging_err.setFormatter(formatter)
        logging_out.addFilter(MaxLevelFilter(logging.WARNING))
        logging_out.setLevel(logging.DEBUG)
        logging_err.setLevel(logging.WARNING)

        # root logger, no __name__ as in submodules further down the hierarchy
        logger.addHandler(logging_out)
        logger.addHandler(logging_err)
        logger.setLevel(logging.DEBUG)

    elif kind_of_logs["single_file"]:
        logging_single_file = logging.FileHandler(f'{name}.log')
        logging_single_file.setFormatter(formatter)
        # logging_single_file.addFilter(MaxLevelFilter(logging.WARNING))  # doesn't wite if it is in
        # logging_single_file.setLevel(logging.WARNING) # doesn't wite if it is in

        # root logger, no __name__ as in submodules further down the hierarchy
        logger.addHandler(logging_single_file)
        logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    set_logger_up(name="master_123")

    # WRITING FROM MAIN
    logger.info("An INFO message from " + __name__)
    logger.error("An ERROR message from " + __name__)

    import time

    time.sleep(1)

    sublog_03.log()
