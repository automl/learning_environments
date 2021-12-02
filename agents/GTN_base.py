import glob
import os

import torch.nn as nn

import logging

logger = logging.getLogger(__name__)

class GTN_Base(nn.Module):
    def __init__(self, bohb_id):
        super().__init__()

        self.bohb_id = bohb_id

        sync_dir_base = os.getcwd()
        self.sync_dir = str(os.path.join(sync_dir_base, 'results/GTN_sync'))
        logger.info('SYNC DIR: ' + str(self.sync_dir))

        os.makedirs(self.sync_dir, exist_ok=True)

    def get_input_file_name(self, id):
        return os.path.join(self.sync_dir, str(self.bohb_id) + '_' + str(id) + '_input.pt')

    def get_input_check_file_name(self, id):
        return os.path.join(self.sync_dir, str(self.bohb_id) + '_' + str(id) + '_input_check.pt')

    def get_result_file_name(self, id):
        return os.path.join(self.sync_dir, str(self.bohb_id) + '_' + str(id) + '_result.pt')

    def get_result_check_file_name(self, id):
        return os.path.join(self.sync_dir, str(self.bohb_id) + '_' + str(id) + '_result_check.pt')

    def get_quit_file_name(self):
        return os.path.join(self.sync_dir, 'quit.flag')

    def clean_working_dir(self):
        files = glob.glob(os.path.join(self.sync_dir, '*'))
        for file in files:
            os.remove(file)
