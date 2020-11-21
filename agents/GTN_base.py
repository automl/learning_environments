import torch.nn as nn
import os
import glob
from envs.env_factory import EnvFactory



class GTN_Base(nn.Module):
    def __init__(self, bohb_id):
        super().__init__()

        self.bohb_id = bohb_id

        self.working_dir = str(os.path.join(os.getcwd(), 'results/GTN_sync2'))

        os.makedirs(self.working_dir, exist_ok=True)

    def get_input_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_input.pt')

    def get_input_check_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_input_check.pt')

    def get_result_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_result.pt')

    def get_result_check_file_name(self, id):
        return os.path.join(self.working_dir, str(self.bohb_id) + '_' + str(id) + '_result_check.pt')

    def get_quit_file_name(self):
        return os.path.join(self.working_dir, 'quit.flag')

    def clean_working_dir(self):
        files = glob.glob(os.path.join(self.working_dir, '*'))
        for file in files:
            os.remove(file)

