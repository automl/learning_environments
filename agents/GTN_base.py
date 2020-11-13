import torch.nn as nn
import os
import glob
from envs.env_factory import EnvFactory



class GTN_Base(nn.Module):
    def __init__(self, config, bohb_id):
        super().__init__()

        # for saving/loading
        self.config = config
        self.device = config["device"]

        gtn_config = config["agents"]["gtn"]
        self.max_iterations = gtn_config["max_iterations"]
        self.minimize_score = gtn_config["minimize_score"]
        self.agent_name = gtn_config["agent_name"]

        self.bohb_id = bohb_id

        self.env_factory = EnvFactory(config)
        self.virtual_env_orig = self.env_factory.generate_virtual_env(print_str='GTN_Base: ')

        self.working_dir = str(os.path.join(os.getcwd(), gtn_config["working_dir"]))

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

