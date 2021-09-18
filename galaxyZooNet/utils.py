import yaml
from easydict import EasyDict


def get_config_from_yaml(file_yaml):
    '''Get the config from a yaml file
        Args:
            file_yaml: path to the config yaml file
        Return:
            config (EasyDict)
    '''

    with open(file_yaml, 'r') as file_config:
        try:
            config = EasyDict(yaml.safe_load(file_config))
            return config
        except ValueError:
            print("INVALID yaml file format.")
            exit(-1)



if __name__=='__main__':
    config = get_config_from_yaml('../configs/resnet50_test.yaml')