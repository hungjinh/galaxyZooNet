import os
import pickle
import torch

class BaseTrainer():
    def __init__(self, config):

        self._setup(config)
    
    def _setup(self, config):

        print('\n------ Parameters ------\n')
        for key in config:
            setattr(self, key, config[key])
            print(f'{key} :', config[key])

        self.is_cuda = torch.cuda.is_available()
        self.cuda = self.is_cuda & self.cuda
        self.device = torch.device(f'cuda:{self.gpu_device}' if self.cuda else "cpu")

        self.dir_exp = os.path.join(self.dir_output, self.exp_name)
        self.file_trainInfo = os.path.join(self.dir_exp, 'trainInfo.pkl')
        self.file_stateInfo = os.path.join(self.dir_exp, 'stateInfo.pth')

    def _init_storage(self):
        '''initialize storage dictionary and directory to save training information'''

        # ------ create storage directory ------
        print('\n------ Create experiment directory ------\n')
        try:
            os.makedirs(self.dir_exp)
        except (FileExistsError, OSError) as err:
            raise FileExistsError(
                f'Default save directory {self.dir_exp} already exit. Change exp_name!') from err
        print(f'Training information will be stored at :\n \t {self.dir_exp}')

        # ------ trainInfo ------
        save_key = ['train_loss', 'train_acc', 'valid_loss', 'valid_acc',
                    'epoch_train_loss', 'epoch_train_acc', 'epoch_valid_loss', 'epoch_valid_acc', 'lr']
        self.trainInfo = {}
        for key in save_key:
            self.trainInfo[key] = []

        # ------ stateInfo ------
        self.statInfo = {}

    def _save_checkpoint(self):

        # ------ stateInfo ------
        with open(self.file_trainInfo, 'wb') as handle:
            pickle.dump(self.trainInfo, handle)

        self.statInfo['epoch'] = self.current_epoch
        self.statInfo['model_state_dict'] = self.model.state_dict()
        self.statInfo['optimizer_state_dict'] = self.optimizer.state_dict()
        self.statInfo['best_epoch'] = self.best_epoch
        self.statInfo['best_model_wts'] = self.best_model_wts
        torch.save(self.statInfo, self.file_stateInfo)

    def _load_checkpoint(self):

        self.statInfo = torch.load(self.file_stateInfo)
        self.trainInfo = pickle.load(open(self.file_trainInfo, 'rb'))
