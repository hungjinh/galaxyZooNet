import time
import copy

from galaxyZooNet.base import BaseTrainer
from galaxyZooNet.data_kits import data_split, GalaxyZooDataset, transforms_galaxy
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torchvision
import torch.optim as optim

class ResNet50_Classifier(BaseTrainer):
    def __init__(self, config):
        
        super().__init__(config)
        self._prepare_data()
        self._build_model(self.pretrained)
        self._define_loss(weight=self.class_weights)
        self._init_optimizer()

    def _gen_Dset_Dloader(self, df, transform):
        dataset = GalaxyZooDataset(df, self.dir_image, transform=transform, label_tag=self.label_tag)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)
        return dataset, dataloader

    def _prepare_data(self):
        
        self.df = {}
        self.df['train'], self.df['valid'], self.df['test'] = data_split(self.file_csv, 
                                                                        self.f_train, self.f_valid, self.f_test,
                                                                        random_state=self.seed, stats=False)
        self.transform = transforms_galaxy(self.input_size, self.norm_mean, self.norm_std)

        self.dataset = {}
        self.dataloader = {}
        for key in ['train', 'valid', 'test']:
            self.dataset[key], self.dataloader[key] = self._gen_Dset_Dloader(df=self.df[key], transform=self.transform[key])
                                              
        print('\n------ Prepare Data ------\n')
        for key in ['train', 'valid', 'test']:
            print(f'Number of {key} galaxies: {len(self.dataset[key])} ({len(self.dataloader[key])} batches)')
    
    def _build_model(self, pretrained):
        self.model = torchvision.models.resnet50(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, self.num_classes)
        self.model = self.model.to(self.device)

        print('\n------ Build Model ------\n')
        print('Number of trainable parameters')
        print('layer1 :', sum(param.numel() for param in self.model.layer1.parameters() if param.requires_grad))
        print('layer2 :', sum(param.numel() for param in self.model.layer2.parameters() if param.requires_grad))
        print('layer3 :', sum(param.numel() for param in self.model.layer4.parameters() if param.requires_grad))
        print('layer4 :', sum(param.numel() for param in self.model.layer3.parameters() if param.requires_grad))
        print('fc     :', sum(param.numel() for param in self.model.fc.parameters() if param.requires_grad))
        print('TOTAL  :', sum(param.numel() for param in self.model.parameters() if param.requires_grad))
    
    def _define_loss(self, weight=None):
        '''
            Args:
                weight (list) : weigh factors associated with morphology classes
        '''
        if weight:
            weight = torch.FloatTensor(weight).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight).to(self.device)
    
    def _init_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def _train_one_epoch(self):
        since = time.time()

        print(f'Epoch {self.current_epoch+1}/{self.num_epochs}')
        for phase in ['train', 'valid']:
            if phase == 'train':
                self.model.train()  # Set model to training mode
            else:
                self.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in self.dataloader[phase]:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
            
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = self.model(inputs)
                    _, preds = torch.max(outputs, 1)
                    #_, preds = torch.max(outputs.detach(), 1)
                    loss = self.criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        self.optimizer.step()
                # keep statstics
                self.trainInfo[f'{phase}_loss'].append( loss.item() )
                self.trainInfo[f'{phase}_acc'].append( torch.sum(preds==labels.data).double() / inputs.size(0) )

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # ------ End training the epoch in train/valid phase ------
            if phase == 'train':
                self.scheduler.step()

            epoch_loss = running_loss / len(self.dataset[phase])
            epoch_acc = running_corrects.double() / len(self.dataset[phase])

            self.trainInfo[f'epoch_{phase}_loss'].append(epoch_loss)
            self.trainInfo[f'epoch_{phase}_acc'].append(epoch_acc)

            print(f'\t{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} ', end='')

            # deep copy the model
            if phase == 'valid' and epoch_acc > self.best_acc:
                self.best_acc = epoch_acc
                self.best_epoch = self.current_epoch
                self.best_model_wts = copy.deepcopy(self.model.state_dict())

        # ------ End training the ephch in both train & valid phases ------ 
        self.trainInfo['lr'].append(self.scheduler.get_last_lr()[0]) # save lr / epoch
        time_elapsed = time.time() - since
        print(f'\tTime: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

    def train(self):
        
        self.current_epoch = 0
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        self.best_acc = 0.0
        self._init_storage()

        print('\n------ Training ------\n')
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._save_checkpoint()

            if self.current_epoch - self.best_epoch >= self.early_stop_threshold:
                print(f'Early stopping... (Model did not imporve after {self.early_stop_threshold} epochs)')
                break
            
        print(f'Best accuracy of {self.best_acc} reached at epoch {self.best_epoch+1}.')
    
    def _test_loop(self, dataset, dataloader):

        print(f'  Number of galaxies: {len(dataset)} ({len(dataloader)} batches)')

        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.long().to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        test_loss = running_loss / len(dataset)
        test_acc = running_corrects / len(dataset)

        print(f'\ttest Loss: {test_loss:.4f}\t test Accuracy: {test_acc:.4f}')

        return test_loss, test_acc
           
    def test(self, mode='full'):
        '''Run test on the best-trained model.
            Args:
                mode = 'full' or 'separate'
                    'full' -> run test on the full test dataset
                    'separate' -> run test on each galaxy morphology class separately
        '''

        self._load_checkpoint()
        self.model.load_state_dict(self.statInfo['best_model_wts'])
        self.model.eval()
        
        if mode == 'full':
            print(f'\n------ Run test on the full test dataset ({self.num_classes} galaxy classes) ------\n')
            test_loss, test_acc = self._test_loop(dataset=self.dataset['test'], dataloader=self.dataloader['test'])

        elif mode == 'separate':
            print(f'\n------ Run test on each galaxy morphology class separately ------\n')

            test_loss = [0.]*self.num_classes
            test_acc = [0.]*self.num_classes
            for classID in range(self.num_classes):
                print(f'Class {classID} galaxies')
                df_class = self.df['test'][self.df['test'][self.label_tag] == classID]
                dataset_c, dataloader_c = self._gen_Dset_Dloader(df_class, transform=self.transform['test'])
                test_loss[classID], test_acc[classID] = self._test_loop(dataset=dataset_c, dataloader=dataloader_c)
        else:
            raise ValueError('Invalid mode. Set mode=\'full\' or \'separate\'.')

        return test_loss, test_acc


if __name__ == '__main__':
    from galaxyZooNet.utils import get_config_from_yaml
    config = get_config_from_yaml('../configs/resnet50_test.yaml')
    classifier = ResNet50_Classifier(config=config)
