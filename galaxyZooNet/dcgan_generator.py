import os
import time
import pickle
import pandas as pd
#from tqdm import tqdm
from tqdm.notebook import tqdm

from galaxyZooNet.base import BaseTrainer
from galaxyZooNet.data_kits import GalaxyZooDataset, transforms_DCGAN
from galaxyZooNet.models.dcgan import Generator, Discriminator, weights_init

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.utils as vutils

class DCGAN_Generator(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)
        self._prepare_data()
        self._build_model(config)
        self._define_loss()
        self._init_optimizer()

        self.Niters_per_epoch = (len(self.dataset) + self.batch_size - 1) // self.batch_size
        self.fixed_noise = torch.randn(self.batch_size, self.n_zlatent, 1, 1, device=self.device)
        
        # init counter
        self.current_iteration = 0
        self.current_epoch = 0

        self.real_label = 1.
        self.fake_label = 0.
    
    def _prepare_data(self):

        self.df = pd.read_csv(self.file_csv)

        self.transform = transforms_DCGAN(self.input_size, self.crop_size, self.norm_mean, self.norm_std)

        self.dataset = GalaxyZooDataset(self.df, self.dir_image, transform=self.transform, label_tag=self.label_tag)

        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.workers)

        print('\n------ Prepare Data ------\n')
        print(f'Number of training galaxies: {len(self.dataset)} ({len(self.dataloader)} batches)')
    
    def _build_model(self, config):

        self.netG = Generator(config).to(self.device)
        self.netD = Discriminator(config).to(self.device)

        self.netG.apply(weights_init)
        self.netD.apply(weights_init)

        print('\n------ Build Model ------\n')
        print('Number of trainable parameters')
        print('Generator  :', sum(param.numel() for param in self.netG.parameters() if param.requires_grad))
        print('Discriminator  :', sum(param.numel() for param in self.netG.parameters() if param.requires_grad))
    
    def _define_loss(self):
        self.loss = nn.BCELoss().to(self.device)

    def _init_optimizer(self):
        # define optimizers for both generator and discriminator
        self.optimG = torch.optim.Adam(self.netG.parameters(), lr=self.lr_g,
                                       betas=(self.beta1, self.beta2))
        self.optimD = torch.optim.Adam(self.netD.parameters(), lr=self.lr_d,
                                       betas=(self.beta1,self.beta2))

    def _init_storage(self):
        '''initialize storage dictionary and directory to save training information'''

        # ------ create storage directory ------
        print('\n------ Create experiment directory ------\n')
        try:
            os.makedirs(self.dir_exp)
        except (FileExistsError, OSError) as err:
            raise FileExistsError(
                f'Default save directory {self.dir_exp} already exit. Change exp_name!') from err
        print(f'Training information will be stored at :\n \t {self.dir_exp}\n')

        # ------ trainInfo ------
        save_key = ['loss_G', 'loss_D', 'epoch_loss_G', 'epoch_loss_D', 'img_list', 'lr']
        self.trainInfo = {}
        for key in save_key:
            self.trainInfo[key] = []

        # ------ stateInfo ------
        # self.statInfo = {}
    
    def _save_checkpoint(self):

        with open(self.file_trainInfo, 'wb') as handle:
            pickle.dump(self.trainInfo, handle)
        
        self.statInfo = {
            'epoch': self.current_epoch,
            'iteration': self.current_iteration,
            'G_state_dict': self.netG.state_dict(),
            'D_state_dict': self.netD.state_dict(),
            'G_optimizer': self.optimG.state_dict(),
            'D_optimizer': self.optimD.state_dict(),
            'fixed_noise': self.fixed_noise
        }

        #outfile_statInfo = os.path.join(self.dir_exp, f'stateInfo_{self.current_iteration}.pth')
        outfile_statInfo = os.path.join(self.dir_exp, f'stateInfo_{self.current_epoch}.pth')
        torch.save(self.statInfo, outfile_statInfo)

    def _train_one_epoch(self):

        #tqdm_batch = tqdm(self.dataloader, total=self.Niters_per_epoch, desc=f'epoch-{self.current_epoch+1}-')

        self.netG.train()
        self.netD.train()

        running_loss_G = 0.0
        running_loss_D = 0.0

        since = time.time()
        #for id_batch, x in enumerate(tqdm_batch):
        for id_batch, x in enumerate(self.dataloader):
            #gal_label = x[1]
            x = x[0].to(self.device)
            y = torch.randn(x.size(0), ).to(self.device)
            fake_noise = torch.randn(x.size(0), self.n_zlatent, 1, 1, device=self.device)

            # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            self.netD.zero_grad()
            D_real_out = self.netD(x)
            y.fill_(self.real_label)
            loss_D_real = self.loss(D_real_out, y)
            loss_D_real.backward()
            D_x = D_real_out.mean().item()

            # train with fake
            G_fake_out = self.netG(fake_noise)
            y.fill_(self.fake_label)

            D_fake_out = self.netD(G_fake_out.detach())
            loss_D_fake = self.loss(D_fake_out, y)
            loss_D_fake.backward()
            D_G_z1 = D_fake_out.mean().item()
            
            loss_D = loss_D_fake + loss_D_real
            self.optimD.step()

            # Update G network: maximize log(D(G(z)))
            self.netG.zero_grad()
            y.fill_(self.real_label)
            D_out = self.netD(G_fake_out)
            loss_G = self.loss(D_out, y)
            loss_G.backward()
            D_G_z2 = D_out.mean().item()

            self.optimG.step()

            # Pring training logs & output generating galaxies
            if id_batch % 500 == 0:
                print(
                    f'[{self.current_epoch+1}/{self.num_epochs} epoch] [{self.current_iteration+1} iter]\t loss_D: {loss_D.item():.3f} \t loss_G: {loss_G.item():.3f}\t D(x):{D_x:.3f} \t D(G(z)): {D_G_z1:.3f} / {D_G_z2:.3f}')
            
            # write statistics
            self.trainInfo['loss_G'].append(loss_G.item())
            self.trainInfo['loss_D'].append(loss_D.item())
            
            running_loss_G += loss_G.item() * x.size(0)
            running_loss_D += loss_D.item() * x.size(0)

            if id_batch % 500 == 0:
                with torch.no_grad():
                    gen_out = self.netG(self.fixed_noise).detach().cpu()
                self.trainInfo['img_list'].append(vutils.make_grid(gen_out, padding=2, normalize=True, nrow=self.nrow))

            self.current_iteration += 1

        # --- END for loop for 1 epoch ---
        #tqdm_batch.close()

        avg_epoch_loss_G = running_loss_G/len(self.dataset)
        avg_epoch_loss_D = running_loss_D/len(self.dataset)
        
        self.trainInfo['epoch_loss_G'].append(avg_epoch_loss_G)
        self.trainInfo['epoch_loss_D'].append(avg_epoch_loss_D)

        print(
            f'\tTraining at epoch-{self.current_epoch} | avg. netD loss: {avg_epoch_loss_D:.3f} | avg. netG loss {avg_epoch_loss_G:.3f}')

        time_elapsed = time.time() - since
        print(f'\tRun time per epoch: {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')

        
    def train(self):
        
        self.current_epoch = 0
        self._init_storage()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.current_epoch = epoch
            self._train_one_epoch()
            self._save_checkpoint()


    def gen_galaxy(self):
        pass
