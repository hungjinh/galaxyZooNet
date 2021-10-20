import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def data_split(file_csv, f_train=0.64, f_valid=0.16, f_test=0.20, random_state=None, stats=False, label_tag='label_8'):
    '''train-valid-test splits
        Args:
            file_csv (str) : path to the full catalog csv file
            f_train, f_valid, f_test : fractions of training, validation, test samples
            stats (bool): display splitting statistics if True 
        Returns:
            df_train (pd.dataframes) : splitted training sample
            df_valid (pd.dataframes) :          validation
            df_test  (pd.dataframes) :          test
    '''
    assert f_train + f_valid + f_test == 1, 'fractions have to sum to 1.'

    df = pd.read_csv(file_csv)

    df_train, df_temp = train_test_split(df, train_size=f_train, random_state=random_state)
    relative_f_valid = f_valid/(f_valid+f_test)
    df_valid, df_test = train_test_split(df_temp, train_size=relative_f_valid, random_state=random_state)
    
    if stats:
        df_stats=df.groupby([label_tag])[label_tag].agg('count').to_frame('count').reset_index()
        df_stats['full'] = df_stats['count']/df_stats['count'].sum()
        df_stats['train'] = df_train.groupby([label_tag]).size()/df_train.groupby([label_tag]).size().sum()
        df_stats['valid'] = df_valid.groupby([label_tag]).size()/df_valid.groupby([label_tag]).size().sum()
        df_stats['test'] = df_test.groupby([label_tag]).size()/df_test.groupby([label_tag]).size().sum()
        
        ax = df_stats.plot.bar(x=label_tag, y=['full', 'train', 'valid', 'test'], rot=0)
        ax.set_ylabel('class fraction')
    
    return df_train.reset_index(drop=True), df_valid.reset_index(drop=True), df_test.reset_index(drop=True)

class GalaxyZooDataset(Dataset):
    '''Galaxy Zoo 2 image dataset
        Args:
            dataframe (pd.dataframe): outputs from the data_split function
                e.g. df_train / df_valid / df_test
            dir_image (str): path to the galaxy images directory
            label_tag (str): class label system to be used for training
                e.g. label_tag = 'label1' / 'label2' / 'label3' / 'label4'
    '''

    def __init__(self, dataframe, dir_image, label_tag='label1', transform=None):
        self.df = dataframe
        self.transform = transform
        self.dir_image = dir_image
        self.label_tag = label_tag

    
    def __getitem__(self, index):
        galaxyID = self.df.iloc[[index]].galaxyID.values[0]
        file_img = os.path.join(self.dir_image, str(galaxyID) + '.jpg')
        image = Image.open(file_img)

        if self.transform:
            image = self.transform(image)
        
        label = self.df.iloc[[index]][self.label_tag].values[0]

        return image, label

    def __len__(self):
        return len(self.df)


def transforms_galaxy(input_size=224, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    '''Create Pytorch data transforms for galaxy images'''

    transform = {}
    transform['train'] = transforms.Compose([transforms.CenterCrop(input_size),
                                              transforms.RandomRotation(90),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(norm_mean, norm_std)])

    transform['valid'] = transforms.Compose([transforms.CenterCrop(input_size),
                                              transforms.RandomRotation(90),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomVerticalFlip(),
                                              transforms.RandomResizedCrop(input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                                              transforms.ToTensor(),
                                             transforms.Normalize(norm_mean, norm_std)])
    
    transform['test'] = transforms.Compose([transforms.CenterCrop(input_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize(norm_mean, norm_std)])

    return transform


def transforms_DCGAN(input_size=64, crop_size=224, norm_mean=[0.485, 0.456, 0.406], norm_std=[0.229, 0.224, 0.225]):
    
    transform = transforms.Compose([transforms.CenterCrop(crop_size),
                                    transforms.Resize(input_size),
                                    transforms.RandomRotation(90),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomResizedCrop(
                                        input_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    return transform