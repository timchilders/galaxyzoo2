import os, shutil
import glob
from tqdm import tqdm

import pandas as pd
import numpy as np
from numpy.random import choice
import matplotlib.pyplot as plt

from skimage import io
from skimage.transform import resize, rotate
from skimage.util import img_as_ubyte


def train_val_test(df, data_dir, img_dir, train_ratio = 0.6, val_ratio = 0.2, test_ratio=0.2, shape=(64,64)):
    train_dir = data_dir+'/train'
    val_dir = data_dir+'/validation'
    test_dir = data_dir+'/test'

    train_ids=[]
    val_ids=[]
    test_ids=[]

    filenames = glob.glob(img_dir+'*.jpg')
    for name in tqdm(filenames):
        # print(f'filename: {name}')
        img_id = (name.split('.')[0]).split('/')[2]
        # print(f'id: {img_id}')
        img = process_img(name,shape)
        split_dir = choice([train_dir,val_dir,test_dir], 1, p = [train_ratio,val_ratio,test_ratio])[0]
        # print(f'split directory: {split_dir}')
        if split_dir==train_dir:
            train_ids.append(int(img_id))
        elif split_dir==val_dir:
            val_ids.append(int(img_id))
        else:
            test_ids.append(int(img_id))
        img_uint8 = img_as_ubyte(img) #convert to uint8
        io.imsave(split_dir+'/'+img_id+'.jpg',img_uint8)
    make_targets(df, train_dir, val_dir, test_dir, train_ids,val_ids,test_ids)
  

def make_dir(base_dir):
    '''
    Creates subfolders for train, test, validate images

    Inputs:
        base_dir: root image folder to create subdir in.
    '''
    train_dir = os.path.join(base_dir,'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

def process_img(path,shape):
    img = plt.imread(path)
    img = img[106:106*3,106:106*3]
    img = resize(img,shape)
    return img

def make_targets(df, train_dir, val_dir, test_dir, train_ids,val_ids,test_ids):
    '''
    Creates target dataframes for each subset of images
    '''
    train_df = pd.DataFrame({'asset_id':train_ids})
    train_df = pd.merge(train_df,df, on='asset_id', how='left')
    train_df.to_csv(train_dir+'/train.csv', index=False)
    
    val_df = pd.DataFrame({'asset_id':val_ids})
    val_df = pd.merge(val_df,df, on='asset_id', how='left')
    val_df.to_csv(val_dir+'/val.csv',index=False)

    test_df = pd.DataFrame({'asset_id':test_ids})
    test_df = pd.merge(test_df,df, on='asset_id', how='left')
    test_df.to_csv(test_dir+'/test.csv',index=False)
    

if __name__ == '__main__':
    
    data_dir = 'data/img'
    img_dir = 'images_gz2/images/'
    make_dir(data_dir) #make initial img directories

    df = pd.read_csv('data/gz2_debiased.csv')
    shape = (64,64) #shape to resize images to

    #begin image processing
    train_val_test(df, data_dir, img_dir, train_ratio = 0.6, val_ratio = 0.2, 
                    test_ratio=0.2, shape=(64,64))
