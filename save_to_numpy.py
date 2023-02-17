import numpy as np
import pandas as pd
import os
from pathlib import Path
import glob
import re
from tqdm import tqdm
from pyper.chemometrics.dataset import HyperSpectralDataset




#dataloader

imagedir=Path('E:\dataset')
filepath=pd.Series(list(imagedir.glob(r'**/FX10/*.h5')), name='Filepath').astype(str)
#determine the amount of seeds and seedtype
seedsnbr= pd.Series(filepath.apply(lambda x: os.path.split(x)[1]), name='#seeds').astype(str)
seedstype= pd.Series(filepath.apply(lambda x: os.path.split(x)[1]), name='type seeds').astype(str)
seedsnbr=pd.Series(seedsnbr.apply(lambda x: x.replace(".png","") )).astype(str)
seedstype=pd.Series(seedstype.apply(lambda x: re.split('(\d+)',x)[0] )).astype(str)
seedsnbr=pd.Series(seedsnbr.apply(lambda x: re.split('(\d+)',x)[1] )).astype(np.int32)
# determine the batch number
batchnbr=pd.Series(filepath.apply(lambda x: os.path.split(x)[0]), name='batch').astype(str)
batchnbr=pd.Series(batchnbr.apply(lambda x: os.path.split(x)[0]), name='batch').astype(str)
batchnbr_0=pd.Series(batchnbr.apply(lambda x: os.path.split(x)[1]), name='batch').astype(str)
batchnbr_0=pd.Series(batchnbr_0.apply(lambda x: re.split('(\d+)',x)[1] )).astype(np.int32)
# determine if mixture or pure
mixorpur=pd.Series(batchnbr.apply(lambda x: os.path.split(x)[0]), name='type').astype(str)
mixorpur=pd.Series(mixorpur.apply(lambda x: os.path.split(x)[1]), name='type').astype(str)
#makes sure that the amount of seeds is zero if dealing with pure sample
seedsnbr[mixorpur=='pure']=0

#make a list with all the data
dataset=pd.concat([filepath,seedsnbr,batchnbr_0,mixorpur,seedstype],axis=1)
dataset

#create new filepath and select bands:

bands=[100,50,5]

filepath2='E:/dataset_numpy/'+batchnbr_0.astype(str)+'_'+(filepath.apply(lambda x: os.path.split(x)[1])).astype(str)
filepath2=(filepath2.apply(lambda x: x.replace('h5','npy'))).astype(str)
print(filepath2)

for i in tqdm(filepath):
    cube=HyperSpectralDataset.load(i)
    npcube=HyperSpectralDataset.get_HC(cube)
    npcube_0=npcube[:,:,bands]
    idx=filepath[filepath == i]
    print(type(cube))
    np.save(filepath2[idx.index[0]],npcube_0)
    


