from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
import warnings
warnings.filterwarnings("ignore")
       

class RadioUNet_3d(Dataset):
    def __init__(self,maps_inds=np.ones(1),phase="train",
                 ind1=1,ind2=1, 
                 dir_dataset="/",
                 numTx=16,                  
                 thresh=0.2,
                 transform= transforms.ToTensor()):
        print("phase",phase)
        
        if phase=="train":
            self.ind1=1
            self.ind2=161
        elif phase=="val":
            self.ind1=161
            self.ind2=181
        elif phase=="test":
            self.ind1=181
            self.ind2=201
            
        if maps_inds.size==1:
            self.maps_inds=np.arange(self.ind1,self.ind2,1,dtype=np.int16)
            #Determenistic "random" shuffle of the maps:
            np.random.seed(42)
            np.random.shuffle(self.maps_inds)
        else:
            self.maps_inds=maps_inds
        
        
        
        self.dir_dataset = dir_dataset
        self.numTx=  numTx                
        self.thresh=thresh
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "antennas/antennas" 
        
        self.height = 256
        self.width = 256
        self.depth=16
        
        self.length=(self.ind2-self.ind1)*self.numTx
        

    # length     
    def __len__(self):
        print("__len__",(self.ind2-self.ind1)*self.numTx)
        return (self.ind2-self.ind1)*self.numTx
    
    
    def __getitem__(self,idx):
        addidx=(self.ind1-1)*16+idx+1
        build_num=np.ceil(addidx/self.numTx).astype(int) #idx/16
        site_num=addidx-(build_num-1)*self.numTx
        
        img_gainlist=[]
        img_name_buildings=self.dir_dataset+"buildings/wall"+str(build_num)+".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))/255  
        
        #load furniture:
        img_name_furniture=self.dir_dataset+"buildings/furniture"+str(build_num)+".png"
        image_furniture = ((20*(np.asarray(io.imread(img_name_furniture))/255))+1)/31 
        
        #Load Tx (transmitter):
        img_name_Tx = self.dir_dataset+"antennas/antennas"+str(build_num)+"_"+str(site_num)+".png"
        image_Tx = np.asarray(io.imread(img_name_Tx))/255 
        
        #Load pathloss:
        for i in range(self.depth):
            img_name_gain = self.dir_dataset+"/3D indoor radio map/pathloss"+str(build_num)+"_"+str(site_num)+"/pathloss"+str(build_num)+"_"+str(site_num)+"_"+str(i)+".png"  
            image_gain = np.asarray(io.imread(img_name_gain))/255
            
            #pathloss threshold transform
            if self.thresh>0:
                mask = image_gain < self.thresh
                image_gain[mask]=self.thresh
                image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
                image_gain=image_gain/(1-self.thresh)
            img_gainlist.append(image_gain)

        img_gain_stack=np.stack(img_gainlist,axis=2)            

        inputs=np.stack([image_buildings, image_furniture, image_Tx], axis=2)
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(img_gain_stack).type(torch.float32)
            
        return [inputs, image_gain]