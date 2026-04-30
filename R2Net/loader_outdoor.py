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
                 numTx=80,                  
                 thresh=0.2,
                 simulation="DPM",
                 carsSimul="no",
                 carsInput="no",
                 IRT2maxW=1,
                 cityMap="complete",
                 missing=1,
                 transform= transforms.ToTensor()):
        print("phase",phase)
 
    
        
            
        if phase=="train":
            self.ind1=0
            self.ind2=500
        elif phase=="val":
            self.ind1=500
            self.ind2=600
        elif phase=="test":
            self.ind1=600
            self.ind2=701
            
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
        
        self.simulation=simulation
        self.carsSimul=carsSimul
        self.carsInput=carsInput

        
        self.IRT2maxW=IRT2maxW
        
        self.cityMap=cityMap
        self.missing=missing

            
              
        self.transform= transform
        
        self.dir_Tx = self.dir_dataset+ "antennas/antennas" 
        
        self.height = 256
        self.width = 256
        
        self.length=(self.ind2-self.ind1)*self.numTx
        

    # length     
    def __len__(self):
        print("__len__",(self.ind2-self.ind1)*self.numTx)
        return (self.ind2-self.ind1)*self.numTx
    
    
    def __getitem__(self,idx):
        addidx=(self.ind1)*self.numTx+idx
        build_num=np.floor(addidx/self.numTx).astype(int) #idx/16
        site_num=addidx-(build_num)*self.numTx
        
        img_name_buildings=self.dir_dataset+"RadioMapSeer/png/buildings_complete/"+str(build_num)+".png"
        image_buildings = np.asarray(io.imread(img_name_buildings))/255  
        
        #Load Tx (transmitter):
        img_name_Tx = self.dir_dataset+"RadioMapSeer/png/antennas/"+str(build_num)+"_"+str(site_num)+".png"
        image_Tx = 1.5*((np.asarray(io.imread(img_name_Tx))/255)/25) 
        
        #Load pathloss:
        img_name_gain = self.dir_dataset+"RadioMapSeer/gain/DPM/"+str(build_num)+"_"+str(site_num)+".png" 
        image_gain = np.asarray(io.imread(img_name_gain))/255
        if self.thresh>0:
            mask = image_gain < self.thresh
            image_gain[mask]=self.thresh
            image_gain=image_gain-self.thresh*np.ones(np.shape(image_gain))
            image_gain=image_gain/(1-self.thresh)
            

        inputs=np.stack([image_buildings, image_Tx], axis=2)
        if self.transform:
            inputs = self.transform(inputs).type(torch.float32)
            image_gain = self.transform(image_gain).type(torch.float32)
        return [inputs, image_gain]
 