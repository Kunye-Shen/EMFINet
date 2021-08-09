# data loader
from __future__ import print_function, division
import torch
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset

class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def __call__(self,sample):
		image, label, edge = sample['image'],sample['label'],sample['edge']

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
		lbl = transform.resize(label,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)
		edg = transform.resize(edge,(self.output_size,self.output_size),mode='constant', order=0, preserve_range=True)

		return {'image':img,'label':lbl,'edge':edg}

class ToTensorLab(object):
	def __init__(self,flag=0):
		self.flag = flag

	def __call__(self, sample):

		image, label, edge = sample['image'], sample['label'],sample['edge']

		tmpLbl = np.zeros(label.shape)
		tmpedg = np.zeros(edge.shape)

		if(np.max(label)<1e-6):
			label = label
		else:
			label = label/np.max(label)
		if(np.max(edge)<1e-6):
			edge = edge
		else:
			edge = edge/np.max(edge)

		# change the color space
		tmpImg = np.zeros((image.shape[0],image.shape[1],3))
		image = image/np.max(image)
		if image.shape[2]==1:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
		else:
			tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
			tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
			tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpLbl[:,:,0] = label[:,:,0]
		tmpedg[:,:,0] = edge[:,:,0]
		
		tmpImg = tmpImg.transpose((2, 0, 1))
		tmpLbl = label.transpose((2, 0, 1))
		tmpedg = edge.transpose((2, 0, 1))

		return {'image': torch.from_numpy(tmpImg),'label': torch.from_numpy(tmpLbl),'edge': torch.from_numpy(tmpedg)}

class SalObjDataset(Dataset):
	def __init__(self,img_name_list,lbl_name_list,edge_name_list,transform=None):
		self.image_name_list = img_name_list
		self.label_name_list = lbl_name_list
		self.edge_name_list = edge_name_list
		self.transform = transform

	def __len__(self):
		return len(self.image_name_list)

	def __getitem__(self,idx):

		image = io.imread(self.image_name_list[idx])

		if(0==len(self.label_name_list)):
			label_3 = np.zeros(image.shape)
			edge_3 = np.zeros(image.shape)
		else:
			label_3 = io.imread(self.label_name_list[idx])
			edge_3 = io.imread(self.edge_name_list[idx])

		label = np.zeros(label_3.shape[0:2])
		edge = np.zeros(edge_3.shape[0:2])
		if(3==len(label_3.shape)):
			label = label_3[:,:,0]
			edge = edge_3[:,:,0]
		elif(2==len(label_3.shape)):
			label = label_3
			edge = edge_3

		if(3==len(image.shape) and 2==len(label.shape)):
			label = label[:,:,np.newaxis]
			edge = edge[:,:,np.newaxis]
		elif(2==len(image.shape) and 2==len(label.shape)):
			image = image[:,:,np.newaxis]
			label = label[:,:,np.newaxis]
			edge = edge[:,:,np.newaxis]

		sample = {'image':image, 'label':label, 'edge':edge}

		if self.transform:
			sample = self.transform(sample)

		return sample
