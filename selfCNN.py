"""
Script for a new CNN architecture, testing different parameters
Pauline Bock
9-02-2019
"""

import sys
import os
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from torchvision import transforms,datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *


#Rename the folder to replace space by _
def read_folder(path):
    files=os.listdir(path)
    for name in files:
        if name.find(' ') != -1:
            os.rename(path + '/' + name,path + '/' + name.replace(' ','_'))

path_train= 'fruits-360/train'
path_test= 'fruits-360/val'

#Call the function to rename subfolders
read_folder(path_train)
read_folder(path_test)


#create the datasets
train_dataset=datasets.ImageFolder(path_train,transform=transforms.ToTensor())
train_loader=DataLoader(train_dataset,batch_size = 300,shuffle=True) #batch_size => prendre 4 images de chaque dossier
test_dataset=datasets.ImageFolder(path_test,transform=transforms.ToTensor())
test_loader=DataLoader(test_dataset,batch_size = 120,shuffle=True)

classes = ('Apple_Braeburn','Apple_Golden_1','Apple_Golden_2','Apple_Golden_3','Apple_Granny_Smith','Apple_Red_1','Apple_Red_2','Apple_Red_3','Apple_Red_Delicious','Apple_Red_Yellow','Apricot','Avocado','Avocado_ripe','Banana','Banana_Red','Cactus_Fruit','Cantaloupe_1','Cantaloupe_2','Carambula','Cherry_1','Cherry_2','Cherry_Rainier','Cherry_Wax_Black','Cherry_Wax_Red','Cherry_Wax_Yellow','Clementine','Cocos','Dates','Granadilla','Grape_Blue','Grape_Pink','Grape_White','Grape_White_2','Grapefruit_Pink','GrapeFruit_White','Guava','Huckleberry','Kaki','Kiwi','Kumquats','Lemon','Lemon_Meyer','Limes','Lychee','Mandarine','Mango','Maracuja','Melon_Piel_de_Sapo','Mulberry','Nectarine','Orange','Papaya','Passion_Fruit','Peach','Peach_Flat','Pear','Pear_Abate','Pear_Monster','Pear_Williams','Pepino','Physalis','Physalis_with_Husk','Pineapple','Pineapple_Mini','Pitahaya_Red','Plum','Pomegranate','Quince','Rambutan','Raspberry','Redcurrant','Salak','Strawberry','Strawberry_Wedge','Tamarillo','Tangelo','Tomato_1','Tomato_2','Tomato_3','Tomato_4','Tomato_Cherry_Red','Tomato_Maroon','Walnut') 

#CNN architecture
class Net(nn.Module):
	'''7.Define the layers in the network'''
	def __init__(self):
		super(Net,self).__init__()

		#1 input image channel,6 output channels,5x5 square convolution kernel
		#2 input image channel,16 output channels,5x5 square convolution kernel
		self.conv1=nn.Conv2d(3,6,5)
		self.conv2=nn.Conv2d(6,16,5)

		#(input features, output features)
		self.fc1=nn.Linear(16*22*22,120)
		self.fc2=nn.Linear(120,84)
		self.fc3=nn.Linear(84,83)

	def forward(self,x):
		x = F.relu(self.conv1(x))
		x = F.max_pool2d(x,(2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)),2)

		x=x.view(-1,self.num_flat_features(x))
		x=F.relu(self.fc1(x))
		x=F.relu(self.fc2(x))
		x=self.fc3(x)
		return x

		
	def num_flat_features(self,x):
		size=x.size()[1:]
		#all dimensions except the batch dimension
		num_features=1
		for s in size:
			num_features*=s
		return num_features

def imshow(img):
	img=img/2+0.5
	npimg=img.numpy()
	plt.imshow(np.transpose(npimg,(1,2,0)))
	
#check GPU on machine
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
net=Net().to(device)
print(net)

#Define the loss function
criterion = nn.CrossEntropyLoss()

#Define the optimizer : different optimizers tested
#optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
optimizer=optim.Adam(net.parameters(),lr=0.001)

epochs=10
for epoch in range(epochs):
	print ("epoch #", epoch)
	running_loss=0.0
	for i, data in enumerate(train_loader,0):
		inputs,labels=data
		inputs,labels= inputs.to(device),labels.to(device)
		optimizer.zero_grad()
		
		#train
		output=net(inputs)
		loss=criterion(output,labels)
		#back propagation
		loss.backward()
		optimizer.step()

		running_loss+=loss.item()
		if (i % 100 == 99): #print every 2000 mini-batches
				print ('[%d,%5d] loss: %.3f' % (epoch+1,i+1,running_loss/2000))
				running_loss=0.0
print ('Finished Training')


#Test
dataiter=iter(test_loader)
images,labels=dataiter.next()

images,labels=images.to(device),labels.to(device)

outputs=net(images)
_,predicted=torch.max(outputs,1)

correct=0
total=0
cpt = 0
with torch.no_grad():
	for data in test_loader:
		cpt+=1
		print (cpt)
		images,labels=data
		images,labels=images.to(device),labels.to(device)
		outputs=net(images).to(device)
		
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()


print ('Accuracy of classification: %d %%' %(100* correct / total))
