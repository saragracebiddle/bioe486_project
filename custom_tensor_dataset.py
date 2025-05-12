# import the necessary packages
from torch.utils.data import Dataset
from plotting import generate_target
import os
import config

def list_files_walk(start_path='.'):
	out = []

	for root, dirs, files in os.walk(start_path):
		for file in files:
			out.append(file)


	return out

class CustomTensorDataset(object):
	# initialize the constructor
	def __init__(self, transforms=None):
		#self.tensors = tensors
		self.transforms = transforms
		self.imgs = list_files_walk('mias_data/')

	def __getitem__(self, index):
		# grab the image, label, and its bounding box coordinates
		#image = self.tensors[0][index]
		#label = self.tensors[1][index]
		#bbox = self.tensors[2][index]

		# transpose the image such that its channel dimension becomes
		# the leading one
		#image = image.permute(2, 0, 1)
		# check to see if we have any image transformations to apply
		# and if so, apply them
		
		# return a tuple of the images, labels, and bounding
		# box coordinates
		image, target = generate_target(index, 'mias_info/labels.txt')

		if self.transforms is not None:
		
			image = self.transforms(image)

		return image, target

	def __len__(self):
		# return the size of the dataset
		return len(self.imgs)	