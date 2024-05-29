import sys
import os
import numpy as np
import PIL
from PIL import Image
from tqdm import tqdm
import math
import torch
from torchvision import transforms, models
from transforms import *
from skimage import exposure
import matplotlib.pyplot as plt

"""
	Function: efficientnet_b7_extractor
	Extract the features from extracted tiles by using pre-trained EfficinetNet B7 as an extractor.
	
	Parameters:
		- model: The pre-trained model.
		- tiles_array: The tiles in numpy array.
		- batch_size: The batch size (number of tiles) to process at one time.

	Returns:
		- output_all: The list of extracted features of all tiles.
"""
def efficientnet_b7_extractor(model, tiles_array, batch_size = 4):
	preprocess = transforms.Compose([RandomNormalize(prob = 1.0),
									ToTensor()])
	n_tiles = tiles_array.shape[0]
	output_all = []
	for idx in range(0, n_tiles, batch_size):
		if idx + batch_size > n_tiles:
			batch_size = n_tiles - idx
		input_arrays = tiles_array[idx:idx + batch_size]
		input_tensor = torch.stack([preprocess(input_arr) for input_arr in input_arrays])
		nonNan = []
		for t in input_tensor:
			if not (np.isnan(t.numpy()).any()):
				nonNan.append(t)
		if len(nonNan) > 0:
			input_tensor = torch.stack(nonNan)
			if torch.cuda.is_available():
				input_tensor = input_tensor.to('cuda:1')
				model.to('cuda:1')
			with torch.no_grad():
				output = model(input_tensor)
			#output = output.squeeze()
			output = output.squeeze(3).squeeze(2)
			output_all.append(output.cpu().numpy())

	output_all = np.concatenate(output_all, axis = 0)
	return output_all

"""
	Function: resnet50_extractorpredict
	Extract the features from extracted tiles by using pre-trained ResNet50 as an extractor.
	
	Parameters:
		- tiles_array: The tiles in numpy array.
		- batch_size: The batch size (number of tiles) to process at one time.

	Returns:
		- output_all: The list of extracted features of all tiles.
"""
def resnet50_extractorpredict(tiles_array, batch_size = 4):
	
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(device)
	model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
	#model = models.resnet152(pretrained = True)
	model = torch.nn.Sequential(*(list(model.children())[:-1]))
	model.eval()
	preprocess = transforms.Compose([RandomNormalize(prob = 1.0),
									ToTensor()])

	n_tiles = tiles_array.shape[0]
	output_all = []
	for idx in range(0, n_tiles, batch_size):
		if idx + batch_size > n_tiles:
			batch_size = n_tiles - idx
		input_arrays = tiles_array[idx:idx + batch_size]

		input_tensor = torch.stack([preprocess(input_arr) for input_arr in input_arrays])
		if torch.cuda.is_available():
			input_tensor = input_tensor.to('cuda:0')
			model.to('cuda:0')
		with torch.no_grad():
			output = model(input_tensor)
		#output = output.squeeze()
		output = output.squeeze(3).squeeze(2)
		output_all.append(output.cpu().numpy())
	output_all = np.concatenate(output_all, axis = 0)
	return output_all

"""
	Function: resNext_extractorpredict
	Extract the features from extracted tiles by using pre-trained resNext101 as an extractor.
	
	Parameters:
		- tiles_array: The tiles in numpy array.
		- batch_size: The batch size (number of tiles) to process at one time.

	Returns:
		- output_all: The list of extracted features of all tiles.
"""
def resNext_extractorpredict(tiles_array, batch_size = 4):
	# Get the reference array
	
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(device)
	model = models.resnext101_32x8d(pretrained = True)
	# Select to avg pool layer
	model = torch.nn.Sequential(*(list(model.children())[:-1]))
	model.eval()
	preprocess = transforms.Compose([RandomNormalize(prob = 1.0),
									ToTensor()])

	n_tiles = tiles_array.shape[0]
	output_all = []
	for idx in range(0, n_tiles, batch_size):
		if idx + batch_size > n_tiles:
			batch_size = n_tiles - idx
		input_arrays = tiles_array[idx:idx + batch_size]

		input_tensor = torch.stack([preprocess(input_arr) for input_arr in input_arrays])
		if torch.cuda.is_available():
			input_tensor = input_tensor.to('cuda:0')
			model.to('cuda:0')
		with torch.no_grad():
			output = model(input_tensor)
		#output = output.squeeze()
		output = output.squeeze(3).squeeze(2)
		output_all.append(output.cpu().numpy())
	output_all = np.concatenate(output_all, axis = 0)
	print(output_all.shape)
	#output_all = output_all.reshape(output_all.shape[0] * output_all.shape[1], output_all.shape[2])
	return output_all

"""
	Function: extract_features
	Extracting the features from the extracted tiles for all patients. The extracted features are save to npz file.

	Parameters:
		- tiles_image_list: List of extracted tiles of all patients.
		- save_folder: The path to folder to save the extracted features.

	
"""
def extract_features(tiles_image_list, save_folder):
	for image_idx in tiles_image_list:
		if '.npz' not in image_idx:
			continue
		print(image_idx)																										
		tile_np = np.load(os.path.join(TILE_DIR,image_idx))['arr_0']
		print(tile_np.shape, np.isnan(tile_np).any())

		# Extract the features by using the pre-trained model
		tiles_features = resnet50_extractorpredict(tile_np, batch_size = BATCH_SIZE)
		#tiles_features = efficientnet_b7_extractor(model, tile_np, batch_size = BATCH_SIZE)
		#tiles_features = resNext_extractorpredict(tile_np, batch_size = BATCH_SIZE)
		print(tiles_features.shape, np.isnan(tiles_features).any())

		# Save tiles features to npz file
		print("Save to file ....")
		save_path = os.path.join(save_folder, image_idx.replace('npy', 'npz'))
		np.savez_compressed(save_path, tiles_features)


if __name__ == '__main__':
	BATCH_SIZE = 512

	# IGR-Perisarc
	save_dir = '/media/monc/SeagateHub/IGR-2-Perisarc/IGR_tiles_40X/R1_2nd/tiles_features_tmp'
	dir_tile = '/media/monc/SeagateHub/IGR-2-Perisarc/IGR_tiles_40X/R1_2nd/tiles_2'

	list_files = sorted(list(os.listdir(dir_tile)))
	try:
		os.makedirs(save_dir)
	except OSError:
		pass
	
	list_npz = [f for f in list_files if '.npz' in f]
	processed = sorted(list(os.listdir(save_dir)))
	print("Processed images: ",len(processed))
	#list_images = [image for image in list_npz if image not in processed]
	list_images = []
	for img in list_npz:
		in_list = False
		for p in processed:
			if p.replace('.npz','') in img:
				in_list = True
				break
		if not in_list:
			list_images.append(img)
	print("It remains {} images".format(len(list_images)))

	tiles_count = extract_features(list_images, save_dir)
	print("Finish")
	

	
	


