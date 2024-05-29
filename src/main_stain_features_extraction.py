import sys
import os
import stain_process_data as prd
import torch
from stain_model import Unet2d
import numpy as np
from torchvision import transforms, models
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})

import PIL
import math
from transforms import *
from skimage import exposure

"""
	Function: transfer_style
	Transfer style to the tiles of a patient.
	
	Parameters:
		- stmodel: The pre-train model for transferring style.
		- patient_dataset: The tiles dataset.
		- batch_size: The number of tiles at one processing time.

	Returns:
		The transferred tiles.
"""
def transfer_style(stmodel, patient_dataset, batch_size=1):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    stmodel = stmodel.to(device)
    num_images = len(patient_dataset)
    num_batches = (num_images // batch_size) + 1
    all_fakes = []

    for i in range(num_batches):
        start_ind = i*batch_size
        end_ind = min((i+1)*batch_size, num_images)
        
        data = [ patient_dataset[idx] for idx in range(start_ind, end_ind) ]
        imgs = [ t for t, _ in data]
        conditions = [ c for _, c in data]
        if len(imgs) > 0:
            # If we would like transform image before prediction
            imgs = torch.stack(imgs, 0)
            
            imgs = imgs.to(device)
            fakes = stmodel(imgs)
            if not stmodel.final_activation:
                fakes=torch.sigmoid(fakes)
            
            f2 = []
            for fidx in range(fakes.shape[0]):
            	f = fakes[fidx]
            	f = (((f.detach().cpu().permute(1,2,0).numpy())/2 - 0.5)*255).astype('uint8')
            	f2.append(f)
            
            f2 = np.array(f2)
            all_fakes.append(f2)
    return all_fakes

# def display_progress(cond, real, fake, path, figsize=(10,5)):
#     cond = cond.detach().cpu().permute(1, 2, 0).numpy()   
#     real = real.detach().cpu().permute(1, 2, 0).numpy()
#     fake = fake.detach().cpu().permute(1, 2, 0).numpy()

#     images = [cond, real, fake]
#     titles = ['input','real','generated']
    
#     fig, ax = plt.subplots(1, 3, figsize=figsize)
#     for idx,img in enumerate(images):
#         img = img/2 + 0.5
#         ax[idx].imshow((img * 255).astype('uint8'))
#         ax[idx].axis("off")
#     for idx, title in enumerate(titles):
#         ax[idx].set_title('{}'.format(title))
#     plt.savefig(path)

"""
	Function: load_model
	Load the stain transfer model.
	
	Parameters:
		- chk_path: The path to checkpoint file.

	Returns:
		The pre-trained model for transferring the stain.
"""
def load_model(chk_path):
    gen_model= Unet2d(3, 5, 
                        n_classes=3, 
                        n_base_filters=32, 
                        final_activation=True)
    print("Loading model from {}".format(chk_path))
    checkpoint = torch.load(chk_path)
    # Remove discrimator
    for key in list(checkpoint['state_dict'].keys()):
        if 'disc' in key:
            checkpoint['state_dict'].pop(key)
    #print(checkpoint['state_dict'].keys())
    for key in list(checkpoint['state_dict'].keys()):
        new_key = key[4:]
        checkpoint['state_dict'][new_key] = checkpoint['state_dict'].pop(key)
        
    gen_model.load_state_dict(checkpoint['state_dict'])
    gen_model.eval()
    return gen_model

"""
	Function: resnet50_extractorpredict
	Extract the features from tiles by using pre-trained ResNet50.
	
	Parameters:
		- tiles_array: The tiles in numpy array.
		- batch_size: The number of tiles at one processing time.

	Returns:
		The extracted features for all tiles.
"""
def resnet50_extractorpredict(tiles_array, batch_size = 4):
	#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	#print(device)
	rsmodel = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
	# Select to avg pool layer
	rsmodel = torch.nn.Sequential(*(list(rsmodel.children())[:-1]))
	rsmodel.eval()
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
			input_tensor = input_tensor.to('cuda:1')
			rsmodel.to('cuda:1')
		with torch.no_grad():
			output = rsmodel(input_tensor)
		#output = output.squeeze()
		output = output.squeeze(3).squeeze(2)
		output_all.append(output.cpu().numpy())
	output_all = np.concatenate(output_all, axis = 0)
	return output_all

"""
	Function: extract_features
	Transfer style and extract the features from the tiles for all images.
	
	Parameters:
		- tiles_image_list: The tiles in numpy array.
		- stain_model: The pre-trained model for stain transfer.
		- save_dir: The path to the folder for saving the extracted features.

	Returns:
		The extracted features are saved to the folder.
"""
def extract_features(tiles_image_list, stain_model, save_dir = SAVE_DIR):
	for image_idx in tiles_image_list:
		if '.npz' not in image_idx:
			continue
		#if 'ST 12' not in image_idx:
		#	continue
		print(image_idx)																										
		tile_np = np.load(os.path.join(TILE_DIR,image_idx))['arr_0']
		print(tile_np.shape, np.isnan(tile_np).any())

		# Transfer stain
		print("Appling stain transfer.....")
		tile_np = transfer_images(tile_np, stain_model)
		print('Image after transferring the stain: ', tile_np.shape)
		
		# Extract the model
		tiles_features = resnet50_extractorpredict2(tile_np, batch_size = BATCH_SIZE)
		print(tiles_features.shape, np.isnan(tiles_features).any())

		# Save tiles features
		print("Save to file ....")
		save_path = os.path.join(save_dir, image_idx.replace('npy', 'npz'))
		np.savez_compressed(save_path, tiles_features)

"""
	Function: transfer_images
	Stain transfer by using a pre-trained model.
	
	Parameters:
		- tiles_array: The tiles in numpy array.
		- model: The pre-trained model for stain transfer.

	Returns:
		The tiles images after transferring the style.
"""
def transfer_images(tiles_array, model):
	test_images = [] # list of images
	for img in tiles_array:
		test_images.append(Image.fromarray(img.astype(np.uint8)).convert('RGB'))
	test_transf = A.Compose([A.CenterCrop(height = 224, width = 224),
                                A.Normalize(mean = [0.5, 0.5, 0.5], std =[0.5, 0.5, 0.5], max_pixel_value = 255.0,),
                                ToTensorV2(),],
                                additional_targets = {'image0': 'image'})
	patient_dataset = prd.Tile_Dataset(test_images, test_transf)
	all_fakes = transfer_style(model, patient_dataset, batch_size = 16)
	all_fakes = np.concatenate(all_fakes, axis  = 0)
	return all_fakes

if __name__ == '__main__':
	BATCH_SIZE = 256

	# BREAST CANCER
	SAVE_DIR = '/media/monc/LaCie10TB/BREAST_CANCER/TILES/Cas_Bergonie/HE_Bergonie/10X/tiles_features_stain'
	TILE_DIR = '/media/monc/LaCie10TB/BREAST_CANCER/TILES/Cas_Bergonie/HE_Bergonie/10X/tiles'

	stain_root = "/media/monc/Disk2/Models/StainTransfer/lightning_logs_Pix2Pix/default"
	# HES Bergonie
	#strain_version = 'version_3'
	#stain_ckpt_version = strain_version + '/checkpoints/epoch=56-step=106874.ckpt'

	# HE Sarcoma
	strain_version = 'version_2'
	stain_ckpt_version = strain_version + '/checkpoints/epoch=2-step=5624.ckpt'
	stain_chk_path = os.path.join(stain_root, stain_ckpt_version)
	dir_tile = TILE_DIR
	list_files = sorted(list(os.listdir(dir_tile)))
	try:
		os.makedirs(SAVE_DIR)
		#os.makedirs(SAVE_DIR_2)
	except OSError:
		pass
	# Load stain model
	stain_model =load_model(stain_chk_path)

	# process data
	list_npz = [f for f in list_files if '.npz' in f]
	processed = sorted(list(os.listdir(SAVE_DIR)))
	print("Processed images: ",len(processed))
	list_images = [image for image in list_npz if image not in processed]
	print("It remains {} images".format(len(list_images)))

	tiles_count = extract_features(list_images, stain_model)

	print("Finish")
	
	
	


