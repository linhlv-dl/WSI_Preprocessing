import sys
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
import openslide as osl
import openslide.deepzoom as deepzoom
from timeit import default_timer as timer  
import itertools
from tqdm import tqdm
import multiprocessing as mlp
import time
import preprocessing as pre
import math
import torch
import skimage.filters as sk_filters
import scipy.ndimage.morphology as sc_morph
import skimage.morphology as sk_morphology
import matplotlib.pyplot as plt

# Function: get_tile_coords
# Get the coordinates of all tiles in WSI at a level (magnification).
#
# Parameters:
#	- slide: The OpenSlide object created by reading the WSI file.
#	- level: The level (magnification) of Slide to extract the tiles. Default: 0 (40X).
#	- tile_size: The size of the tile to extract. Default: 224 x 224 pixels.
#
# Returns:
# List of coordinates and list of indexes of tiles extracted from WSI.
def get_tile_coords(slide, level = 0, tile_size = 224):
	slide_w, slide_h = slide.level_dimensions[level]
	return get_tile_coords_image(slide_w, slide_h, tile_size = tile_size)

"""
	Function: get_tile_coords_image
	Get the coordinates (and indexes) of tiles extracted from an image.
	
	Parameters:
		- image_w: The width of the image.
		- image_h: The height of the image.
		- tile_size: The size of tile. Default: 7 x 7 pixels.

	Returns:
		List of coordinates (and indexes) of tiles extracted from image.
"""
def get_tile_coords_image(image_w, image_h, tile_size = 7):
	print(image_w, image_h, tile_size)
	num_w = int(((image_w - tile_size)// tile_size) + 1)
	num_h = int(((image_h - tile_size)// tile_size) + 1)
	tile_coords = [(i * tile_size, j * tile_size) for i, j in itertools.product(range(num_w), range(num_h))]
	tile_indexs = np.array([(i , j) for i, j in itertools.product(range(num_w), range(num_h))])
	print("Total tiles: {}".format(len(tile_indexs)))
	return tile_coords, tile_indexs

"""
	Function: slide_to_png
	Read the Slide to indicate the size of image at the selected level and the size down-sampled image.
	
	Parameters:
		- slide: The OpenSlide object created by reading the WSI file.
		- selected_level: The selected level of WSI to extract the tiles.
		- scale: The down-sample scale to select the down-sampled image from selected level for pre-processing. Default: 32.
		- save_path: The file path to save the down-sampled image for verifying.

	Returns:
		- img: The down-sampled image.
		- selected_w: The width of image at selected level.
		- selected_h: The height of image at selected level.
		- new_w: The width of down-sampled image.
		- new_h: The height of down-sampled image.
		- level: The level of down-sampled image corresponding to the scale.
"""
def slide_to_png(slide, selected_level, scale = 32, save_path = None):
	w, h = slide.dimensions
	new_w = math.floor(w / scale)
	new_h = math.floor(h / scale)
	level = slide.get_best_level_for_downsample(scale)
	whole_slide_image = slide.read_region((0,0), level, slide.level_dimensions[level])
	whole_slide_image = whole_slide_image.convert("RGB")
	img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
	if save_path is not None:
		img.save(save_path)
	selected_w, selected_h = slide.level_dimensions[selected_level]
	print(level,selected_w, selected_h, new_w, new_h)
	return img, selected_w, selected_h, new_w, new_h, level

# def slide_to_png2(slide, scale = 32, save_path = None):
# 	w, h = slide.dimensions
# 	new_w = math.floor(w / scale)
# 	new_h = math.floor(h / scale)
# 	level = slide.get_best_level_for_downsample(scale)
# 	whole_slide_image = slide.read_region((0,0), level, slide.level_dimensions[level])
# 	whole_slide_image = whole_slide_image.convert("RGB")
# 	img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
# 	if save_path is not None:
# 		img.save(save_path)
# 	return img, w, h, new_w, new_h, level

"""
	Function: tissu_percent
	Compute the percentage of tissue in a tile.
	
	Parameters:
		- np_tile: The tile in numpy array.

	Returns:
		The percentage of tissue region in the tile.
"""
def tissu_percent(np_tile):
	if (len(np_tile.shape) == 3) and (np_tile.shape[2] == 3):
		np_sum = np_tile[:,:,0] + np_tile[:,:,1] + np_tile[:,:,2]
		mask_percent = 100 - np.count_nonzero(np_sum)/ np_sum.size * 100
	else:
		mask_percent = 100 - np.count_nonzero(np_tile)/ np_tile.size * 100
	return 100 - mask_percent

"""
	Function: extract_tile
	Extract the tile from WSI.
	
	Parameters:
		- level: The level of WSI where extracting the tile.
		- coordinates: The coordinates of tile that we would like to extract from WSI.
		- zoom: The zoom object to focus on the level of Slide and extract the tile.

	Returns:
		The extracted tile.
"""
def extract_tile(level, coordinates, zoom):
	tile = np.array(zoom.get_tile(level, (coordinates[0], coordinates[1])))
	return tile

"""
	Function: extract_mask_tile
	Extract the mask of a tile from WSI mask and give a label for tile.
	
	Parameters:
		- mask_array: The WSI mask
		- indices: The coordinates of tile to extract the mask.
		- tile_size: The size of tile. Default: 7 x 7 pixels.

	Returns:
		- patch: The tile mask.
		- label: The label of the extracted mask (depending on the percentage of tissue).
"""
def extract_mask_tile(mask_array, indices, tile_size = 7):
	patch = mask_array[indices[1]:indices[1]+tile_size, indices[0]:indices[0]+tile_size]
	p_tissu = tissu_percent(patch)
	label = ''
	if p_tissu >= 50:
		label = 'A'
	else:
		label = 'B'
	return patch, label

"""
	Function: tile_border_color
	Define the border color for a tile depending on its label.
	
	Parameters:
		tile_label: The label of the tile

	Returns:
		The border color for the tile.
"""
def tile_border_color(tile_label):
	if tile_label == 'A':
		border_color = (0, 255, 0)
	elif tile_label == 'B':
		border_color = (255, 0, 0)

	return border_color

"""
	Function: tile_border
	Draw the border of a tile (rectangle).
	
	Parameters:
		- draw: The drawing object
		- c_s, r_s: The coordinates of top left corner.
		- c_e, r_e: The coordinates of bottom right corner.
		- color: The color to draw the rectangle.
		- border_size: The size to draw the border line. Default: 2.

	Returns:
"""
def tile_border(draw, r_s, r_e, c_s, c_e, color, border_size=2):
	for x in range(0, border_size):
		draw.rectangle([(c_s + x, r_s + x), (c_e - 1 - x, r_e - 1 - x)], outline=color)

"""
	Function: draw_tiles
	Draw the border for all tiles.
	
	Parameters:
		- draw: The drawing object.
		- tiles_label: The labels of all tiles.
		- tiles_indices: The coordinates of all tiles.
		- tile_size: The size of tile. Default: 7 x 7 pixels.

	Returns:
"""
def draw_tiles(draw, tiles_label, tiles_indices, tile_size = 7):
	for idx in range(len(tiles_indices)):
		tile_label = tiles_label[idx]
		tile_index = tiles_indices[idx]
		color = tile_border_color(tile_label)
		c_s, r_s = tile_index
		c_e, r_e = c_s + tile_size, r_s + tile_size
		tile_border(draw,r_s, r_e, c_s, c_e, color)

"""
	Function: preprocessing_image
	Preprocessing a WSI.
	
	Parameters:
		- slide: The OpenSlide object created by reading the WSI file.
		- selected_level: The selected level to extract the tiles.
		- otsu: The Otsu threshold to binary the down-sampled image.
		- scale: The down-sample scale to select the down-sampled image from selected level for pre-processing. Default: 32.
		- save_path: The file path to save the pre-processed image

	Returns:
		- bin_image: The binary image of down-sampled image.
		- rgb_image: The RGB image of down-sampled image.
		- selected_w, selected_h: The width, height of image at selected level.
		- new_w, new_h: The width, height of down-sampled image.
		- level: The level of down-sampled image corresponding to the scale.
"""
def preprocessing_image(slide, selected_level, otsu = 'global', scale = 32, save_path = None):
	rgb_image, selected_w, selected_h, new_w, new_h, level = slide_to_png(slide, selected_level, scale = scale, save_path = None)
	# Process: RGB -> HSV -> S -> Otsu -> Binary Closing -> Remove objects > 3000
	np_img = np.asarray(rgb_image)
	hsv_image = pre.filter_rgb_to_hsv(np_img)
	s_image = pre.filter_hsv_to_s(hsv_image)
	
	# Local otsu threshold
	if otsu == 'local':
		local_otsu = sk_filters.rank.otsu(s_image, sk_morphology.disk(3))
		local_otsu = local_otsu.astype(np.uint8) * 255
		local_otsu = local_otsu > 0
	elif otsu == 'global':
		# global otsu threshold 
		local_otsu = pre.filter_otsu_threshold(s_image)

	result = sc_morph.binary_closing(local_otsu, sk_morphology.disk(3), iterations = 1)
	bin_image = pre.filter_remove_small_objects(result, 1000)
	if save_path is not None:
		Image.fromarray(bin_image).save(save_path)
	return bin_image, rgb_image, selected_w, selected_h, new_w, new_h, level
'''
def preprocessing_image(slide, scale = 32, save_path = None):
	# Process: PNG -> GrayScale -> Otsu -> Binary Closing -> Remove objects > 5000
	# 
	rgb_image, w, h, new_w, new_h, level = slide_to_png(slide, scale = scale, save_path = None)
	#rgb_image, w, h, new_w, new_h, level = slide_to_png2(slide, scale = scale, save_path = save_path)
	rgb = np.asarray(rgb_image)
	# Filters
	grayscale = pre.filter_rgb_to_grayscale(rgb)
	component = pre.filter_complement(grayscale)
	result = pre.filter_otsu_threshold(component) # as numpy array
	# Erode and dilate
	e_result = pre.filter_binary_closing(result, 3)
	e_result = pre.filter_remove_small_objects(e_result) # remove objects which have the size < 5000
	if save_path is not None:
		Image.fromarray(e_result).save(save_path)
	return e_result, rgb_image, w, h, new_w, new_h, level
'''

"""
	Function: select_tiles
	Filter the tiles from all tiles
	
	Parameters:
		- slide: The OpenSlide object created by reading the WSI file.
		- selected_level: The selected level to extract the tiles.
		- org_tile_size: The size of tile to extract at selected level. 
		- scale: The down-sample scale to select the down-sampled image from selected level for pre-processing. Default: 32.
		- otsu: The Otsu threshold to binary the down-sampled image. Default: global (Otsu).
		- save_path: The file path to save the pre-processed image

	Returns:
		- selected_tiles: The list of extracted tiles at selected level.
		- scaled_tiles: The list of extracted tiles at down-sampled level.
"""
def select_tiles(slide, selected_level, org_tile_size = 224, scale = 32, otsu = 'global', save_path = None):
	bin_array, rgb_image, selected_w, selected_h, new_w, new_h, _ = preprocessing_image(slide, 
																						selected_level, 
																						otsu = otsu,
																						scale = scale, 
																						save_path = save_path)

	org_scale = 2**selected_level
	scaled_tile_size = (org_tile_size // scale) * org_scale
	print(scaled_tile_size)
	tiles_coords, tiles_indices = get_tile_coords_image(new_w, new_h, tile_size = scaled_tile_size)
	
	mask_tiles = [extract_mask_tile(bin_array, coord, tile_size = scaled_tile_size) for coord in tiles_coords]
	labels = [m[1] for m in mask_tiles]

	# Draw the tiles border
	draw = ImageDraw.Draw(rgb_image)
	draw_tiles(draw, labels, tiles_coords, tile_size = scaled_tile_size)

	rgb_image.save(SAVE_DIR + '/tmp/' + save_path.split('/')[-1])

	_, org_tiles_indices = get_tile_coords_image(selected_w, selected_h, tile_size = org_tile_size)
	print(len(org_tiles_indices), len(tiles_indices))
	assert(len(org_tiles_indices) == len(tiles_indices))
	selected_tiles = []
	scaled_tiles = []
	for idx in range(len(labels)):
		label = labels[idx]
		if label == 'A':
			selected_tiles.append(org_tiles_indices[idx])
			scaled_tiles.append(tiles_coords[idx])
	print("Selected tiles: ", len(selected_tiles), len(scaled_tiles))
	return selected_tiles, scaled_tiles

"""
	Function: process_image
	Process the image and extract the tile from WSI.
	
	Parameters:
		- slide_name: The name of the WSI
		- selected_level: The selected level to extract the tiles.

	Returns:
		- slide_name: The name of the WSI.
		- selected_tiles: The coordinates of extracted tiles.
		- tiles: The extracted tiles.
"""
def process_image(slide_name, selected_level = 0):
	slide_path = os.path.join(ROOT_DIR, slide_name)
	slide = osl.OpenSlide(slide_path)

	zoom = deepzoom.DeepZoomGenerator(slide, tile_size = 224, overlap = 0)
	save_bin_path = SAVE_DIR + '/tmp/' + slide_name.replace('.ndpi', '.png')
	selected_tiles, scaled_tiles = select_tiles(slide, 
												selected_level, 
												org_tile_size = 224, 
												scale = 32, 
												otsu = 'global', 
												save_path = save_bin_path)
	
	# Extract and save the tiles
	clevel = zoom.level_count - selected_level - 1
	tiles = np.array([extract_tile(clevel, coord, zoom) for coord in tqdm(selected_tiles)])
	print("Number of selected (scaled) tiles and original tiles: ",len(selected_tiles), tiles.shape[0])
	save_tile = os.path.join(SAVE_DIR, 'tiles', slide_name.replace('.ndpi', '.npz'))
	np.savez_compressed(save_tile, tiles)
	#tiles = None
	print("Ok")

	# Save the coordinates of the selected tiles
	tiles_org_coordinates = np.array(selected_tiles)
	tiles_scaled_coordinates = np.array(scaled_tiles)
	print(tiles.shape, tiles_org_coordinates.shape, tiles_scaled_coordinates.shape)

	save_org_tiles = os.path.join(SAVE_DIR, 'coords_org', os.path.basename(slide_name).replace('.ndpi', '.npz'))
	np.savez_compressed(save_org_tiles, tiles_org_coordinates)
	save_scaled_tiles = os.path.join(SAVE_DIR, 'coords_scaled', os.path.basename(slide_name).replace('.ndpi', '.npz'))
	np.savez_compressed(save_scaled_tiles, tiles_scaled_coordinates)
	
	return (slide_name, selected_tiles, tiles)

"""
	Function: process_range_images
	Convert the color from HEX to DEC code.
	
	Parameters:
		hex_color: The color in HEX.

	Returns:
		The color in DEC.
"""
# def process_range_images(sublist):
# 	list_tiles = []
# 	for slide in sublist:
# 		list_tiles.append(process_image(slide, selected_level = LEVEL))
# 	return list_tiles

if __name__ == '__main__':
	#
	# LEVEL = 	0		1		2		3		4		5		6		7		8
	# MAGNI. =	40X		20X		10X 	5X
	# scale = 	1		2		4		8		16		32	
	#
	ROOT_DIR = 'path to ndpi folder'
	LEVEL = 0
	TILE_SIZE = 224
	SAVE_DIR = 'path to the export folder'
	'''
		The SAVE_DIR have contains some sub-folders:
		- tiles: to contain extracted tiles
		- tmp: to contain the png file for verifying the binary and tiling.
		- coords_org: to contain the coordinates of tiles at LEVEL magnification.
		- coords_scaled: to contain the coordinates of tiles in down-sample level.
	'''
	
	dir_image = ROOT_DIR
	list_files = sorted(list(os.listdir(dir_image)))
	list_images = [image for image in list_files if image.split('.')[-1] == 'ndpi']
	try:																															
		os.makedirs(SAVE_DIR)
	except OSError:
		pass
	list_extracted = list(os.listdir(SAVE_DIR + '/tiles'))
	list_images = [image for image in list_images if image.replace('.ndpi', '.npz') not in list_extracted]

	print('It remains {} images'.format(len(list_images)))
	
	# With one process
	for image in list_images:
		print(image)
		
		slide_path = os.path.join(ROOT_DIR, image)
		slide = osl.OpenSlide(slide_path)
		print(image, slide.dimensions, slide.level_count)
		#print(slide.properties['openslide.objective-power'])
		for idx in range(slide.level_count):
			print(idx,slide.level_dimensions[idx], slide.level_downsamples[idx])
		# Extract the tiles at 20X
		selected_level = 0 # if maximum is 20X
		if int(slide.properties['openslide.objective-power']) == 40:
			print("Change level to ", slide.properties['openslide.objective-power'])
			selected_level = 1
		process_image(image, selected_level = selected_level)
		

	print("Finish.")
	
	


