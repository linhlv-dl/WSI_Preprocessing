import sys
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw
import openslide as osl
import openslide.deepzoom as deepzoom
import itertools
from tqdm import tqdm
import xml.etree.ElementTree as ET
import math
import shutil


# Function: read_XML_2
# Read the XML file and get the coordinates of the points to limit the selected region.
#
# Parameters:
#	xml_file: The file path to XML file.
#
# Returns:
# A dictionary contains the list of points (of regions) and color.
def read_XML_2(xml_file):
	tree = ET.parse(xml_file)
	root = tree.getroot()
	x_list = []
	y_list = []
	dict_annos = {}
	for anno in root:
		print(anno.tag, anno.attrib['id'])
		annokey = anno.tag + anno.attrib['id']
		#color = anno.attrib['LineColor']
		list_regions = []
		for region in anno.iter('annotation'):
			color = region.attrib['color']
			region_idx_vertex = []
			for pointlist in region.iter('pointlist'):
				for point in pointlist:
					x = point[0].text
					y = point[1].text
					region_idx_vertex.append((int(x), int(y)))
			list_regions.append(region_idx_vertex)
			print(region.tag, color, len(region_idx_vertex))
		#print(len(list_regions))
		dict_annos[annokey] = (color, list_regions)
	return dict_annos

# Function: read_slide
# Read the Slide and get information for converting the coordinates.
#
# Parameters:
# 	npdi_file: The NDPI file (Slide).
#
# Returns:
# The information of slide: Offset coordinate (x,y), width and height of slide.
def read_slide(npdi_file):
	slide = osl.OpenSlide(npdi_file)
	x_offset = slide.properties['hamamatsu.XOffsetFromSlideCentre']
	y_offset = slide.properties['hamamatsu.YOffsetFromSlideCentre']
	mppx = float(slide.properties['openslide.mpp-x']) * 1000
	mppy = float(slide.properties['openslide.mpp-y']) * 1000
	w = slide.properties['openslide.level[0].width']
	h = slide.properties['openslide.level[0].height']
	return int(x_offset), int(y_offset), float(mppx), float(mppy), int(w), int(h)

# Function: convert_coordinates
# Convert the coordinates of a point (in the selected region) from nanometre to pixel.
#
# Parameters:
# 	- x, y: The coordinates of a point in nanometre.
# 	- x_offset, y_offset: The x, y offsets of the coordinates to the center of Slide.
# 	- mppx, mppy: The mppx, mppy.
# 	- w, h: The width and height of the Slide.
#
# Returns:
# The coordinates of a point in pixel.
def convert_coordinates(x, y, x_offset, y_offset, mppx, mppy, w, h):
	# Convert x coordinate
	x1 = x - x_offset
	x2 = x1 / mppx
	x3 = w //2 - x2
	x4 = w - x3
	# Convert y coordinate
	y1 = y - y_offset
	y2 = y1 / mppy
	y3 = h //2 - y2
	y4 = h - y3
	return x4, y4

"""
	Function: convert_to_pixels
	Convert the coordinates of the points in the selected region from nanometre to pixel.
	
	Parameters:
		- xml_file: The XML file contains coordinates of points in selected region in nanometre.
		- npdi_file: The Slide in NDPI file.
	
	Returns:
		The dictionary of points (in selected region) in pixel.
"""
def convert_to_pixels(xml_file, npdi_file):
	x_offset, y_offset, mppx, mppy, w, h = read_slide(npdi_file)
	dict_annos = read_XML_2(xml_file)
	convert_dict = {}
	for dict_idx in dict_annos.keys():
		color, list_regions = dict_annos[dict_idx]
		region_idx_verticles = list_regions[0]
		print(len(region_idx_verticles))
		list_vertex = []
		for (x,y) in region_idx_verticles:
			new_x, new_y = convert_coordinates(x,y, x_offset, y_offset, mppx, mppy, w, h)
			list_vertex.append((new_x, new_y))
		convert_dict[dict_idx] = (color, list_vertex)
	return convert_dict

"""
	Function: convert_color
	Convert the color from HEX to DEC code.
	
	Parameters:
		hex_color: The color in HEX.

	Returns:
		The color in DEC.
"""
def convert_color(hex_color):
	if hex_color == '#000000':
		return 0
	if hex_color == '#ff0000':
		return 255
	if hex_color == '#0000ff':
		return 16711680
	if hex_color == '#00ff00':
		return 65280
	if hex_color == '#00ffff':
		return 65535

"""
	Function: write_xml
	Export to XML file.

	Parameters:
		- converted_dict: The dictionary of points (selected region) in pixel.
		- xml_save: The file path to save the exported XML file.

	Returns:
		
"""
def write_xml(converted_dict, xml_save):
	root = ET.Element('Annotations')
	idx = 1
	for anno in converted_dict.keys():
		color, list_vertex = converted_dict[anno]
		print(color)
		color2 = convert_color(color)

		anno_idx = ET.SubElement(root, 'Annotation')
		anno_idx.set('Id', str(idx))
		idx += 1
		anno_idx.set('LineColor', str(color2))
		regions = ET.SubElement(anno_idx, 'Regions')
		region = ET.SubElement(regions, 'Region')
		region.set('Id', str(1))
		attribs = ET.SubElement(region, 'Attributes')
		verticles = ET.SubElement(region, 'Vertices')
		for (x,y) in list_vertex:
			vertex = ET.SubElement(verticles, 'Vertex')
			vertex.set('X', str(float(x)))
			vertex.set('Y', str(float(y)))
		plots = ET.SubElement(anno_idx,'Plots')
		
	tree = ET.ElementTree(root)
	tree.write(xml_save)
	return

"""
	Function: convert_all_images
	Convert the coordinates of point in selected regions of all Slides to pixel.
	
	Parameters:
		- image_folder: The file path to Slide folder.
		- ndpa_folder: The file path to xml file (ndpi.ndpa - in nanometre).
		- save_dir: The file path to folder for saving the exported XML files (in pixels).
	
	Returns:
"""
def convert_all_images(image_folder, ndpa_folder, save_dir):
	list_files = sorted(list(os.listdir(ndpa_folder)))
	print(list_files)
	list_ndpa = [fname for fname in list_files if '.ndpa' in fname]
	
	list_processed = sorted(list(os.listdir(save_dir)))
	#list_processed = [fname for fname in list_processed if '.ndpa' in fname]

	for fndpa in list_ndpa:
		if fndpa.replace('.ndpi.ndpa','.xml') in list_processed:
			continue
		if '272922' in fndpa or '306710-C-D' in fndpa:# or 'IS225' in fndpa or 'JC872' in fndpa or 'KN596' in fndpa or 'LD829' in fndpa:
			continue
		ndpa_path = os.path.join(ndpa_folder, fndpa)
		wsi_path = os.path.join(image_folder, fndpa.replace('.ndpi.ndpa','.ndpi'))
		xml_path = os.path.join(save_dir, fndpa.replace('.ndpi.ndpa','.xml'))
		#print(ndpa_path)
		print(xml_path)
		print(wsi_path)
		if not os.path.exists(wsi_path):
			print("Don't find the file: ", wsi_path)
			continue
		converted_dict = convert_to_pixels(ndpa_path, wsi_path)
		write_xml(converted_dict, xml_path)
	return

"""
	Function: __main__
	Execute the functions to convert the coordinates of selected regions from nanometre to pixel.
	
	Parameters:
		- image_folder: The file path to Slide folder
		- ndpa_folder: The file path to xml file (ndpi.ndpa - in nanometre)
		- save_xml_dir: The file path to folder for saving the exported XML files (in pixels)
	
"""

if __name__ == '__main__':
	# Extract the region from the WSI of all images
	image_folder = '/media/monc/LaCie10TB/BREAST_CANCER/SLIDES/Cas_exterieurs/Cas_extern2611/HES_EXT'
	ndpa_folder = '/media/monc/LaCie10TB/BREAST_CANCER/SLIDES/Cas_exterieurs/Cas_extern2611/HES_EXT'
	save_xml_dir = '/media/monc/LaCie10TB/BREAST_CANCER/XML/HES_exterieurs_EXT'

	try:
		os.makedirs(save_xml_dir)
	except OSError:
		pass

	convert_all_images(image_folder, ndpa_folder, save_xml_dir)
	print("Finish !")

