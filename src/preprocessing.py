import sys
import os
import numpy as np
import PIL
from PIL import Image
from skimage import io
import openslide as osl
import math
import scipy.ndimage.morphology as sc_morph
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation

"""
  Function: filter_rgb_to_grayscale
  Convert an RGB NumPy array to a grayscale NumPy array. Shape (h, w, c) to (h, w).

  Parameters:
    - np_img: RGB Image as a NumPy array.
    - output_type: Type of array to return (float or uint8)

  Returns:
    - grayscale: Grayscale image as NumPy array with shape (h, w).
"""
def filter_rgb_to_grayscale(np_img, output_type="uint8"):
  # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
  grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
  if output_type != "float":
    grayscale = grayscale.astype("uint8")
  #util.np_info(grayscale, "Gray", t.elapsed())
  return grayscale

"""
  Function: filter_complement
  Obtain the complement of an image as a NumPy array.
  
  Parameters:
    - np_img: Image as a NumPy array.
    - output_type: Type of array to return (float or uint8).

  Returns:
    Complement image as Numpy array.
"""
def filter_complement(np_img, output_type="uint8"):
  if output_type == "float":
    complement = 1.0 - np_img
  else:
    complement = 255 - np_img
  #util.np_info(complement, "Complement", t.elapsed())
  return complement

"""
  Function: filter_hysteresis_threshold
  Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
  
  Parameters:
    - np_img: Image as a NumPy array.
    - low: Low threshold.
    - high: High threshold.
    -output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
"""
def filter_hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
  hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
  if output_type == "bool":
    pass
  elif output_type == "float":
    hyst = hyst.astype(float)
  else:
    hyst = (255 * hyst).astype("uint8")
  #util.np_info(hyst, "Hysteresis Threshold", t.elapsed())
  return hyst

"""
  Function: filter_otsu_threshold
  Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
  
  Parameters:
    - np_img: Image as a NumPy array.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
  """
def filter_otsu_threshold(np_img, output_type="uint8"): 
  otsu_thresh_value = sk_filters.threshold_otsu(np_img)
  otsu = (np_img > otsu_thresh_value)
  if output_type == "bool":
    pass
  elif output_type == "float":
    otsu = otsu.astype(float)
  else:
    otsu = otsu.astype("uint8") * 255
  #util.np_info(otsu, "Otsu Threshold", t.elapsed())
  return otsu

"""
  Function: filter_local_otsu_threshold
  Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the
  local Otsu threshold.
  
  Parameters:
    - np_img: Image as a NumPy array.
    - disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
"""
def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8"): 
  local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
  if output_type == "bool":
    pass
  elif output_type == "float":
    local_otsu = local_otsu.astype(float)
  else:
    local_otsu = local_otsu.astype("uint8") * 255
  #util.np_info(local_otsu, "Otsu Local Threshold", t.elapsed())
  return local_otsu

"""
  Function: filter_entropy
  Filter image based on entropy (complexity).

  Parameters:
    - np_img: Image as a NumPy array.
    - neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
    - threshold: Threshold value.
    - output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
"""
def filter_entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"): 
  entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
  if output_type == "bool":
    pass
  elif output_type == "float":
    entr = entr.astype(float)
  else:
    entr = entr.astype("uint8") * 255
  #util.np_info(entr, "Entropy", t.elapsed())
  return entr

"""
  Function: filter_canny
  Filter image based on Canny algorithm edges.

  Parameters:
    - np_img: Image as a NumPy array.
    - sigma: Width (std dev) of Gaussian.
    - low_threshold: Low hysteresis threshold value.
    - high_threshold: High hysteresis threshold value.
    - output_type: Type of array to return (bool, float, or uint8).

  Returns:
    NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
"""
def filter_canny(np_img, sigma=1, low_threshold=0, high_threshold=25, output_type="uint8"): 
  can = sk_feature.canny(np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    can = can.astype(float)
  else:
    can = can.astype("uint8") * 255
  #util.np_info(can, "Canny Edges", t.elapsed())
  return can

"""
  Function: mask_percent
  Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).
  
  Parameters:
    - np_img: Image as a NumPy array.
  
  Returns:
    The percentage of the NumPy array that is masked.
"""
def mask_percent(np_img):
  if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
    np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
    mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
  else:
    mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
  return mask_percentage

"""
  Function: tissue_percent
  Determine the percentage of a NumPy array that is tissue (not masked).
  
  Parameters:
    - np_img: Image as a NumPy array.
  
  Returns:
    The percentage of the NumPy array that is tissue.
"""
def tissue_percent(np_img):
  return 100 - mask_percent(np_img)

"""
  Function: filter_remove_small_objects
  Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
  is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
  reduce the amount of masking that this filter performs.
  
  Parameters:
    - np_img: Image as a NumPy array of type bool.
    - min_size: Minimum size of small object to remove.
    - avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
    - overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8).
"""
def filter_remove_small_objects(np_img, min_size=3000, avoid_overmask=True, overmask_thresh=95, output_type="uint8"):
  rem_sm = np_img.astype(bool)  # make sure mask is boolean
  rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
  mask_percentage = mask_percent(rem_sm)
  if (mask_percentage >= overmask_thresh) and (min_size >= 1) and (avoid_overmask is True):
    new_min_size = min_size / 2
    print("Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d" % (
      mask_percentage, overmask_thresh, min_size, new_min_size))
    rem_sm = filter_remove_small_objects(np_img, new_min_size, avoid_overmask, overmask_thresh, output_type)
  np_img = rem_sm

  if output_type == "bool":
    pass
  elif output_type == "float":
    np_img = np_img.astype(float)
  else:
    np_img = np_img.astype("uint8") * 255

  #util.np_info(np_img, "Remove Small Objs", t.elapsed())
  return np_img

"""
  Function: filter_remove_small_holes
  Filter image to remove small holes less than a particular size.
  
  Parameters:
    - np_img: Image as a NumPy array of type bool.
    - min_size: Remove small holes below this size.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8).
"""
def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8"):
  rem_sm = sk_morphology.remove_small_holes(np_img, min_size=min_size)

  if output_type == "bool":
    pass
  elif output_type == "float":
    rem_sm = rem_sm.astype(float)
  else:
    rem_sm = rem_sm.astype("uint8") * 255

  #util.np_info(rem_sm, "Remove Small Holes", t.elapsed())
  return rem_sm

"""
  Function: filter_contrast_stretch
  Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in
  a specified range.
  
  Parameters:
    - np_img: Image as a NumPy array (gray or RGB).
    - low: Range low value (0 to 255).
    - high: Range high value (0 to 255).
  
  Returns:
    Image as NumPy array with contrast enhanced.
"""
def filter_contrast_stretch(np_img, low=40, high=60): 
  low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
  contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
  #util.np_info(contrast_stretch, "Contrast Stretch", t.elapsed())
  return contrast_stretch

"""
  Function: filter_histogram_equalization
  Filter image (gray or RGB) using histogram equalization to increase contrast in image.
  
  Parameters:
    - np_img: Image as a NumPy array (gray or RGB).
    - nbins: Number of histogram bins.
    - output_type: Type of array to return (float or uint8).
  
  Returns:
     NumPy array (float or uint8) with contrast enhanced by histogram equalization.
"""
def filter_histogram_equalization(np_img, nbins=256, output_type="uint8"):
  # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
  if np_img.dtype == "uint8" and nbins != 256:
    np_img = np_img / 255
  hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
  if output_type == "float":
    pass
  else:
    hist_equ = (hist_equ * 255).astype("uint8")
  #util.np_info(hist_equ, "Hist Equalization", t.elapsed())
  return hist_equ

"""
  Function: filter_adaptive_equalization
  Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
  is enhanced.
  
  Parameters:
    - np_img: Image as a NumPy array (gray or RGB).
    - nbins: Number of histogram bins.
    - clip_limit: Clipping limit where higher value increases contrast.
    - output_type: Type of array to return (float or uint8).
  
  Returns:
     NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
  """
def filter_adaptive_equalization(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
  adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
  if output_type == "float":
    pass
  else:
    adapt_equ = (adapt_equ * 255).astype("uint8")
  #util.np_info(adapt_equ, "Adapt Equalization", t.elapsed())
  return adapt_equ

"""
  Function: filter_local_equalization
  Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.
  
  Parameters:
    - np_img: Image as a NumPy array.
    - disk_size: Radius of the disk structuring element used for the local histograms
  
  Returns:
    NumPy array with contrast enhanced using local equalization.
"""
def filter_local_equalization(np_img, disk_size=50):
  local_equ = sk_filters.rank.equalize(np_img, selem=sk_morphology.disk(disk_size))
  #util.np_info(local_equ, "Local Equalization", t.elapsed())
  return local_equ

"""
  Function: filter_rgb_to_hed
  Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.
  
  Parameters:
    - np_img: RGB image as a NumPy array.
    - output_type: Type of array to return (float or uint8).
  
  Returns:
    NumPy array (float or uint8) with HED channels.
"""
def filter_rgb_to_hed(np_img, output_type="uint8"):
  hed = sk_color.rgb2hed(np_img)
  if output_type == "float":
    hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
  else:
    hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")

  #util.np_info(hed, "RGB to HED", t.elapsed())
  return hed

"""
  Function: filter_rgb_to_hsv
  Filter RGB channels to HSV (Hue, Saturation, Value).
  
  Parameters:
    - np_img: RGB image as a NumPy array.
    - display_np_info: If True, display NumPy array info and filter time.
  
  Returns:
    Image as NumPy array in HSV representation.
"""
def filter_rgb_to_hsv(np_img, display_np_info=True):
  hsv = sk_color.rgb2hsv(np_img)
  #if display_np_info:
    #util.np_info(hsv, "RGB to HSV", t.elapsed())
  return hsv

"""
  Function: filter_hsv_to_h
  Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float
  values are multiplied by 360 for their degree equivalents for simplicity. For more information, see
  https://en.wikipedia.org/wiki/HSL_and_HSV
  
  Parameters:
    - hsv: HSV image as a NumPy array.
    - output_type: Type of array to return (float or int).
    - display_np_info: If True, display NumPy array info and filter time.
  
  Returns:
    Hue values (float or int) as a 1-dimensional NumPy array.
"""
def filter_hsv_to_h(hsv, output_type="int", display_np_info=True):   
  h = hsv[:, :, 0]
  h = h.flatten()
  if output_type == "int":
    h *= 360
    h = h.astype("int")
  #if display_np_info:
    #util.np_info(hsv, "HSV to H", t.elapsed())
  return h

"""
  Function: filter_hsv_to_s
  Experimental HSV to S (saturation).
  
  Parameters:
    - hsv:  HSV image as a NumPy array.
  
  Returns:
    Saturation values as a 1-dimensional NumPy array.
"""
def filter_hsv_to_s(hsv, flatten = False):
  s = hsv[:, :, 1]
  if flatten:
    s = s.flatten()
  return s

"""
  Function: filter_hsv_to_v
  Experimental HSV to V (value).
  
  Parameters:
    - hsv:  HSV image as a NumPy array.
  
  Returns:
    Value values as a 1-dimensional NumPy array.
"""
def filter_hsv_to_v(hsv):
  v = hsv[:, :, 2]
  v = v.flatten()
  return v

"""
  Function: filter_hed_to_hematoxylin
  Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.
  
  Parameters:
    - np_img: HED image as a NumPy array.
    - output_type: Type of array to return (float or uint8).
  
  Returns:
    NumPy array for Hematoxylin channel.
"""
def filter_hed_to_hematoxylin(np_img, output_type="uint8"): 
  hema = np_img[:, :, 0]
  if output_type == "float":
    hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
  else:
    hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
  #util.np_info(hema, "HED to Hematoxylin", t.elapsed())
  return hema

"""
  Function: filter_hed_to_eosin
  Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
  contrast.
  
  Parameters:
    - np_img: HED image as a NumPy array.
    - output_type: Type of array to return (float or uint8).
  
  Returns:
    NumPy array for Eosin channel.
"""
def filter_hed_to_eosin(np_img, output_type="uint8"): 
  eosin = np_img[:, :, 1]
  if output_type == "float":
    eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
  else:
    eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype("uint8")
  #util.np_info(eosin, "HED to Eosin", t.elapsed())
  return eosin

"""
  Function: filter_binary_fill_holes
  Fill holes in a binary object (bool, float, or uint8).
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where holes have been filled.
"""
def filter_binary_fill_holes(np_img, output_type="bool"): 
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_fill_holes(np_img)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Fill Holes", t.elapsed())
  return result

"""
  Function: filter_binary_erosion
  Erode a binary object (bool, float, or uint8).
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - disk_size: Radius of the disk structuring element used for erosion.
    - iterations: How many times to repeat the erosion.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where edges have been eroded.
"""
def filter_binary_erosion(np_img, disk_size=5, iterations=1, output_type="uint8"): 
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_erosion(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Erosion", t.elapsed())
  return result

"""
  Function:filter_binary_dilation
  Dilate a binary object (bool, float, or uint8).
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - disk_size: Radius of the disk structuring element used for dilation.
    - iterations: How many times to repeat the dilation.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) where edges have been dilated.
"""
def filter_binary_dilation(np_img, disk_size=5, iterations=1, output_type="uint8"): 
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_dilation(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Dilation", t.elapsed())
  return result

"""
  Function: filter_binary_opening
  Open a binary object (bool, float, or uint8). Opening is an erosion followed by a dilation.
  Opening can be used to remove small objects.
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - disk_size: Radius of the disk structuring element used for opening.
    - iterations: How many times to repeat.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) following binary opening.
"""
def filter_binary_opening(np_img, disk_size=3, iterations=1, output_type="uint8"): 
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_opening(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Opening", t.elapsed())
  return result

"""
  Function: filter_binary_closing
  Close a binary object (bool, float, or uint8). Closing is a dilation followed by an erosion.
  Closing can be used to remove small holes.
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - disk_size: Radius of the disk structuring element used for closing.
    - iterations: How many times to repeat.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array (bool, float, or uint8) following binary closing.
"""
def filter_binary_closing(np_img, disk_size=3, iterations=1, output_type="uint8"): 
  if np_img.dtype == "uint8":
    np_img = np_img / 255
  result = sc_morph.binary_closing(np_img, sk_morphology.disk(disk_size), iterations=iterations)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Binary Closing", t.elapsed())
  return result

"""
  Function: filter_kmeans_segmentation
  Use K-means segmentation (color/space proximity) to segment RGB image where each segment is
  colored based on the average color for that segment.
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - compactness: Color proximity versus space proximity factor.
    - n_segments: The number of segments.
  
  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment.
"""
def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800):
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  result = sk_color.label2rgb(labels, np_img, kind='avg')
  #util.np_info(result, "K-Means Segmentation", t.elapsed())
  return result

"""
  Function: filter_rag_threshold
  Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine
  similar regions based on threshold value, and then output these resulting region segments.
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - compactness: Color proximity versus space proximity factor.
    - n_segments: The number of segments.
    - threshold: Threshold value for combining regions.
  
  Returns:
    NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
    color for that segment (and similar segments have been combined).
"""
def filter_rag_threshold(np_img, compactness=10, n_segments=800, threshold=9):
  labels = sk_segmentation.slic(np_img, compactness=compactness, n_segments=n_segments)
  g = sk_future.graph.rag_mean_color(np_img, labels)
  labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
  result = sk_color.label2rgb(labels2, np_img, kind='avg')
  #util.np_info(result, "RAG Threshold", t.elapsed())
  return result

"""
  Function: filter_threshold
  Return mask where a pixel has a value if it exceeds the threshold value.
  
  Parameters:
    - np_img: Binary image as a NumPy array.
    - threshold: The threshold value to exceed.
    - output_type: Type of array to return (bool, float, or uint8).
  
  Returns:
    NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
    pixel exceeds the threshold value.
  """
def filter_threshold(np_img, threshold, output_type="bool"): 
  result = (np_img > threshold)
  if output_type == "bool":
    pass
  elif output_type == "float":
    result = result.astype(float)
  else:
    result = result.astype("uint8") * 255
  #util.np_info(result, "Threshold", t.elapsed())
  return result