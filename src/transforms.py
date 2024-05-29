import torch
import numpy as np

"""
	Class: RandomNormalize
	The random z-normalization.
"""
class RandomNormalize(object):
	"""
		Function: __init__
		The initialization function.
	"""
	def __init__(self, prob=0.5):
		self.prob = prob
	"""
		Function: normalize_
		The normalization function.

		Parameters:
			- image_array: The RGB image in numpy array

		Returns:
			- The image array after applying z-normalization.
	"""
	def normalize_(self, image_array):
		random_number=torch.rand(1)
		apply_transf= bool(self.prob>random_number)
		if not apply_transf:
			return image_array
		if image_array.shape[2] != 3:
			return image_array
		m = np.mean(image_array, axis = (0,1))
		std = np.std(image_array, axis = (0,1))
		output_array = (image_array - m)/std
		return output_array

	"""
		Function: __call__
		The call function to apply the transformation.

		Parameters:
			- image_array: The RGB image in numpy array

		Returns:
			- The image array after applying z-normalization.
	"""
	def __call__(self, image_array):
		out_array = self.normalize_(image_array)
		return out_array.astype('float32')
"""
	Class: ToTensor
	Convert the numpy array to tensor.
	
	Parameters:
		- np_array: The numpy array.

	Returns:
		The tensor.
"""
class ToTensor(object):
	def __init__(self):
		super().__init__()

	def __call__(self, np_array):
		np_array = np_array.astype('float32')
		np_array = np.transpose(np_array, axes = (2,0,1))
		tensor_array = torch.from_numpy(np_array)
		return tensor_array