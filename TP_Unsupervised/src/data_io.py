import numpy as np
import os.path as osp


def stl_read_labels(path_to_labels):
	with open(path_to_labels, 'rb') as f:
		labels = np.fromfile(f, dtype=np.uint8)
		return labels


def stl_read_all_images(path_to_data):
	# code by Martin Tutek, from the STL-10 repository
	with open(path_to_data, 'rb') as f:
		# read whole file in uint8 chunks
		everything = np.fromfile(f, dtype=np.uint8)
		images = np.reshape(everything, (-1, 3, 96, 96))
		images = np.transpose(images, (0, 3, 2, 1))
		return images


def stl_load_stl10(folder):
	''' Reads files from the STL-10 dataset.

	Reads the files of the STL-10 dataset and returns the training images,
	training labels, test images and test labels as numpy arrays.

	:param folder: Folder containing the files of the STL-10 dataset.
	:type folder: str

	:return: four Numpy arrays: training images (dimensions (5000, 96, 96,
		3)), training labels (dimensions (5000,), test images
		(dimensions (8000, 96, 96, 3)), and test labels (dimensions
		(8000,)).
	:rtype: (array of int, array of int, array of int, array of int)
	'''
	train_images = stl_read_all_images(osp.join(folder, 'train_X.bin'))
	train_labels = stl_read_labels(osp.join(folder, 'train_y.bin'))
	test_images = stl_read_all_images(osp.join(folder, 'test_X.bin'))
	test_labels = stl_read_labels(osp.join(folder, 'test_y.bin'))
	return train_images, train_labels, test_images, test_labels
