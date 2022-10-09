
from itertools import count
import os
from time import sleep
import numpy as np
from scipy import stats
from glob import glob
from scipy.ndimage.interpolation import shift
import gc
from .My_Image import Dataset_Config, Image_Cropper, Normalize_MOS_STD
from .My_Learning import Load_Model

#region Local Functions

def Patches_std(algo_array):
	return np.std(algo_array)

def Fix_std():
	return 0.02

def Sense_std(img_mos):
	# std = 0
	# if img_mos <= 0.5:
	# 	std = (0.03*img_mos) + 0.01
	# else:
	# 	std = (-0.03*img_mos) + 0.04
	
	# if std < 0.01:
	# 	std = 0.01
	# elif std > 0.025:
	# 	std = 0.025
	
	# return std

	if img_mos >= 0.7:
		return 0.01
	elif img_mos < 0.4:
		return 0.01
	else:
		return 0.02

def Min_std(std):
	if std < 0.01:
		std = 0.01
	return std

def Normal_Distribution(score, std):
	x_values = np.arange(0, 1, 0.01)

	mean = score
	if mean < 0:
		mean = 0
	if mean > 1:
		mean = 1

	# if std < 0.003:
	# 	std = 0.003
	
	# if std < 0.01:
	# 	std = 0.01
	
	y_values = stats.norm(mean, std)
	y_values = y_values.pdf(x_values)
	
	#########################################

	if np.sum(y_values) == 0:
		print("Error - Histogram 0!")
		print("Mean:", mean)
		print("std:", std)
		exit()

	#########################################

	# y_values = y_values/np.max(y_values)
	# y_values[y_values < 0.000001] = 0

	#########################################

	y_values = y_values/np.sum(y_values)
	y_values[y_values < 0.000001] = 0

	diff = 1 - np.sum(y_values)
	index = int(mean*100)
	if index > 99:
		index = 99
	y_values[index] = y_values[index] + diff
	
	#########################################

	return y_values

#endregion

def Patch_Data(dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip = []):
	
	if not os.path.isdir(save_path + '/nn1_data/'):
		os.makedirs(save_path + '/nn1_data/')

	exception, reference_path, distortion_folders, distortion_path = Dataset_Config(dataset_name, dataset_path)

	if len(reference_array)>0:

		patch_data_array = np.array([])
		patch_algo_array = np.array([])

		dist_image_array = []
		dist_patch_array = []
		dist_fix_array = []
		dist_sense_array = []
		dist_min_array = []
		dist_one_array = []

		musig_image_array = []
		musig_patch_array = []
		musig_sense_array = []
		
		row = 0
		for file in distortion_folders:
			print(file)

			for image_path in glob(distortion_path + file + '/*' + exception):

				image_name = image_path.split(os.path.sep)[-1]

				distortion_type = file
				if dataset_name == 'TID2013' or dataset_name == 'TID2008':
					distortion_type = str(int(image_name.split('_')[1]))

				patch_array, algo_array, img_mos, img_alg, var_array, img_std, img_ref, img_lvl = Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold, distortion_skip)
				
				img_mos, img_std = Normalize_MOS_STD(dataset_name, img_mos, img_std)

				row += 1
				if img_ref not in reference_array:
					print(row, distortion_type, '->', image_name, '(Skip)')
				
				elif len(patch_array) > 0:
					print(row, distortion_type, '->', image_name)

					# For Patch
					if len(patch_data_array) == 0:
						patch_data_array = patch_array
						patch_algo_array = algo_array
					
					else:
						patch_data_array = np.append(patch_data_array, patch_array, axis=0)
						patch_algo_array = np.append(patch_algo_array, algo_array, axis=0)
					
					# For Histogram
					patches_std = Patches_std(algo_array)
					fix_std = Fix_std()
					sense_std = Sense_std(img_mos)

					for alg in algo_array:
						# STD distribution
						if img_std == -1:
							img_std = patches_std
						dist_image_array.append(Normal_Distribution(alg, Min_std(img_std)))
						musig_image_array.append(np.array([alg, Min_std(img_std)]))
						
						# Patch distribution
						dist_patch_array.append(Normal_Distribution(alg, Min_std(patches_std)))
						musig_patch_array.append(np.array([alg, Min_std(patches_std)]))

						# Fix distribution
						dist_fix_array.append(Normal_Distribution(alg, fix_std))
						
						# Sense distribution
						dist_sense_array.append(Normal_Distribution(alg, sense_std))
						musig_sense_array.append(np.array([alg, sense_std]))

						# Min-std distribution
						dist_min_array.append(Normal_Distribution(alg, 0.005))

						# One-bit distribution
						one_hot = np.zeros(100)
						index = int(alg*100)
						if index > 99:
							index = 99
						one_hot[index] = 1
						dist_one_array.append(one_hot)

				else:
					print(row, distortion_type, '->', image_name, '(No Patch)')
		
		#region save data
		dist_image_array = np.asarray(dist_image_array)
		dist_patch_array = np.asarray(dist_patch_array)
		dist_fix_array = np.asarray(dist_fix_array)
		dist_sense_array = np.asarray(dist_sense_array)
		dist_min_array = np.asarray(dist_min_array)
		dist_one_array = np.asarray(dist_one_array)

		musig_image_array = np.asarray(musig_image_array)
		musig_patch_array = np.asarray(musig_patch_array)
		musig_sense_array = np.asarray(musig_sense_array)

		np.save(save_path + '/nn1_data/' + save_name + '_input-patch.npy', patch_data_array)
		np.save(save_path + '/nn1_data/' + save_name + '_algo.npy', patch_algo_array)

		np.save(save_path + '/nn1_data/' + save_name + '_image.npy', dist_image_array)
		np.save(save_path + '/nn1_data/' + save_name + '_patch.npy', dist_patch_array)
		np.save(save_path + '/nn1_data/' + save_name + '_fix.npy', dist_fix_array)
		np.save(save_path + '/nn1_data/' + save_name + '_sense.npy', dist_sense_array)
		np.save(save_path + '/nn1_data/' + save_name + '_min.npy', dist_min_array)
		np.save(save_path + '/nn1_data/' + save_name + '_one.npy', dist_one_array)

		# np.save(save_path + '/nn1_data/' + save_name + '_musig_image.npy', musig_image_array)
		# np.save(save_path + '/nn1_data/' + save_name + '_musig_patch.npy', musig_patch_array)
		# np.save(save_path + '/nn1_data/' + save_name + '_musig_sense.npy', musig_sense_array)
		#endregion

	return len(patch_algo_array)

def Distribution_Data(label, model_loss, dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip = []):
	
	if not os.path.isdir(save_path + '/nn2_data/'):
		os.makedirs(save_path + '/nn2_data/')
	
	exception, reference_path, distortion_folders, distortion_path = Dataset_Config(dataset_name, dataset_path)

	if len(reference_array)>0:

		patch_model = 'NN_Simple_Distribution_' + model_loss.upper() + '_' + label
		patch_model = 'patch_model_' + patch_model + '.h5'
		patch_model = Load_Model(save_path + '/patch/' + patch_model)

		input_original_array = []
		mos_original_array = []
		target_original_array = []

		input_shifted_array = []
		mos_shifted_array = []
		target_shifted_array = []

		row = 0
		for file in distortion_folders:
			print(file)

			for image_path in glob(distortion_path + file + '/*' + exception):
				image_name = image_path.split(os.path.sep)[-1]

				distortion_type = file
				if dataset_name == 'TID2013' or dataset_name == 'TID2008':
					distortion_type = str(int(image_name.split('_')[1]))

				patch_array, algo_array, img_mos, img_alg, var_array, img_std, img_ref, img_lvl = Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold, distortion_skip)
				
				img_mos, img_std = Normalize_MOS_STD(dataset_name, img_mos, img_std)

				row += 1
				if img_ref not in reference_array:
					print(row, distortion_type, '->', image_name, '(Skip)')
				
				# elif img_mos <= 0:
				# 	print(row, distortion_type, '->', image_name, '(Refrence)')
				
				elif len(patch_array) > 0:
					print(row, distortion_type, '->', image_name)

					# Normalize patches from 0-255 to 0-1
					patch_array = patch_array.astype('float32')
					patch_array /= 255

					# predict patches score
					patch_predict = patch_model.predict(patch_array)

					histogram_sum = np.sum(patch_predict, axis=0)
					if np.sum(histogram_sum) > 0:
						histogram_sum = histogram_sum / np.sum(histogram_sum)
						histogram_sum[histogram_sum < 0.00001] = 0

						# print(np.sum(histogram_sum))
						# print(histogram_sum)
						# sleep(3)

						target_array = []

						# STD distribution
						if label == "image":
							if img_std == -1:
								img_std = Patches_std(algo_array)
							target_array = Normal_Distribution(img_mos, Min_std(img_std))
						
						# Patch distribution
						elif label == "patch":
							patches_std = Patches_std(algo_array)
							target_array = Normal_Distribution(img_mos, Min_std(patches_std))
						
						# Fix distribution
						elif label == "fix":
							fix_std = Fix_std()
							target_array = Normal_Distribution(img_mos, fix_std)
						
						# Sense distribution
						elif label == "sense":
							sense_std = Sense_std(img_mos)
							target_array = Normal_Distribution(img_mos, sense_std)
						
						# Min-std distribution
						elif label == "min":
							target_array = Normal_Distribution(img_mos, 0.005)

						# One-bit distribution
						elif label == "one":
							target_array = np.zeros(100)
							index = int(img_mos*100)
							if index > 99:
								index = 99
							target_array[index] = 1

						# Error
						else:
							print("NN2 Label Error!")
							print(label)
							exit()
						
						input_original_array.append(histogram_sum)
						mos_original_array.append(img_mos)
						target_original_array.append(target_array)
						
						max_shift = 5 + 1
						non_zeros = np.nonzero(histogram_sum)[0]
						left_shift = non_zeros[0]
						right_shift = non_zeros[-1]

						# count = 0

						for x in range(left_shift, left_shift - max_shift, -1):
							if x < 0:
								break
							# count += 1
							shift_step = left_shift-x
							shifted_histogram = shift(histogram_sum, -shift_step, order= 1, cval=0.0)

							input_shifted_array.append(shifted_histogram)
							mos_shifted_array.append(img_mos)
							target_shifted_array.append(target_array)
						
						for x in range(right_shift, right_shift + max_shift):
							if x > len(histogram_sum)-1:
								break
							# count += 1
							shift_step = x-right_shift
							shifted_histogram = shift(histogram_sum, shift_step, order= 1, cval=0.0)
							
							input_shifted_array.append(shifted_histogram)
							mos_shifted_array.append(img_mos)
							target_shifted_array.append(target_array)
						
						# print("shift:", count)

					else:
						print("Error - Evalute 0!")
						print("Mean:", img_mos)
						print("std:", img_std)
					
				else:
					print(row, distortion_type, '->', image_name, '(No Patch)')

		# save file

		input_original_array = np.asarray(input_original_array)
		mos_original_array = np.asarray(mos_original_array)
		target_original_array = np.asarray(target_original_array)

		input_shifted_array = np.asarray(input_shifted_array)
		mos_shifted_array = np.asarray(mos_shifted_array)
		target_shifted_array = np.asarray(target_shifted_array)

		np.save(save_path + '/nn2_data/' + save_name + '_hist_original_' + model_loss.upper() + '_' + label + '_input.npy', input_original_array)
		np.save(save_path + '/nn2_data/' + save_name + '_hist_original_mos.npy', mos_original_array)
		np.save(save_path + '/nn2_data/' + save_name + '_hist_original_' + label + '.npy', target_original_array)

		np.save(save_path + '/nn2_data/' + save_name + '_hist_shifted_' + model_loss.upper() + '_' + label + '_input.npy', input_shifted_array)
		np.save(save_path + '/nn2_data/' + save_name + '_hist_shifted_' + model_loss.upper() + '_' + label + '_mos.npy', mos_shifted_array)
		np.save(save_path + '/nn2_data/' + save_name + '_hist_shifted_' + model_loss.upper() + '_' + label + '.npy', target_shifted_array)

		gc.collect()

	return len(input_original_array), len(input_shifted_array)
