
import os
import numpy as np
from PIL import Image
from glob import glob

def Dataset_Config(dataset_name, dataset_path):
	exception = ''
	reference_path = ''
	distortion_path = ''
	distortion_folders = np.array([])

	# Dataset Config
	if dataset_name == 'LIVE':
		exception = '.bmp'
		reference_path = dataset_path + 'refimgs/'
		distortion_path = dataset_path
		distortion_folders = np.array([
										'fastfading',
										'gblur',
										'jp2k',
										'jpeg',
										'wn'
									])
	
	elif dataset_name == 'TID2013' or dataset_name == 'TID2008':
		exception = '.bmp'
		reference_path = dataset_path + 'reference_images/'
		# distortion_path = dataset_path + 'distorted_images/'
		distortion_path = dataset_path
		distortion_folders = np.array(['distorted_images'])
	
	elif dataset_name == 'CSIQ':
		exception = '.png'
		reference_path = dataset_path + 'src_imgs/'
		distortion_path = dataset_path + 'dst_imgs/'
		distortion_folders = np.array([
										'awgn',
										'blur',
										'contrast',
										'fnoise',
										'jpeg',
										'jpeg2000'
									])
	
	else:
		print('Error: Undefine Dataset')
		exit()
	
	return exception, reference_path, distortion_folders, distortion_path

def Reference_List(dataset_name, dataset_path):
	exception, reference_path, distortion_folders, distortion_path = Dataset_Config(dataset_name, dataset_path)
	
	reference_array = np.array([])
	for reference in glob(reference_path + '*' + exception):
		reference_array = np.append(reference_array, (reference.split(os.path.sep)[-1]).split('.')[0])
	
	return reference_array

def Reference_Split(dataset_name, dataset_path, validation_rate=0.2, test_rate=0.2):
	from random import randrange
	
	reference_array = Reference_List(dataset_name, dataset_path)
	print(reference_array)
	print()

	selected_array = np.array([])

	test_array = np.array([])
	x = 0
	rate = np.round(len(reference_array)*test_rate)
	while x < rate:
		selected_reference = reference_array[randrange(reference_array.shape[0])]
		if selected_reference not in selected_array:
			selected_array = np.append(selected_array, selected_reference)
			test_array = np.append(test_array, selected_reference)
			x+=1

	validation_array = np.array([])
	x = 0
	rate = np.round(len(reference_array)*validation_rate)
	while x < rate:
		selected_reference = reference_array[randrange(reference_array.shape[0])]
		if selected_reference not in selected_array:
			selected_array = np.append(selected_array, selected_reference)
			validation_array = np.append(validation_array, selected_reference)
			x+=1

	train_array = np.array([])
	for x in reference_array:
		if x not in selected_array:
			train_array = np.append(train_array, x)

	test_array = np.sort(test_array)
	validation_array = np.sort(validation_array)
	train_array = np.sort(train_array)

	return train_array, validation_array, test_array

def Reference_Split_Reader(dataset_name, dataset_path, save_path, save_name, test_mode = 'single'):
	train_array = np.array([])
	validation_array = np.array([])
	test_array = np.array([])

	if not os.path.isfile(save_path + '/' + str(save_name) + '.txt'):
		validation_rate = 0.2
		test_rate = 0.2
		if test_mode == 'generality':
			test_rate = 0
		
		train_array, validation_array, test_array = Reference_Split(dataset_name, dataset_path, validation_rate, test_rate)

		text_file = open(save_path + '/' + str(save_name) + '.txt', "a")
		text_file.write('Train Images:\n')
		text_file.write(str(train_array))
		text_file.write('\n')
		text_file.write('Validation Images:\n')
		text_file.write(str(validation_array))
		text_file.write('\n')
		text_file.write('Test Images:\n')
		text_file.write(str(test_array))
		text_file.write('\n\n')
		text_file.close()
	
	else:
		text_file = open(save_path + '/' + str(save_name) + '.txt', "r")
		text_file_data = text_file.read()
		text_file_lines = text_file_data.split('\n')
		line_num = 0
		while line_num < len(text_file_lines):
			if text_file_lines[line_num] == "Train Images:":
				plus = 1
				while plus > 0:
					array_data = text_file_lines[line_num+plus]
					left_box = False
					if len(array_data.split('[')) == 2:
						left_box = True
					right_box = False
					if len(array_data.split(']')) == 2:
						right_box = True
					if left_box == True and right_box == True:
						array_data = array_data.split('[')[1]
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')
						plus = 0
					elif left_box == True:
						array_data = array_data.split('[')[1]
						array_data = array_data.split(' ')
						plus += 1
					elif right_box == True:
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')[1:]
						plus = 0
					else:
						array_data = array_data.split(' ')[1:]
						plus += 1
					for ref_name in array_data:
						train_array = np.append(train_array, ref_name.split("'")[1])
			
			if text_file_lines[line_num] == "Validation Images:":
				plus = 1
				while plus > 0:
					array_data = text_file_lines[line_num+plus]
					left_box = False
					if len(array_data.split('[')) == 2:
						left_box = True
					right_box = False
					if len(array_data.split(']')) == 2:
						right_box = True
					if left_box == True and right_box == True:
						array_data = array_data.split('[')[1]
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')
						plus = 0
					elif left_box == True:
						array_data = array_data.split('[')[1]
						array_data = array_data.split(' ')
						plus += 1
					elif right_box == True:
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')[1:]
						plus = 0
					else:
						array_data = array_data.split(' ')[1:]
						plus += 1
					for ref_name in array_data:
						validation_array = np.append(validation_array, ref_name.split("'")[1])
			
			if text_file_lines[line_num] == "Test Images:":
				plus = 1
				while plus > 0:
					array_data = text_file_lines[line_num+plus]
					left_box = False
					if len(array_data.split('[')) == 2:
						left_box = True
					right_box = False
					if len(array_data.split(']')) == 2:
						right_box = True
					if left_box == True and right_box == True:
						if len(array_data.split('[]')) == 2:
							break
						array_data = array_data.split('[')[1]
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')
						plus = 0
					elif left_box == True:
						array_data = array_data.split('[')[1]
						array_data = array_data.split(' ')
						plus += 1
					elif right_box == True:
						array_data = array_data.split(']')[0]
						array_data = array_data.split(' ')[1:]
						plus = 0
					else:
						array_data = array_data.split(' ')[1:]
						plus += 1
					for ref_name in array_data:
						test_array = np.append(test_array, ref_name.split("'")[1])
			
			line_num += 1
	
	print('Train length: ' + str(len(train_array)))
	print(train_array)
	print()

	print('Validation length: ' + str(len(validation_array)))
	print(validation_array)
	print()

	print('Test length: ' + str(len(test_array)))
	print(test_array)
	print()

	return train_array, validation_array, test_array

def Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold = -1, distortion_skip = []):
	
	image_name = image_path.split(os.path.sep)[-1]
	
	score_file = score_file.loc[score_file['Image Name'] == image_name]
	score_file['Distortion Type'] = score_file['Distortion Type'].astype(str)
	score_file = score_file.loc[score_file['Distortion Type'] == distortion_type]

	if len(score_file) == 0:
		print("No data in score file!")
		print("image name:", image_name)
		print("distortion type:", distortion_type)
		print("variance threshold:", variance_threshold)
		exit()
	
	img_ref = str(score_file['Reference Name'].values[0])
	img_lvl = -1
	img_mos = -1
	img_std = -1
	img_alg = -1
	patch_array = []
	var_array = []
	algo_array = []

	if variance_threshold >= 0:
		score_file = score_file.loc[score_file['Patch Variance'] >= variance_threshold]
	
	if len(score_file) == 0:
		pass
	
	elif distortion_type in distortion_skip:
		pass

	elif img_ref in reference_array:
		img_lvl = int(score_file['Distortion Level'].values[0])
		img_mos = float(score_file['Score'].values[0])
		img_std = float(score_file['STD'].values[0])
		img_alg = float(score_file[algorithm_name + ' Image'].values[0])
		
		img_data = np.array(Image.open(image_path))

		for index in range(0, len(score_file)):
			patch = score_file.iloc[index]
			patch_X = int(patch['X'])
			patch_Y = int(patch['Y'])
		
			patch_XL = int(patch['X Length'])
			patch_YL = int(patch['Y Length'])
		
			patch_var = float(patch['Patch Variance'])
			patch_alg = float(patch[algorithm_name + ' Patch'])
			
			patch_data = img_data[patch_Y:patch_Y+patch_YL, patch_X:patch_X+patch_XL, :]
		
			patch_array.append(patch_data)
			var_array.append(patch_var)
			algo_array.append(patch_alg)
			
		patch_array = np.asarray(patch_array)
		algo_array = np.asarray(algo_array)
		var_array = np.asarray(var_array)

	return patch_array, algo_array, img_mos, img_alg, var_array, img_std, img_ref, img_lvl

def Reverse_Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold = -1, distortion_skip = []):
	image_name = image_path.split(os.path.sep)[-1]
	
	score_file = score_file.loc[score_file['Image Name'] == image_name]
	score_file['Distortion Type'] = score_file['Distortion Type'].astype(str)
	score_file = score_file.loc[score_file['Distortion Type'] == distortion_type]

	if len(score_file) == 0:
		print("No data in score file!")
		print("image name:", image_name)
		print("distortion type:", distortion_type)
		print("variance threshold:", variance_threshold)
		exit()
	
	score_file = score_file.sort_values(by=['Y'])

	img_ref = str(score_file['Reference Name'].values[0])
	img_lvl = -1
	img_mos = -1
	img_std = -1
	img_alg = -1
	patch_array = []
	var_array = []
	algo_array = []

	if variance_threshold >= 0:
		score_file = score_file.loc[score_file['Patch Variance'] >= variance_threshold]
	
	if len(score_file) == 0:
		pass
	
	elif distortion_type in distortion_skip:
		pass

	elif img_ref in reference_array:
		img_lvl = int(score_file['Distortion Level'].values[0])
		img_mos = float(score_file['Score'].values[0])
		img_std = float(score_file['STD'].values[0])
		img_alg = float(score_file[algorithm_name + ' Image'].values[0])
		
		img_data = np.array(Image.open(image_path))

		for index in range(0, len(score_file)):
			patch = score_file.iloc[index]
			patch_X = int(patch['X'])
			patch_Y = int(patch['Y'])
		
			patch_XL = int(patch['X Length'])
			patch_YL = int(patch['Y Length'])
		
			patch_var = float(patch['Patch Variance'])
			patch_alg = float(patch[algorithm_name + ' Patch'])
			
			patch_data = img_data[patch_Y:patch_Y+patch_YL, patch_X:patch_X+patch_XL, :][:,::-1,:]

			patch_array.append(patch_data)
			var_array.append(patch_var)
			algo_array.append(patch_alg)
			
		patch_array = np.asarray(patch_array)
		algo_array = np.asarray(algo_array)
		var_array = np.asarray(var_array)

	return patch_array, algo_array, img_mos, img_alg, var_array, img_std, img_ref, img_lvl

def Normalize_MOS_STD(dataset_name, img_mos, img_std):
	if dataset_name == 'LIVE':
		img_mos = img_mos / 100
		img_std = img_std
		return img_mos, img_std
	if dataset_name == 'TID2013' or dataset_name == 'TID2008':
		img_mos = 1-(img_mos / 9)
		img_std = img_std / 9
		return img_mos, img_std
	if dataset_name == 'KADID':
		img_mos = img_mos / 5
		img_std = np.sqrt(img_std)
		img_std = img_std / 5
		return img_mos, img_std
	if dataset_name == 'KONIQ':
		img_mos = img_mos / 5
		img_std = img_std / 5
		return img_mos, img_std
	
	return img_mos, img_std
