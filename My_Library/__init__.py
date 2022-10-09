
from .My_Data import *
from .My_Fitting import *
from .My_Image import *

from .My_Learning import *
from .My_Networks import *

##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
#region Data

def Patch_Data(dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip = []):
	return My_Data.Patch_Data(dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip)

def Distribution_Data(label, model_loss, dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip = []):
	return My_Data.Distribution_Data(label, model_loss, dataset_name, dataset_path, score_file, algorithm_name, variance_threshold, reference_array, save_path, save_name, distortion_skip)

#endregion
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
#region Fitting

def Find_Nonlinear_Fitting(file_path, real_score_index, predict_data_index, dataset_name, plot_name = 'Non-Linear Fitting Result'):
	return My_Fitting.Find_Nonlinear_Fitting(file_path, real_score_index, predict_data_index, dataset_name, plot_name)

#endregion
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
#region Image

def Reference_List(dataset_name, dataset_path):
	return My_Image.Reference_List(dataset_name, dataset_path)

def Reference_Split_Reader(dataset_name, dataset_path, save_path, save_name, test_mode = 'single'):
	return My_Image.Reference_Split_Reader(dataset_name, dataset_path, save_path, save_name, test_mode)

def Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold = -1, distortion_skip = []):
	return My_Image.Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold, distortion_skip)

def Reverse_Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold = -1, distortion_skip = []):
	return My_Image.Reverse_Image_Cropper(score_file, image_path, distortion_type, algorithm_name, reference_array, variance_threshold, distortion_skip)

def Normalize_MOS_STD(dataset_name, img_mos, img_std):
	return My_Image.Normalize_MOS_STD(dataset_name, img_mos, img_std)

#endregion
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
#region Learning

def Train(model, x_train, y_train, x_valid, y_valid, model_loss, model_metric, epochs, save_path, weight_name, early_stop = 0):
	return My_Learning.Train(model, x_train, y_train, x_valid, y_valid, model_loss, model_metric, epochs, save_path, weight_name, early_stop)

def Test(model, x_test, y_test):
	return My_Learning.Test(model, x_test, y_test)

def Save_Model(model, save_path, model_name, weight_file_name = ""):
	return My_Learning.Save_Model(model, save_path, model_name, weight_file_name)

def Load_Model(model_path):
	return My_Learning.Load_Model(model_path)

def GPU_Setup():
	return My_Learning.GPU_Setup()

#endregion
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
#region Networks

def NN_Simple_Regression(img_patch_height, img_patch_width):
	return My_Networks.NN_Simple_Regression(img_patch_height, img_patch_width)

def NN_Simple_Mu_Sigma(img_patch_height, img_patch_width):
	return My_Networks.NN_Simple_Mu_Sigma(img_patch_height, img_patch_width)

def NN_Simple_Distribution(img_patch_height, img_patch_width):
	return My_Networks.NN_Simple_Distribution(img_patch_height, img_patch_width)

def NN_Histogram_Regression(hist_bins):
	return My_Networks.NN_Histogram_Regression(hist_bins)

def NN_Histogram_Distribution(hist_bins):
	return My_Networks.NN_Histogram_Distribution(hist_bins)

#endregion
##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%##
