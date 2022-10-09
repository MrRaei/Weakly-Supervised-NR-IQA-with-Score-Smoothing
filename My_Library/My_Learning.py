
import os

def Train(model, x_train, y_train, x_valid, y_valid, model_loss,
			model_metric, epochs, save_path, weight_name, early_stop = 0):

	from .My_Loses import DCL, EMD
	from tensorflow.keras.losses import MAE
	from tensorflow.keras.losses import kullback_leibler_divergence as kld
	from tensorflow.keras.losses import categorical_crossentropy as cat_cross
	from tensorflow.keras.losses import sparse_categorical_crossentropy as spars
	from tensorflow.keras.losses import cosine_similarity as cos_sim
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

	model_optimizer = Adam(learning_rate = 0.0001)

	if save_path != '':
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
	
	if model_loss == 'dcl':
		model.compile(optimizer=model_optimizer, loss=DCL, metrics=[EMD, kld, MAE])
	
	elif model_loss == 'kld':
		model.compile(optimizer=model_optimizer, loss=kld, metrics=[EMD, DCL, MAE])
	
	elif model_loss == 'emd':
		model.compile(optimizer=model_optimizer, loss=EMD, metrics=[DCL, kld, MAE])
	
	elif model_loss == 'mae':
		model.compile(optimizer=model_optimizer, loss=MAE, metrics=[EMD, kld, DCL])
	
	elif model_loss == 'cat-cross':
		model.compile(optimizer=model_optimizer, loss=cat_cross, metrics=[EMD, kld, DCL])
	
	elif model_loss == 'spars':
		model.compile(optimizer=model_optimizer, loss=spars, metrics=[EMD, kld, DCL])
	
	elif model_loss == 'cos-sim':
		model.compile(optimizer=model_optimizer, loss=cos_sim, metrics=[EMD, kld, DCL])
	
	else:
		model.compile(optimizer=model_optimizer, loss=model_loss, metrics=[model_metric])
	
	if early_stop == 0:
		early_stop = epochs
	
	earlyStop = EarlyStopping(monitor='val_loss', patience=early_stop, verbose=1, mode='auto')
	check_point_name = save_path + weight_name + '.hdf5'

	if len(x_valid) > 0:
		check_point = ModelCheckpoint(filepath = check_point_name, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
		history = model.fit(x_train, y_train,
							batch_size=32,
							epochs=epochs,
							verbose=1,
							validation_data=(x_valid, y_valid),
							callbacks=[earlyStop, check_point],
							shuffle=True)
	else:
		history = model.fit(x_train, y_train,
							batch_size=32,
							epochs=epochs,
							verbose=1,
							shuffle=True)
	
	return model, history, check_point_name

def Test(model, x_test, y_test):
	score = model.evaluate(x_test, y_test, verbose=0)
	return score

def Save_Model(model, save_path, model_name, weight_file_name = ""):
	if save_path != '':
		if not os.path.isdir(save_path):
			os.makedirs(save_path)
	
	if weight_file_name != "":
		# Save File With Best Founded Weights
		model.load_weights(weight_file_name)
	
	# Create a HDF5 File
	model.save(save_path + model_name + '.h5')

	return model

def Load_Model(model_path):
	from tensorflow.keras.models import load_model

	# Load Pre-Trained Model
	model = load_model(model_path, compile=False)
	return model

def GPU_Setup():
	import tensorflow as tf
	os.environ["KERAS_BACKEND"] = "tensorflow"
	kerasBKED = os.environ["KERAS_BACKEND"]
	# os.environ["TF_GPU_ALLOCATOR"]="cuda_malloc_async"
	# os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"

	gpus = tf.config.experimental.list_physical_devices('GPU')

	if gpus:
		try:
			# tf.config.experimental.set_virtual_device_configuration(
			# 	gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			# tf.config.experimental.set_memory_growth(gpus[0], True)
		except RuntimeError as e:
			print(e)
