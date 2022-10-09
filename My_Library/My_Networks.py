
def NN_Simple_Regression(img_patch_height, img_patch_width):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout

	print('Making Network ...')

	model = Sequential()
	model.add(Conv2D(32, (3,3), padding='same', activation='elu', input_shape=(img_patch_height, img_patch_width, 3)))
	model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(1, activation='relu'))

	print('Network Summary:')
	model.summary()
	print()

	return model

def NN_Simple_Mu_Sigma(img_patch_height, img_patch_width):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout

	print('Making Network ...')

	model = Sequential()
	model.add(Conv2D(32, (3,3), padding='same', activation='elu', input_shape=(img_patch_height, img_patch_width, 3)))
	model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(512, activation='elu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='relu'))

	print('Network Summary:')
	model.summary()
	print()

	return model

def NN_Simple_Distribution(img_patch_height, img_patch_width):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D, Dropout

	print('Making Network ...')

	model = Sequential()
	model.add(Conv2D(32, (3,3), padding='same', activation='elu', input_shape=(img_patch_height, img_patch_width, 3)))
	model.add(Conv2D(32, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(64, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(128, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(256, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(Conv2D(512, (3,3), padding='same', activation='elu'))
	model.add(MaxPool2D(pool_size=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(512, activation='sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='softmax'))
	# model.add(Dense(100, activation='relu'))

	print('Network Summary:')
	model.summary()
	print()
	
	return model

def NN_Histogram_Regression(hist_bins):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Flatten, Dense, Dropout

	print('Making Network ...')

	model = Sequential()
	model.add(Dense(128, activation='elu', input_shape=(hist_bins,)))
	model.add(Dense(128, activation='elu'))
	model.add(Dense(128, activation='elu'))
	model.add(Dense(128, activation='elu'))

	model.add(Dropout(0.5))
	model.add(Dense(hist_bins, activation='softmax'))
	model.add(Dense(1, activation='relu'))

	print('Network Summary:')
	model.summary()
	print()

	return model

def NN_Histogram_Distribution(hist_bins):
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Flatten, Dense, Dropout

	print('Making Network ...')

	model = Sequential()
	model.add(Dense(128, activation='elu', input_shape=(hist_bins,)))
	model.add(Dense(128, activation='elu'))
	model.add(Dense(128, activation='elu'))
	model.add(Dense(128, activation='elu'))

	model.add(Dropout(0.5))
	model.add(Dense(hist_bins, activation='softmax'))

	print('Network Summary:')
	model.summary()
	print()

	return model