
def dice_coefficient(y_true, y_pred, smooth=0):
	from keras import backend as K

	intersection = K.sum((y_true * y_pred), axis=-1)
	dc = (2. * intersection + smooth) / (K.sum(K.square(y_true), axis=-1) + K.sum(K.square(y_pred), axis=-1) + smooth)
	return dc

def DCL(y_true, y_pred):
	return 1 - dice_coefficient(y_true, y_pred)

def EMD(y_true, y_pred):
	from keras import backend as K

	"""
	Compute the Earth Mover's Distance loss.
	Hou, Le, Chen-Ping Yu, and Dimitris Samaras. "Squared Earth Mover's
	Distance-based Loss for Training Deep Neural Networks." arXiv preprint
	arXiv:1611.05916 (2016).
	"""
	cdf_ytrue = K.cumsum(y_true, axis=-1)
	cdf_ypred = K.cumsum(y_pred, axis=-1)
	# samplewise_emd = K.sqrt(K.mean(K.square(K.abs(cdf_ytrue - cdf_ypred)), axis=-1))
	samplewise_emd = K.sqrt(K.mean(K.square(cdf_ytrue - cdf_ypred), axis=-1))
	return K.mean(samplewise_emd)

def KLD(y_true, y_pred):
	from keras import backend as K
	
	y_true = K.clip(y_true, K.epsilon(), 1)
	y_pred = K.clip(y_pred, K.epsilon(), 1)
	return K.sum(y_true * K.log(y_true / y_pred), axis=-1)
