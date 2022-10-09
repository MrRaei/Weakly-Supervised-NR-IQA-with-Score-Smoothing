
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

def logistic(values, beta1, beta2, beta3, beta4, beta5):
	logisticPart = 0.5 - 1/(1 + np.exp(beta2 * (values - beta3)))
	yHat = beta1 * logisticPart + beta4*values + beta5
	return yHat

def ModelAndScatterPlot(Data, MOS, fittedParameters, graphWidth, graphHeight, plot_name):
	f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
	axes = f.add_subplot(111)

	# first the raw data as a scatter plot
	axes.plot(Data, MOS, 'D')

	# create data for the fitted equation plot
	xModel = np.linspace(min(Data), max(Data))
	yModel = logistic(xModel, *fittedParameters)

	# now the model as a line plot 
	axes.plot(xModel, yModel)

	axes.set_xlabel('Predicted') # X axis data label
	axes.set_ylabel('MOS') # Y axis data label

	plt.savefig(plot_name+'.png') # save result figure

	# plt.show()
	plt.close('all') # clean up after using pyplot

def Find_Nonlinear_Fitting(file_path, real_score_index, predict_data_index, dataset_name, plot_name = 'Non-Linear Fitting Result'):

	dataset = pd.read_excel(file_path, engine='openpyxl')
	Data = dataset.iloc[1:, predict_data_index].values
	MOS = dataset.iloc[1:, real_score_index].values

	Data = Data.astype('float32')
	MOS = MOS.astype('float32')

	##########################################################

	# initialize the parameters used by the nonlinear fitting function

	# LIVE
	beta = np.array([np.max(MOS), np.min(MOS), np.mean(Data), 0.1, 40])

	# TID
	if dataset_name == 'TID2013' or dataset_name == 'TID2008':
		beta = np.array([np.max(MOS), 10, np.mean(Data), 0.1, 0.1])
	
	# curve fit the test data
	fittedParameters, pcov = curve_fit(logistic, Data, MOS, beta, maxfev=1000000)

	print('Parameters', fittedParameters)
	
	modelPredictions = logistic(Data, *fittedParameters)
	
	SROCC, _ = spearmanr(MOS, Data)
	print('SROCC (Data):', SROCC)

	SROCC_predict, _ = spearmanr(MOS, modelPredictions)
	print('SROCC (Predict):', SROCC_predict)

	RMSE = np.sqrt(mean_squared_error(MOS, modelPredictions))
	print('RMSE:', RMSE)

	PLCC, _ = pearsonr(MOS, Data)
	print('PLCC (Data):', PLCC)

	PLCC_predict, _ = pearsonr(MOS, modelPredictions)
	print('PLCC (Predict):', PLCC_predict)

	##########################################################
	
	# draw the fitted curve
	# graphics output section
	graphWidth = 800
	graphHeight = 600
	ModelAndScatterPlot(Data, MOS, fittedParameters, graphWidth, graphHeight, plot_name)

	return SROCC, RMSE, PLCC_predict, fittedParameters

# Find_Nonlinear_Fitting('NN_Simple_Regression_algo_TID2013_avg.xlsx', 5, 7, 'TID2013', plot_name = 'Non-Linear Fitting Result')