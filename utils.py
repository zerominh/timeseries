import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import classification_report
from collections import Counter


def create_table_index(dim , tau, m):
	"""
	Arguments:
	m-- length of time series

	Returns:
	indx-- the indx of each vector
	"""
	num_vectors = m - (dim - 1)*tau
	indx = np.zeros((num_vectors, dim))
	indx[0,:] = np.arange(0, dim*tau, tau).astype(int)

	for i in range(1,num_vectors):
		indx[i,:] = indx[i-1,:]+1
	
	return indx.ravel()




def filter(data_well, label_well, label, dim ,tau):
	

	length = []

	vector_data_well_label = []

	index = np.where(label_well == label)[0]
	b = index[0]
	e = index[0]

	for i in range(len(index)-1):
		# khong lien tiep
		if(index[i+1] - index[i] != 1):
			if((e + 1 - b) > ((dim-1)*tau)):
				vector_data_well_label.append(data_well[b:e+1, 1:])

			# statistic length of sub timeseries
			if e+1-b > 0:
				length.append(e+1-b)

			b = index[i+1]
		# lien tiep
		else:
			e = index[i+1]



	# print('vector_data_well_label.shape: ', vector_data_well_label[0].shape)

	# # compute length of each sub timeseries
	# print('length: ', sorted(Counter(length).items(), key=lambda i: i[0]))

	return vector_data_well_label


def get_vector_each_vector_timeseries(vector_data_well_label, dim, tau):


	# vector train for feature in column 1
	timeseries_train = np.array(vector_data_well_label[:, 0], copy=True)		
	indx_vectors_timeseries_train = create_table_index(dim, tau, timeseries_train.shape[0]).astype(int)
	vectors_train = timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))
	#vector train for another feature
	for i in range(1, vector_data_well_label.shape[1]):
		timeseries_train = np.array(vector_data_well_label[:, i], copy=True)
		vectors_train = np.concatenate((vectors_train, timeseries_train[indx_vectors_timeseries_train].reshape((timeseries_train.shape[0] - (dim -1)*tau, dim))), axis=1)


	return vectors_train


def extract_vector_train_each_well(X, y, dim, tau, label, train_well):


	data_well = X[np.where(X[:, 0] == train_well)[0], :]

	'''________________________________________________'''




	'''________________________________________________'''





	label_well = y[np.where(X[:, 0] == train_well)[0]]

	vector_data_well_label = filter(data_well, label_well, label, dim, tau)


	"""______vector train for first vector_____"""
	vectors_train = get_vector_each_vector_timeseries(vector_data_well_label[0], dim, tau)
	"""______vector train for another vector_____"""

	if(len(vector_data_well_label) >=2):
		for i in range(1, len(vector_data_well_label)):
			vectors_train = np.concatenate((vectors_train, get_vector_each_vector_timeseries(vector_data_well_label[i], dim, tau)))

	return vectors_train


def extract_vector_test(X, dim, tau):




	data_well_label = X

	timeseries_test = np.array(data_well_label[:, 1], copy=True)
	indx_vectors_timeseries_test = create_table_index(dim, tau, timeseries_test.shape[0]).astype(int)
	vectors_test = timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))

	# # vector train for other feature
	# for i in range(2, X.shape[1]):
	# 	timeseries_test = np.array(data_well_label[:, i], copy=True)
	# 	vectors_test = np.concatenate((vectors_test, timeseries_test[indx_vectors_timeseries_test].reshape((timeseries_test.shape[0] - (dim -1)*tau, dim))), axis=1)

	return vectors_test, indx_vectors_timeseries_test


def minMaxScalerPreprocessing(X, minScaler = 0.0, maxScaler = 253):
	return (X - minScaler)/(maxScaler - minScaler)





def create_train(training_well_ts, dim, tau, curve_number, facies_class_number):
	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# training_well_ts[:,0] = label_enc.fit_transform(training_well_ts[:, 0])

	X = training_well_ts[:, [0,curve_number]]
	y = training_well_ts[:, -1].astype(int)
	# print(type(y))
	train_well = list(set(X[:, 0]))

	# print('y: ')
	# print(y)
	# print('train_well', train_well)

	""" vector train of train_well[0]"""
	vectors_train = extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[0])
	"""_____________"""
	""" vector train for another train well"""
	if(len(train_well) > 1):
		for i in range(1, len(train_well)):
			vectors_train = np.concatenate((vectors_train, extract_vector_train_each_well(X, y, dim, tau, facies_class_number, train_well[i])))
			
	"""___________________________________________"""

	return vectors_train




def create_test(testing_well_ts, dim, tau, curve_number):

	# from sklearn.preprocessing import LabelEncoder
	# label_enc = LabelEncoder()
	# testing_well_ts[:,0] = label_enc.fit_transform(testing_well_ts[:, 0])

	X = testing_well_ts[:, [0, curve_number]]
	y = testing_well_ts[:, -1].astype(int)

	return extract_vector_test(X, dim, tau)








def rqa(training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number):

	'''
	
	@Parameters:
	training_well_ts -- numpy array 2D:
		the 1st column of type string (name of well)
		the last column of type integer: facies_class
		the another column: features

		Example:
		array([['RD-1P'		2555.4432	2434.7698	108.8463	0.2312	2.5599	84.4916	0.6982	0.036	5],
				['RD-1P'	2555.5956	2434.9184	101.5264	0.2011	2.586	81.334	0.617	0.0333	5],
				['RD-1P'	2557.7292	2436.9983	74.2481		0.1072	2.5488	68.2637	0.3139	0		3]])


	testing_well_ts -- numpy array 2d like training_well_ts but containing only data of one well
	
	dim : the dimension -- type: integer, greater than or equal to 2
	tau : the step -- type: integer, greater than or equal to 1
	epsilon: type of float greater than 0
	lambd: the positive integer
	percent: the float number between 0 and 1
	curve_number: the positive integer is index of column feature in training_well_ts
	facies_class_number: the integer greater than or equal to 0-- the name of class to detect

	@Return:
	predict_label-- numpy array 1D of shape (the length of testing_well_ts, ) containg only 0, 1:
		0: predict not facies_class_number
		1: predict belong to facies_class_number
	'''
	if not (percent > 0 and percent <= 1.0): 
		print('percent must > 0 and <= 1.0')
		raise AssertionError


	vectors_train = create_train(training_well_ts, dim, tau, curve_number, facies_class_number)


	vectors_test, indx_vectors_timeseries_test = create_test(testing_well_ts, dim, tau, curve_number)
	r_dist = cdist(vectors_train, vectors_test, 'minkowski', p=1)



	r = np.sum(r_dist < epsilon, axis=0)
	if __debug__:
		print(np.sum(r))

# 	"""____________________________________________________"""

	predict_label = np.zeros((testing_well_ts[:, -1].shape[0], ), dtype=int)
	indx_vectors_timeseries_test = indx_vectors_timeseries_test.reshape((vectors_test.shape[0], dim))
	# print('indx_vectors_timeseries_test.shape: ', indx_vectors_timeseries_test)

	index = indx_vectors_timeseries_test[r > lambd, :]
	index = index[:, 0]
	add_index = list(np.arange(0, dim*percent, dtype=int))

#	index = index[:, :int(dim*percent)].ravel()
	for i in add_index:
		predict_label[(index+i).ravel()] = 1

	return predict_label.ravel().tolist()


# def load_dataset(file_name='../data/data.csv'):
# 	# load dataset
# 	data = pd.read_csv(file_name)

# 	X = data.iloc[:, :].values


# 	# return X

def convert_list_to_nparray(data):

	data = np.array(data)

	new_data = np.array(data[:, 1:], dtype=np.float32)


	from sklearn.preprocessing import LabelEncoder
	label_enc = LabelEncoder()
	name_well = label_enc.fit_transform(data[:, 0].reshape(-1, 1)).reshape(-1, 1)

	new_data = np.concatenate((name_well, new_data), axis = 1)

	return new_data



def get_data_from_json(data_json):

	data = data_json['data']	
	if data == None:
		raise AssertionError
	params = data_json['params']
	if params == None:
		raise AssertionError

	training_well_ts_list = data['train']
	if training_well_ts_list == None:
		raise AssertionError

	training_well_ts = convert_list_to_nparray(training_well_ts_list)
	
	# print(type(training_well_ts[0, -1]))

	testing_well_ts_list = data['test']

	if testing_well_ts_list == None:
		raise AssertionError

	testing_well_ts = convert_list_to_nparray(testing_well_ts_list)

	dim = params['dim']
	if dim == None:
		raise AssertionError

	tau = params['tau']
	if tau == None:
		raise AssertionError

	epsilon = params['epsilon']
	if epsilon == None:
		raise AssertionError

	lambd = params['lambd']
	if lambd == None:
		raise AssertionError

	percent = params['percent']
	if percent == None:
		raise AssertionError

	curve_number = params['curve_number']
	if curve_number == None:
		raise AssertionError

	facies_class_number = params['facies_class_number']
	if facies_class_number == None:
		raise AssertionError

	return training_well_ts, testing_well_ts, dim, tau, epsilon, lambd, percent, curve_number, facies_class_number

def main():
	dim = 4
	tau = 2
	epsilon = 0.1
	lambd = 20
	percent = 1
	training_well_ts, testing_well_ts = get_data()
	# print(training_well_ts)
	# print(testing_well_ts)
	predict_vector = rqa(training_well_ts, testing_well_ts, dim=dim, tau=tau, epsilon=epsilon, lambd=lambd, percent=percent, curve_number=1, facies_class_number=5)

if __name__ == '__main__':
	main()

