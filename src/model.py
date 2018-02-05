import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import keras
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.regularizers import l2, l1
from keras import backend as K
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import fbeta_score
from sklearn.model_selection import KFold
import numpy as np
import cv2
import sys
import tqdm
from tqdm import *
K.set_image_dim_ordering('th')
from os import listdir
from os.path import isfile, join
from random import shuffle

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model


def new_model(input_shape=(128, 128,3),weight_path=None):
        model = Sequential()
        model.add(BatchNormalization(input_shape=input_shape))
        model.add(Conv2D(64, kernel_size=(3, 3),padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(38, activation='sigmoid'))
	return model


def star_model(input_shape=(128,128,3),weight_path=None):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Conv2D(32, kernel_size=(5, 5),padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    return model

def Train(input_shape = (128, 128, 1)):

	global x_train
	global y_train

	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)

	x_t = []
	y_t = []

	for idx in range(x_train.shape[0]):
		x_t.append(x_train[idx])
		y_t.append(y_train[idx])
		aux_x_t = x_t[idx]
		aux_y_t = y_t[idx]
		flipped_img=cv2.flip(x_t[idx],1)
		rows,cols,channel = input_shape
		x_t.append(flipped_img)
		y_t.append(y_train[idx])

		for rotate_degree in [90,180,270]:
		    M = cv2.getRotationMatrix2D((cols/2,rows/2),rotate_degree,1)
		    dst = cv2.warpAffine(aux_x_t,M,(cols,rows))
		    x_t.append(dst)
		    y_t.append(aux_y_t)

		    dst = cv2.warpAffine(flipped_img,M,(cols,rows))
		    x_t.append(dst)
		    y_t.append(aux_y_t)

	y_train = np.array(y_t, np.uint8)
	x_train = np.array(x_t, np.uint8)

	h = KFold_Train(x_train,y_train)
	return h

def KFold_Train(x_train,y_train,nfolds=6,batch_size=128):
	#model = new_model(input_shape)
	model = star_model(input_shape)
	print model.summary()
	kf = KFold(n_splits=nfolds, shuffle=True, random_state=1)
	num_fold = 0
	for train_index, test_index in kf.split(x_train, y_train):
		start_time_model_fitting = time.time()
		X_train = x_train[train_index]
		Y_train = y_train[train_index]
		X_valid = x_train[test_index]
		Y_valid = y_train[test_index]

		#model = new_model(input_shape)
		model = star_model(input_shape)
		num_fold += 1
		print('Start KFold number {} from {}'.format(num_fold, nfolds))
		print('Split train: ', len(X_train), len(Y_train))
		print('Split valid: ', len(X_valid), len(Y_valid))

		kfold_weights_path = os.path.join('', 'weights_kfold_' + str(num_fold) + '.h5')

		epochs_arr =  [50, 30, 20, 10]
		learn_rates = [0.001, 0.0001, 0.00001, 0.000001]

		for learn_rate, epochs in zip(learn_rates, epochs_arr):
		    print('Start Learn_rate number {} from {}'.format(epochs,learn_rate))
		    opt  = optimizers.Adam(lr=learn_rate)
		    model.compile(loss='binary_crossentropy', # We NEED binary here, since categorical_crossentropy l1 norms the output before calculating loss.
		                  optimizer=opt,
		                  metrics=['accuracy'])
		    callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=1),
		    ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0)]

		    history = model.fit(x = X_train, y= Y_train, validation_data=(X_valid, Y_valid),
		          batch_size=32,verbose=1, epochs=epochs,callbacks=callbacks,shuffle=True)

		if os.path.isfile(kfold_weights_path):
		    model.load_weights(kfold_weights_path)
		p_valid = model.predict(X_valid, batch_size = 32, verbose=2)
	return history

def KFold_Predict(x_test,nfolds=6,batch_size=128, result = './star_result_full'):
    model = star_model(input_shape)
    yfull_test = []
    for num_fold in range(1,nfolds+1):
        weight_path = os.path.join('', result + 'weights_kfold_' + str(num_fold) + '.h5')
        if os.path.isfile(weight_path):
            model.load_weights(weight_path)

        p_test = model.predict(x_test, batch_size = batch_size, verbose=2)
        yfull_test.append(p_test)

    result = np.array(yfull_test[0])
    for i in range(1, nfolds):
	print 'Nfold = ' + str(i)
        result += np.array(yfull_test[i])
    result /= nfolds
    return result


def get_one(list_v):
	for idx, e in enumerate(list_v):
		max_v = np.max(list_v)
		if e == max_v:
			return idx
	return 37

def Predict(x_test):
	output = KFold_Predict(x_test)
	result = []
	matrix = [[0 for x in range(38)] for y in range(38)]
	for i in range(0, output.shape[0]):
	    predict_one = get_one(output[i])
	    y_one = get_one(y_test[i])
	    if predict_one == y_one:
                result.append(1)
	    else:
                result.append(0)
	    x_var = int(y_one)
	    y_var = int(predict_one)
	    matrix[x_var][y_var] += 1

	return result, matrix




seed = 7
numpy.random.seed(seed)


mypath = './db/'
nfolds=6
size_img = 50
input_shape= (1, size_img, size_img)
num_classes = 38
sum_score = 0
#size_train = 700#
size_train = 50000


x_train = []
y_train = []
x_test = []
y_test = []

#onlyfiles = [f for f in listdir(mypath + '/train_reduced/')]
onlyfiles = [f for f in listdir(mypath + '/train/')]

X = []
Y = []


#shuffle(onlyfiles)
out = pd.read_csv(mypath + 'training_solutions_rev1.csv')

def get_y(num):
    id_v = 0
    for idx, e in enumerate(out['GalaxyID']):
        if e == num:
            id_v = idx
    y = 0
    y_id = 0
    for idx, e in enumerate(out):
        if idx>0:
            if y < out[e][id_v]:
                y = out[e][id_v]
                y_id = idx
    a = np.zeros(38)
    a[y_id] = 1
    return a

print 'Iniciando leitura dos dados : ', len(onlyfiles), 'imagens para test'

time.sleep(3)

for f in tqdm(onlyfiles):

    img = cv2.imread(mypath+'train/{}'.format(f))
    Y.append(get_y(int(f[:-4])))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    X.append(cv2.resize(gray, (size_img, size_img)))

X = np.array(X)
Y = np.array(Y)

X = X.reshape(X.shape[0], 1, size_img, size_img).astype('float32')
X = X/255

x_train = X[:size_train]
y_train = Y[:size_train]
x_test = X[size_train:]
y_test = Y[size_train:]


print 'Iniciando programa'


# execfile('main_model.py')
# execfile('preparation.py')
# execfile('train.py')
# execfile('predict.py')


def main():

    if(sys.argv[1] == 'train'):
	h = Train(input_shape)
    if(sys.argv[1] == 'predict'):
	p, matrix = Predict(x_test)
	a = 0
	for e in range(y_test.shape[0]):
		if p[e] == 1:
			a +=1
	#print y_test
	print 'Total de acerto = ',a
	print 'Acc = ', float(a)/y_test.shape[0]
	for e in range(num_classes):
		print matrix[e]

if __name__ == '__main__':
    main()
