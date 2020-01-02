import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.preprocessing import Normalizer
from sklearn.metrics import confusion_matrix,accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,roc_curve,auc
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

import datetime

def my_LSTM():
	model = tf.keras.Sequential()
	model.add(layers.LSTM(128))
	model.add(layers.BatchNormalization())
	model.add(layers.Dense(64,activation = "sigmoid"))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(64,activation = "sigmoid"))
	model.add(layers.BatchNormalization())
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(2,activation = "softmax"))

	model.compile(optimizer = tf.keras.optimizers.Adam(0.001),
		loss = "binary_crossentropy",
		metrics = ["binary_accuracy"])


	return model

def training_transform(df):
	return df[parameters].to_numpy().reshape(-1,1,length)

def one_hot_to_label(pred):
	return np.argmax(pred,axis =1)

def print_curve(answer,pred):
	fpr,tpr,thresholds = roc_curve(answer,pred[:,-1])
	auc_val =auc(fpr,tpr)
	plt.plot(fpr,tpr,label = f"ROC curve(area = {auc_val})")
	plt.legend()
	plt.title("ROC curve")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.grid(True)
	plt.show()

def print_metrics(answer,pred):
	labels = [0,1]
	cm = confusion_matrix(answer,pred,labels = labels)
	cm_labeled = pd.DataFrame(cm, columns = labels, index = labels)
	print("confusion_matrix= \n", cm_labeled)
	print("accuracy = ", accuracy_score(y_true = answer, y_pred = pred))
	print("precision = ", precision_score(y_true = answer, y_pred = pred))
	print("recall_score = ", recall_score(y_true = answer, y_pred = pred))
	print("f1_score = ", f1_score(y_true = answer, y_pred = pred))

def preprocessing(df):
	df.fillna(0,inplace = True)
	df[parameters] = pd.DataFrame(Normalizer().fit_transform(df[parameters]))
	return df

if __name__ == '__main__':
	df = pd.read_csv("./CSV_Files/creditcard.csv")
	parameters = df.columns[:-1]
	length = len(parameters)
	df = preprocessing(df)

	X =df.loc[:,parameters]
	Y =df.loc[:,"Class"]

	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 99)
	X_train,X_test = training_transform(X_train),training_transform(X_test)
	Y_train = to_categorical(Y_train)

	model = my_LSTM()
	model.fit(X_train, Y_train, epochs = 10, batch_size = 64, validation_split = 0.2, verbose = 1)
	model.summary()

	pred = model.predict(X_test)
	print('auc_roc', roc_auc_score(Y_test, pred[:,1]))
	Y_predict = one_hot_to_label(pred)
	print_metrics(Y_test,Y_predict)
	print_curve(Y_test,pred)