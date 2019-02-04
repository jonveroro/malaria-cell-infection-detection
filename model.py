
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Embedding, LSTM,Dropout, BatchNormalization

class modeler:

	def __init__(self):
		self.dataset = pd.read_csv('data/datasets/img_feature_dataset.csv',header=None)

	def split_data(self):
		label_encoder = LabelEncoder()
		X = self.dataset.drop(2048,axis=1)
		Y = self.dataset[self.dataset.columns[2048]]
		X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
		y_train = label_encoder.fit_transform(y_train)
		y_test = label_encoder.fit_transform(y_test)
		return X_train, X_test, y_train, y_test

	def lgbm_train(self):
		print('Training Light GBM Model..')

		####Split data to train and test vals######
		X_train, X_test, y_train, y_test = self.split_data()
		lgb_train = lgb.Dataset(X_train, y_train)
		lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
		
		##########Declare model parameters#########
		params = {}
		params['task'] = 'train'
		params['objective'] = 'binary'
		params['boosting_type'] = 'gbdt'
		params['learning_rate'] = 0.1
		params['num_leaves'] = 31	#default
		params['metric'] = {'l2', 'mse', 'binary'}
		params['is_training_metric'] = True
		params['feature_fraction'] = 0.99
		params['bagging_fraction'] = 0.9
		params['bagging_freq'] = 5
		params['num_boost_round'] = 500
		params['verbose'] = 1
		#params['early_stopping_rounds'] = 20

		###############Train model#################
		gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval)

		################Save model#################
		print('Saving Model..')
		gbm.save_model('data/models/gbm_model.model')

	def xgb_train(self):
		print('Training XGBoost Model..')

		####Split data to train and test vals######
		X_train, X_test, y_train, y_test = self.split_data()
		dtrain = xgb.DMatrix(X_train, label=y_train)
		dtest = xgb.DMatrix(X_test, label=y_test)

		##########declare model parameters#########
		params = {}
		params['task'] = 'train'
		params['booster'] = 'gbtree'
		params['objective'] = 'binary:logistic'
		params['max_leaves'] = 31
		params['learning_rate'] = 0.1
		params['grow_policy '] = 'depthwise'
		params['eval_metric'] = 'rmse'
		num_round = 500
		watchlist = [(dtrain, 'train')]

		###############Train model#################
		xgb_model = xgb.train(params, dtrain, num_round, watchlist)

		################Save model#################
		print('Saving Model..')
		xgb_model.save_model('data/models/xgb_model.model')

	def cnn_train(self):
		print('Training keras CNN Model..')

		####Split data to train and test vals######
		X_train, X_test, y_train, y_test = self.split_data()

		##########Declare model parameters#########
		model = Sequential()
		model.add(Dense(100, activation="relu" ,input_dim = X_train.shape[1]))
		model.add(Dense(50, activation="relu",kernel_initializer='normal'))
		model.add(Dense(10, activation="relu",kernel_initializer='normal'))
		#model.add(Dense(16, activation="relu",kernel_initializer='normal'))
		model.add(Dropout(0.5))
		model.add(Dense(1, activation="sigmoid",kernel_initializer='normal'))
		model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=0.01),metrics = ['accuracy'])
		print(model.summary())

		###############Train model#################
		model.fit(X_train, y_train,validation_data=(X_test,y_test), nb_epoch = 1, batch_size=32, verbose = 1)

		################Save model#################
		model_json = model.to_json()
		with open("data/models/cnn_model.json", "w+") as json_file:
			json_file.write(model_json)
		model.save_weights("data/models/cnn_model.h5")

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Training Arguments')
	help_message="Model Train Methods\n1. lgbm - Light Gradient Boosting 2. xgbm - Xtreme Gradient Boosting."
	parser.add_argument("-m", "--model", help = help_message, type=str, required=True)
	args = parser.parse_args()
	if args.model == 'lgbm':
		modeler().lgbm_train()
	if args.model == 'xgbm':
		modeler().xgbm_train()
	if args.model == 'cnn':
		modeler().cnn_train()