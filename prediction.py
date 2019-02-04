
import os
import sys
import math
import random
import argparse
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from keras.optimizers import Adam
from image_feature import image_feature_extract
from keras.models import Sequential,model_from_json

class image_class_prediction:

	def __init__(self,model_type,limit):
		self.model_type = model_type
		self.data_limit = limit
		if model_type == 'lgbm':
			self.lgb_model = lgb.Booster(model_file='data/models/gbm_model.model')
		if model_type == 'xgbm':
			self.xgb_model = xgb.Booster(model_file='data/models/xgb_model.model')
		if model_type == 'cnn':
			self.cnn_model = self.load_cnn_model()
		self.image_list = self.load_images()

	def load_cnn_model(self):

		json_file = open('data/models/cnn_model.json','r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		loaded_model.load_weights('data/models/cnn_model.h5')
		loaded_model.compile(loss = 'binary_crossentropy', optimizer=Adam(lr=0.1),metrics = ['accuracy'])
		print("Loaded Model from disk")
		return loaded_model

	def predict_image(self,img_location):
		if self.model_type == 'lgbm':
			feature = image_feature_extract().extract_feature(img_location)
			feature_reshape = np.array(feature).reshape((1, -1))
			predict_val = self.lgb_model.predict(feature_reshape)[0]
			if predict_val > 0.5:
				predict = 1
			else:
				predict = 0
			return predict,predict_val

		if self.model_type == 'xgbm':
			feature = image_feature_extract().extract_feature(img_location)
			feature_reshape = np.array(feature).reshape((1, -1))
			feature_reshape = xgb.DMatrix(feature_reshape)
			predict_val = self.xgb_model.predict(feature_reshape)
			if predict_val > 0.5:
				predict = 1
			else:
				predict = 0
			return predict,predict_val[0]

		if self.model_type == 'cnn':
			feature = image_feature_extract().extract_feature(img_location)
			feature_reshape = np.array(feature).reshape((1, -1))
			predict_val = self.cnn_model.predict(feature_reshape)
			if predict_val[0][0] > 0.5:
				predict = 1
			else:
				predict = 0
			return predict,predict_val[0][0]

	def evaluate_model(self):
		correct_count = 0
		class_0_conf_score = 0
		class_1_conf_score = 0
		class_0_count = 0
		class_1_count = 0
		for image_data in self.image_list:
			prediction,predict_val = self.predict_image(image_data[0])	#0 if uninfected 1 if otherwise
			if prediction == image_data[1]:
				correct_count +=1
				if prediction == 0:
					class_0_conf_score += predict_val
				elif prediction == 1:
					class_1_conf_score += predict_val
				sys.stdout.write('\r'+'Correct Predictions {}/{}..'.format(correct_count,len(self.image_list)))

			if prediction == 0:
				class_0_count +=1
			else:
				class_1_count +=1

		print('\n{} Prediction Accuracy.\n{} Uninfected prediction average.\n{} Infected prediction average'.format(correct_count/len(self.image_list),
																														class_0_conf_score/class_0_count,
																														class_1_conf_score/class_1_count))

	def load_images(self):
		class_0_directory = "data/cell-images/Uninfected"	#Uninfected cell images directory
		class_1_directory = "data/cell-images/Parasitized"	#Infected cell images directory
		limit = int(self.data_limit/2)
		class_0_images = os.listdir(class_0_directory)[0:limit]
		class_1_images = os.listdir(class_1_directory)[0:limit]
		total_file_list = []

		for image in class_0_images:
			if image.split('.')[1] == 'png':
				total_file_list.append([class_0_directory+'/'+image,0])

		for image in class_1_images:
			if image.split('.')[1] == 'png':
				total_file_list.append([class_1_directory+'/'+image,1])

		random.shuffle(total_file_list)	#shuffle the data
		return total_file_list

	def sigmoid(self,x):
		return 1 / (1 + math.exp(-x))


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Training Arguments')
	help_message="Model Train Methods\n1. lgbm - Light Gradient Boosting 2. xgbm - Xtreme Gradient Boosting. 3.cnn - Keras Convolutional Neural Network"
	parser.add_argument("-m", "--model", help = help_message, type=str, required=True)
	parser.add_argument("-l", "--limit", help = help_message, type=int, required=True)
	args = parser.parse_args()
	image_class_prediction(args.model,args.limit).evaluate_model()