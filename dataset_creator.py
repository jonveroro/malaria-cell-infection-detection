
import os
import sys
import random
import argparse
import numpy as np
import pandas as pd
from image_feature import image_feature_extract
class dataset_creator:

	def create_dataset(self,limit):
		dataset = []
		class_0_directory = "data/cell-images/Uninfected"	#Uninfected cell images directory
		class_1_directory = "data/cell-images/Parasitized"	#Infected cell images directory
		class_0_images = os.listdir(class_0_directory)[0:limit/2]
		class_1_images = os.listdir(class_1_directory)[0:limit/2]
		total_procesed = 0
		total_file_list = class_0_images+class_1_images

		file_data = open('data/datasets/file_list.csv','a+')

		for image in class_0_images:
			if image.split('.')[1] == 'png':
				file_data.write('{},0\n'.format(image))
				feature = image_feature_extract().extract_feature(class_0_directory+'/'+image)
				feature = np.append(feature,0)	#label as uninfected
				dataset.append(feature)
				total_procesed+=1
				sys.stdout.write('\r'+'Images feature extracted {}/{}..'.format(total_procesed,len(class_0_images)+len(class_1_images)))

		for image in class_1_images:
			if image.split('.')[1] == 'png':
				file_data.write('{},1\n'.format(image))
				feature = image_feature_extract().extract_feature(class_1_directory+'/'+image)
				feature = np.append(feature,1)	#label as infected
				dataset.append(feature)
				total_procesed+=1
				print('{}/{}'.format(total_procesed,len(class_0_images)+len(class_1_images)))
				sys.stdout.write('\r'+'Images feature extracted {}/{}..'.format(total_procesed,len(class_0_images)+len(class_1_images)))


		random.shuffle(dataset)	#shuffle the data
		dataset = np.array(dataset)
		dataset = pd.DataFrame(dataset)
		print("Writing to csv..")
		dataset.to_csv('data/datasets/img_feature_dataset.csv', index = None, header=True)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Training Arguments')
	help_message="Dataset sample size 1-27000"
	parser.add_argument("-l", "--limit", help = help_message, type=str, required=True)
	args = parser.parse_args()
	sample_limit = int(args.limit)
	dataset_creator().create_dataset(sample_limit)