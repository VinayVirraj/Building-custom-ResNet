### YOUR CODE HERE
import torch
import os, argparse
import numpy as np
from Model import MyModel
from Network import MyNetwork
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs
from ImageUtils import visualize, parse_record


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs)
	network = MyNetwork(model_configs)
	# print(network)

	if args.mode == 'train':
		x_train, y_train, _, _ = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)
		# print(torch.tensor(parse_record(x_train[0], True)).unsqueeze(0).shape)
		# network.forward(torch.tensor(parse_record(x_train[0], True), dtype=torch.float32).unsqueeze(0))

		# model.train(x_train, y_train, training_configs, x_valid, y_valid)
		# model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		_, _, x_test, y_test = load_data(args.data_dir)
		# model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# # Loading private testing dataset
		# x_test = load_testing_images(args.data_dir)
		# # visualizing the first testing image to check your image shape
		# visualize(x_test[0], 'test.png')
		# # Predicting and storing results on private testing dataset 
		# predictions = model.predict_prob(x_test)
		# np.save(args.result_dir, predictions)
		...

### END CODE HERE

