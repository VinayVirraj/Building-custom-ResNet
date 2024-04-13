### YOUR CODE HERE
import argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")

args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs).cuda()

	x_train, y_train, x_test, y_test = load_data(args.data_dir)

	if args.mode == 'train':
		x_train_p, y_train_p, x_valid, y_valid = train_valid_split(x_train, y_train)
		# print(torch.tensor(parse_record(x_train[0], True)).unsqueeze(0).shape)
		# network.forward(torch.tensor(parse_record(x_train[0], True), dtype=torch.float32).unsqueeze(0))

		model.train(x_train_p, y_train_p, training_configs, x_valid, y_valid)
		model.evaluate(x_valid, y_valid, training_configs["validation_epochs"], training_configs["save_dir"])

	elif args.mode == 'test':
		# Testing on public testing dataset
		# _, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test, [training_configs["validation_epochs"][-1]], training_configs["save_dir"])

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

