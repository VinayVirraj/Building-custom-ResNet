# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"num_of_classes": 10,
    "stack_lengths": [5, 7, 4],
    "first_num_filters": 64,
}

training_configs = {
	"learning_rate": 0.01,
	"batch_size": 32,
    "max_epochs": 50,
    "weight_decay": 2e-4,
    "validation_epochs": [10, 20, 30, 40 ,50],
    "save_interval": 10,
    "save_dir": './saved_models/',
}

### END CODE HERE