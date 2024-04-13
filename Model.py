### YOUR CODE HERE
import torch
import torch.nn as nn
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record
from tqdm import tqdm

"""This script defines the training, validation and testing process.
"""

class MyModel(nn.Module):

    def __init__(self, configs):
        super(MyModel, self).__init__()
        self.model_configs = configs
        self.network = MyNetwork(configs)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=0.1)

    def train(self, x_train, y_train, configs, x_valid=None, y_valid=None):
        self.network.train()

        num_samples = x_train.shape[0]
        num_batches = num_samples // configs["batch_size"]

        print("### Training... ###")
        for epoch in range(1, configs["max_epochs"]+1):
            start_time = time.time()

            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = x_train[shuffle_index]
            curr_y_train = y_train[shuffle_index]

            if epoch % 50 == 0:
                for params in self.optimizer.param_groups:
                    params["learning_rate"] /= 10

            for i in range(num_batches):
                start_idx = i * configs["batch_size"]
                end_idx = min((i + 1) * configs["batch_size"], curr_x_train.shape[0])
                batch_data = curr_x_train[start_idx: end_idx]
                batch_labels = torch.tensor(curr_y_train[start_idx: end_idx], dtype=torch.int64).cuda()
                parsed_batch_data = []
                for record in batch_data:
                    parsed_record = parse_record(record, True)
                    parsed_batch_data.append(parsed_record)
                parsed_batch_data = torch.tensor(np.array(parsed_batch_data), dtype=torch.float32).cuda()
                y_pred = self.network(parsed_batch_data)
                self.loss = self.criterion(y_pred, batch_labels)
                for w in self.network.parameters():
                    self.loss += configs["weight_decay"] * torch.norm(w)
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                print('Batch {:d}/{:d} Loss {:.6f}'.format(i, num_batches, self.loss), end='\r', flush=True)
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.6f} Duration {:.3f} seconds.'.format(epoch, self.loss, duration))

            if epoch % configs["save_interval"] == 0:
                checkpoint_path = os.path.join(configs["save_dir"], f'{self.model_configs["name"]}-{epoch}.ckpt')
                os.makedirs(configs["save_dir"], exist_ok=True)
                torch.save(self.network.state_dict(), checkpoint_path)
                print("Checkpoint has been created.")


    def evaluate(self, x, y, checkpoint_num_list, model_dir):
        self.network.eval()
        if len(checkpoint_num_list) == 1:
            print('### Validation ###')
            type_ = "Validation"
        else:
            print('### Testing ###')
            type_ = "Test"

        for checkpoint_num in checkpoint_num_list:
            checkpointfile = os.path.join(model_dir, f'{self.model_configs["name"]}-{checkpoint_num}.ckpt')
            ckpt = torch.load(checkpointfile, map_location="cpu")
            self.network.load_state_dict(ckpt, strict=True)
            print(f"Restored model parameters from {checkpointfile}")

            preds = []
            for i in tqdm(range(x.shape[0])):

                record = x[i]
                parsed_record = torch.tensor(parse_record(record, True), dtype=torch.float32).cuda()
                out = self.network(parsed_record.unsqueeze(0))
                _, pred = torch.max(out,1)
                preds.append(pred)

            y = torch.tensor(y)
            preds = torch.tensor(preds)
            print(f'{type_} accuracy: {torch.sum(preds==y)/y.shape[0]:.4f}')
