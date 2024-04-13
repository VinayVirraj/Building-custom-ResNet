### YOUR CODE HERE
import torch
import torch.nn as nn
from torch.functional import Tensor

"""This script defines the network.
"""

class MyNetwork(nn.Module):

    def __init__(self, configs):
        self.first_num_filters = configs["first_num_filters"]
        self.stack_lengths = configs["stack_lengths"]
        self.num_of_classes = configs["num_of_classes"]

        super(MyNetwork, self).__init__()

        self.start_layer = nn.Conv2d(3, self.first_num_filters, kernel_size=3, stride=1, padding=1)

        self.stack_layers = nn.ModuleList()
        for i in range(len(self.stack_lengths)):
            if i != 0:
                filters_in = self.first_num_filters * (2**(i-1))
            else:
                filters_in = self.first_num_filters
            filters_out = self.first_num_filters * (2**i)
            strides = 1 if i == 0 else 2
            self.stack_layers.append(build_stack_layer(filters_in, strides, self.stack_lengths[i], filters_out))
        self.output_layer = output_layer(filters_out, self.num_of_classes)
    
    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.start_layer(inputs)
        for i in range(len(self.stack_lengths)):
            outputs = self.stack_layers[i](outputs)
        outputs = self.output_layer(outputs)
        return outputs
    

class bottleneck_block(nn.Module):

    def __init__(self, in_filters, layer_first_block_stride, residual_projection, out_filters):
        super(bottleneck_block, self).__init__()
        self.bn1 = batch_norm_relu_layer(in_filters)
        self.conv1 = nn.Conv2d(in_filters, in_filters//4, kernel_size=1, stride=layer_first_block_stride)
        self.bn2 = batch_norm_relu_layer(in_filters//4)
        self.conv2 = nn.Conv2d(in_filters//4, in_filters//4, kernel_size=3, stride=1, padding=1)
        self.bn3 = batch_norm_relu_layer(in_filters//4)
        self.conv3 = nn.Conv2d(in_filters//4, in_filters//4, kernel_size=3, stride=1, padding=1)
        self.bn4 = batch_norm_relu_layer(in_filters//4)
        self.conv4 = nn.Conv2d(in_filters//4, out_filters, kernel_size=1, stride=1)
        self.residual_projection = residual_projection
    
    def forward(self, inputs: Tensor) -> Tensor:
        
        residual = inputs
        out = self.bn1(inputs)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.conv3(out)
        out = self.bn4(out)
        out = self.conv4(out)

        if self.residual_projection is not None:
            residual = self.residual_projection(inputs)

        out += residual
        return out


class batch_norm_relu_layer(nn.Module):

    def __init__(self, num_features, eps=1e-5, momentum=0.997) -> None:
        super(batch_norm_relu_layer, self).__init__()

        self.bn2d = nn.BatchNorm2d(num_features, eps, momentum)
        self.relu = nn.ReLU()

    def forward(self, inputs: Tensor) -> Tensor:
        out = self.bn2d(inputs)
        out = self.relu(out)
        return out
    

class build_stack_layer(nn.Module):

    def __init__(self, filters_in, layer_first_block_stride, stack_size, filters_output) -> None:
        super(build_stack_layer, self).__init__()

        projection_shortcut = nn.Conv2d(filters_in,
                                        filters_output,
                                        stride=layer_first_block_stride,
                                        kernel_size=1)
        # Only the first block per stack_layer uses projection_shortcut and layer_first_block_stride

        self.blocks = nn.ModuleList([
            bottleneck_block(filters_in, layer_first_block_stride, projection_shortcut, filters_output)
        ])

        for _ in range(1, stack_size):
            self.blocks.append(bottleneck_block(filters_output, 1, None, filters_output))

    
    def forward(self, inputs: Tensor) -> Tensor:

        for i,block in enumerate(self.blocks):
            if i == 0:
                out = block(inputs)
            else:
                out = block(out)

        return out


class output_layer(nn.Module):

    def __init__(self, filters, num_classes) -> None:
        super(output_layer, self).__init__()

        self.bn_relu = batch_norm_relu_layer(filters, eps=1e-5, momentum=0.997)
        self.fc = nn.Linear(filters, num_classes)

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, inputs: Tensor) -> Tensor:

        inputs = self.bn_relu(inputs)
        out = self.pooling(inputs)
        out = self.fc(out.view(out.size(0), -1))
        return out