# import x2paddle
# from x2paddle import torch2paddle
import paddle.nn as nn
from paddle.nn import Linear
from paddle.nn import Conv2D
from paddle.nn import BatchNorm1D
from paddle.nn import BatchNorm2D
from paddle.nn import PReLU
from paddle.nn import ReLU
from paddle.nn import Sigmoid
from paddle.nn import Dropout
from paddle.nn import MaxPool2D
from paddle.nn import AdaptiveAvgPool2D
from paddle.nn import Sequential
from paddle.nn import Layer
from collections import namedtuple
import paddle


class Flatten(nn.Layer):

    def forward(self, input):
        return input.view(input.size(0), -1)


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = input / norm
    return output


class SEModule(nn.Layer):

    def __init__(self, channels, reduction):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.fc1 = Conv2D(channels, channels // reduction, kernel_size=1,
            padding=0, bias_attr=False)
        # torch2paddle.xavier_normal_(self.fc1.weight.data)
        self.relu = nn.ReLU()
        self.fc2 = Conv2D(channels // reduction, channels, kernel_size=1,
            padding=0, bias_attr=False)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class bottleneck_IR(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2D(in_channel, depth, (1, 
                1), stride, bias_attr=False), BatchNorm2D(depth))
        self.res_layer = Sequential(BatchNorm2D(in_channel), Conv2D(
            in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(
            depth), Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False
            ), BatchNorm2D(depth))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class bottleneck_IR_SE(nn.Layer):

    def __init__(self, in_channel, depth, stride):
        super(bottleneck_IR_SE, self).__init__()
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(Conv2D(in_channel, depth, (1, 
                1), stride, bias_attr=False), BatchNorm2D(depth))
        self.res_layer = Sequential(BatchNorm2D(in_channel), Conv2D(
            in_channel, depth, (3, 3), (1, 1), 1, bias_attr=False), PReLU(
            depth), Conv2D(depth, depth, (3, 3), stride, 1, bias_attr=False
            ), BatchNorm2D(depth), SEModule(depth, 16))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth,
        depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 8:
        blocks = [get_block(in_channel=64, depth=64, num_units=3)]
    elif num_layers == 16:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4)]
    elif num_layers == 34:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4), get_block(
            in_channel=128, depth=256, num_units=9)]
    elif num_layers == 44:
        blocks = [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4),
            get_block(in_channel=128, depth=256, num_units=14)]
    elif num_layers == 50:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=4), get_block(
            in_channel=128, depth=256, num_units=14), get_block(in_channel=\
            256, depth=512, num_units=3)]
    elif num_layers == 100:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13), get_block(
            in_channel=128, depth=256, num_units=30), get_block(in_channel=\
            256, depth=512, num_units=3)]
    elif num_layers == 152:
        blocks = [get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=8), get_block(
            in_channel=128, depth=256, num_units=36), get_block(in_channel=\
            256, depth=512, num_units=3)]
    return blocks


class IRSEV2(nn.Layer):

    def __init__(self, input_size, num_layers, mode='ir', with_head=False,
        pretrained=None, return_index=(2,)):
        super().__init__()
        assert input_size[0] in [112, 224
            ], 'input_size should be [112, 112] or [224, 224]'
        assert num_layers in [0, 8, 16, 34, 44, 50, 100, 152
            ], 'num_layers should be 50, 100 or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        self.num_layers = num_layers
        self.return_index = return_index
        if num_layers == 0:
            return
        self.with_head = with_head
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2D(3, 64, (3, 3), 1, 1, bias_attr
            =False), BatchNorm2D(64), PReLU(64))
        if with_head:
            if input_size[0] == 112:
                self.output_layer = Sequential(BatchNorm2D(512), Dropout(),
                    Flatten(), Linear(512 * 7 * 7, 512), BatchNorm1D(512))
            else:
                self.output_layer = Sequential(BatchNorm2D(512), Dropout(),
                    Flatten(), Linear(512 * 14 * 14, 512), BatchNorm1D(512))
        modules = []
        max_stage = max(return_index)
        for block in blocks[:max_stage+1]:
            block_module = []
            for bottleneck in block:
                block_module.append(
                    unit_module(bottleneck.in_channel,
                                bottleneck.depth,
                                bottleneck.stride))
            modules.append(Sequential(*block_module))
        self.body = nn.LayerList(modules)

        if pretrained:
            self.init_weights(pretrained)

    def forward(self, x):
        if self.num_layers == 0:
            return x
        x = self.input_layer(x)
        output = []
        return_index = set(self.return_index)
        for index, m in enumerate(self.body):
            x = m(x)
            if index in return_index:
                output.append(x)
        if self.with_head:
            x = self.output_layer(x)

        return output
