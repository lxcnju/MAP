import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import copy

from utils import weights_init


def get_basic_net(
        net, n_classes, input_size=None, input_channel=None, bias=False):
    if net == "MLPNet":
        model = MLPNet(input_size, input_channel, n_classes, bias)
    elif net == "LeNet":
        model = LeNet(input_size, input_channel, n_classes, bias)
    elif net == "TFCNN":
        model = TFCNN(n_classes, bias)
    elif net == "VGG8":
        model = VGG(8, n_classes, False, bias)
    elif net == "VGG11":
        model = VGG(11, n_classes, False, bias)
    elif net == "VGG13":
        model = VGG(13, n_classes, False, bias)
    elif net == "VGG16":
        model = VGG(16, n_classes, False, bias)
    elif net == "VGG19":
        model = VGG(19, n_classes, False, bias)
    elif net == "VGG8-BN":
        model = VGG(8, n_classes, True, bias)
    elif net == "VGG11-BN":
        model = VGG(11, n_classes, True, bias)
    elif net == "VGG13-BN":
        model = VGG(13, n_classes, True, bias)
    elif net == "VGG16-BN":
        model = VGG(16, n_classes, True, bias)
    elif net == "VGG19-BN":
        model = VGG(19, n_classes, True, bias)
    elif net == "ResNet8":
        model = ResNet(8, n_classes, bias)
    elif net == "ResNet20":
        model = ResNet(20, n_classes, bias)
    elif net == "ResNet32":
        model = ResNet(32, n_classes, bias)
    elif net == "ResNet44":
        model = ResNet(44, n_classes, bias)
    elif net == "ResNet56":
        model = ResNet(56, n_classes, bias)
    elif net == "FeMnistNet":
        model = FeMnistNet(n_classes, bias)
    elif net == "CharLSTM":
        model = CharLSTM(n_classes, bias)
    elif net == "AudioDSCNN":
        model = AudioDSCNN(172, n_classes, bias)
    elif net == "OhsumedGRU":
        model = OhsumedGRU(20000, 300, n_classes, bias)
    else:
        raise ValueError("No such net: {}".format(net))

    model.apply(weights_init)

    return model


class FLDANet(nn.Module):

    def __init__(
        self, model
    ):
        super().__init__()
        self.model = copy.deepcopy(model)

        model.apply(weights_init)
        self.private_model = copy.deepcopy(model)

        self.private_weight_layer = nn.Linear(
            self.model.h_size, 1
        )

    def forward(self, xs):
        hs, logits = self.model(xs)
        phs, plogits = self.private_model(xs)

        alpha = self.private_weight_layer(hs).sigmoid()
        logits = alpha * logits + (1.0 - alpha) * plogits
        plogits = logits
        return logits, plogits

    def global_forward(self, xs):
        _, logits = self.model(xs)
        return logits


class FedDMLNet(nn.Module):

    def __init__(
        self, model
    ):
        super().__init__()

        self.model = copy.deepcopy(model)

        model.apply(weights_init)
        self.private_model = copy.deepcopy(model)

    def forward(self, xs):
        hs, logits = self.model(xs)
        phs, plogits = self.private_model(xs)

        return logits, plogits

    def global_forward(self, xs):
        _, logits = self.model(xs)
        return logits


class FedRODNet(nn.Module):

    def __init__(
        self, model
    ):
        super().__init__()
        self.model = copy.deepcopy(model)

        model.apply(weights_init)
        self.private_classifier = copy.deepcopy(
            model.classifier
        )

    def forward(self, xs):
        hs, logits = self.model(xs)
        plogits = self.private_classifier(hs)

        plogits = 0.5 * (plogits + logits)
        return logits, plogits

    def global_forward(self, xs):
        _, logits = self.model(xs)
        return logits


class Reshape(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, xs):
        return xs.reshape((xs.shape[0], -1))


class MLPNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10, bias=False):
        super().__init__()
        self.input_size = input_channel * input_size ** 2
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            Reshape(),
            nn.Linear(self.input_size, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 128),
            nn.ReLU(True),
        )

        self.h_size = 128

        self.classifier = nn.Linear(self.h_size, n_classes, bias=bias)

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class LeNet(nn.Module):
    def __init__(self, input_size, input_channel, n_classes=10, bias=False):
        super().__init__()
        self.input_size = input_size
        self.input_channel = input_channel
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channel, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 16, 5),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        if self.input_size == 28:
            self.h_size = 16 * 4 * 4
        elif self.input_size == 32:
            self.h_size = 16 * 5 * 5
        else:
            raise ValueError("No such input_size.")

        self.classifier = nn.Linear(self.h_size, n_classes, bias=bias)

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class FeMnistNet(nn.Module):
    def __init__(self, n_classes, bias=False):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 4)),
            Reshape(),
        )

        self.h_size = 64 * 4 * 4

        self.classifier = nn.Linear(self.h_size, n_classes, bias=bias)

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class TFCNN(nn.Module):
    def __init__(self, n_classes, bias=False):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            Reshape(),
        )

        self.h_size = 64 * 4 * 4

        self.classifier = nn.Linear(self.h_size, n_classes, bias=bias)

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class InnerLSTM(nn.Module):
    def __init__(self, lstm):
        super().__init__()
        self.lstm = lstm
        self.lstm_size = lstm.hidden_size

    def forward(self, xs):
        bs = xs.shape[0]
        h0 = torch.zeros(2, bs, self.lstm_size)
        c0 = torch.zeros(2, bs, self.lstm_size)

        h0 = h0.to(device=xs.device)
        c0 = c0.to(device=xs.device)

        outputs, _ = self.lstm(xs, (h0, c0))
        output = outputs[:, -1, :]
        return output


class CharLSTM(nn.Module):
    """ StackedLSTM for NLP
    """

    def __init__(self, n_classes, bias=False):
        super().__init__()
        self.n_classes = n_classes

        self.n_vocab = 81
        self.w_dim = 8
        self.lstm_size = 256

        embeddings = nn.Embedding(self.n_vocab, self.w_dim)
        lstm = nn.LSTM(
            input_size=self.w_dim,
            hidden_size=self.lstm_size,
            num_layers=2,
            batch_first=True,
        )
        inner_lstm = InnerLSTM(lstm)

        self.encoder = nn.Sequential(
            embeddings,
            inner_lstm,
        )

        self.h_size = self.lstm_size

        self.classifier = nn.Linear(
            self.lstm_size, self.n_classes, bias=bias
        )

    def forward(self, xs):
        code = self.encoder(xs)
        logits = self.classifier(code)
        return code, logits


class VGG(nn.Module):
    def __init__(
        self,
        n_layer=11,
        n_classes=10,
        use_bn=False,
        bias=False
    ):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes
        self.use_bn = use_bn

        self.cfg = self.get_vgg_cfg(n_layer)

        self.encoder = nn.Sequential(
            self.make_layers(self.cfg),
            Reshape(),
        )

        self.h_size = 512
        self.classifier = nn.Linear(self.h_size, n_classes, bias=bias)

    def get_vgg_cfg(self, n_layer):
        if n_layer == 8:
            cfg = [
                64, 'M',
                128, 'M',
                256, 'M',
                512, 'M',
                512, 'M'
            ]
        elif n_layer == 11:
            cfg = [
                64, 'M',
                128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 13:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 'M',
                512, 512, 'M',
                512, 512, 'M'
            ]
        elif n_layer == 16:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 'M',
                512, 512, 512, 'M',
                512, 512, 512, 'M'
            ]
        elif n_layer == 19:
            cfg = [
                64, 64, 'M',
                128, 128, 'M',
                256, 256, 256, 256, 'M',
                512, 512, 512, 512, 'M',
                512, 512, 512, 512, 'M'
            ]
        return cfg

    def conv3x3(self, in_channel, out_channel):
        layer = nn.Conv2d(
            in_channel, out_channel,
            kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        return layer

    def make_layers(self, cfg, init_c=3):
        block = nn.ModuleList()

        in_c = init_c
        for e in cfg:
            if e == "M":
                block.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                block.append(self.conv3x3(in_c, e))
                if self.use_bn is True:
                    block.append(nn.BatchNorm2d(e))
                block.append(nn.ReLU(inplace=True))
                in_c = e
        block = nn.Sequential(*block)
        return block

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes,
            kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes,
            kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ 6n + 2: 8, 14, 20, 26, 32, 38, 44, 50, 56
    """

    def __init__(self, n_layer=20, n_classes=10, bias=False):
        super().__init__()
        self.n_layer = n_layer
        self.n_classes = n_classes

        conv1 = nn.Conv2d(
            3, 16, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        bn1 = nn.BatchNorm2d(16)

        assert ((n_layer - 2) % 6 == 0), "SmallResNet depth is 6n+2"
        n = int((n_layer - 2) / 6)

        self.cfg = (BasicBlock, (n, n, n))
        self.in_planes = 16

        layer1 = self._make_layer(
            block=self.cfg[0], planes=16, stride=1, num_blocks=self.cfg[1][0],
        )
        layer2 = self._make_layer(
            block=self.cfg[0], planes=32, stride=2, num_blocks=self.cfg[1][1],
        )
        layer3 = self._make_layer(
            block=self.cfg[0], planes=64, stride=2, num_blocks=self.cfg[1][2],
        )

        avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.encoder = nn.Sequential(
            conv1,
            bn1,
            nn.ReLU(True),
            layer1,
            layer2,
            layer3,
            avgpool,
            Reshape(),
        )

        self.h_size = 64 * self.cfg[0].expansion
        self.classifier = nn.Linear(
            self.h_size, n_classes, bias=bias
        )

    def _make_layer(self, block, planes, stride, num_blocks):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits


class DepthSeparableConv(nn.Module):
    def __init__(
            self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            in_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_planes,
        )
        self.conv2 = nn.Conv2d(
            in_planes, out_planes, kernel_size=1
        )

    def forward(self, xs):
        out = self.conv1(xs)
        out = self.conv2(out)
        return out


class AudioDSCNN(nn.Module):
    def __init__(self, n_channel=172, n_classes=35, bias=False):
        super().__init__()
        self.n_classes = n_classes

        self.encoder = nn.Sequential(
            nn.Conv2d(1, n_channel, kernel_size=(10, 4), stride=(2, 2)),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            DepthSeparableConv(
                n_channel, n_channel, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Reshape(),
        )

        self.h_size = n_channel
        self.classifier = nn.Linear(n_channel, n_classes, bias=bias)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(dim=1)

        out = self.encoder(x)
        logits = self.classifier(out)
        return out, logits


class OhsumedGRUEncoder(nn.Module):

    def __init__(self, n_vocab, w_dim):
        super().__init__()
        self.w_dim = w_dim
        self.w2v = nn.Embedding(n_vocab, w_dim)

        self.gru = nn.GRU(
            input_size=w_dim,
            hidden_size=int(0.5 * w_dim),
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.attn = nn.Sequential(
            nn.Linear(w_dim, w_dim, bias=False),
            nn.Tanh(),
            nn.Linear(w_dim, 1, bias=False),
        )

    def attend(self, xs):
        weights = self.attn(xs)
        weights = weights.squeeze(dim=-1)
        weights = F.softmax(weights, dim=-1)
        attn_xs = torch.bmm(
            weights.unsqueeze(dim=1), xs
        ).squeeze(dim=1)
        return attn_xs

    def forward(self, xs):
        bs = xs.shape[0]
        seq_len = xs.shape[1]
        embeddings = self.w2v(xs)

        self.gru.flatten_parameters()

        h0 = torch.zeros((4, bs, int(0.5 * self.w_dim)))
        h0 = h0.to(xs.device)
        outputs, _ = self.gru(embeddings, h0)
        outputs = outputs.view(bs, seq_len, -1)
        hs = self.attend(outputs)
        return hs

    def load_w2v(self, embeddings):
        if embeddings is not None:
            weights = torch.FloatTensor(embeddings)
            self.w2v = nn.Embedding.from_pretrained(weights)


class OhsumedGRU(nn.Module):

    def __init__(self, n_vocab, w_dim, n_classes, bias=False):
        super().__init__()
        self.encoder = OhsumedGRUEncoder(n_vocab, w_dim)

        self.h_size = w_dim
        self.classifier = nn.Linear(w_dim, n_classes, bias=bias)

    def forward(self, xs):
        hs = self.encoder(xs)
        logits = self.classifier(hs)
        return hs, logits

    def load_w2v(self, embeddings):
        self.encoder.load_w2v(embeddings)


if __name__ == "__main__":
    xs = torch.randn(32, 1, 28, 28)
    net = LeNet(n_classes=10)
    code, logits = net(xs)
    print(code.shape, logits.shape)

    xs = torch.randn(32, 1, 28, 28)
    net = FeMnistNet(n_classes=10)
    code, logits = net(xs)
    print(code.shape, logits.shape)
