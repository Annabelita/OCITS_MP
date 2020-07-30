import torch
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
import torch.nn as nn

num_classes = 2


class ResNet_2GPU_Sync(ResNet):
    def __init__(self, resnet_type, *args, **kwargs):
        resnet_dict = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
        }

        block_sizes = resnet_dict[resnet_type]
        super(ResNet_2GPU_Sync, self).__init__(
            Bottleneck, block_sizes, num_classes=num_classes, *args, **kwargs
        )

        self.seq1 = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2
        ).to("cuda:1")

        self.seq2 = nn.Sequential(self.layer3, self.layer4, self.avgpool,).to("cuda:2")

        self.fc.to("cuda:2")

    def forward(self, x):
        x = self.seq2(self.seq1(x).to("cuda:2"))
        return self.fc(x.view(x.size(0), -1)).to("cuda:1")


class ResNet_2GPU_Async(ResNet_2GPU_Sync):
    def __init__(self, resnet_type, split_size, *args, **kwargs):
        super(ResNet_2GPU_Async, self).__init__(
            *args, resnet_type=resnet_type, **kwargs
        )
        self.split_size = split_size

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s_next = next(splits)
        s_prev = self.seq1(s_next).to("cuda:2")
        ret = []

        for s_next in splits:
            # A. s_prev runs on cuda:1
            s_prev = self.seq2(s_prev)
            ret.append(self.fc(s_prev.view(s_prev.size(0), -1)).to("cuda:1"))

            # B. s_next runs on cuda:0, which can run concurrently with A
            s_prev = self.seq1(s_next).to("cuda:2")

        s_prev = self.seq2(s_prev)
        ret.append(self.fc(s_prev.view(s_prev.size(0), -1)).to("cuda:1"))

        return torch.cat(ret)


class ResNet_3GPU_Async(ResNet):
    def __init__(self, split_size, *args, **kwargs):
        self.split_size = split_size
        block_sizes = [3, 4, 23, 3]

        super(ResNet_3GPU_Async, self).__init__(
            Bottleneck, block_sizes, num_classes=num_classes, *args, **kwargs
        )

        self.seq1 = nn.Sequential(
            self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2
        ).to("cuda:1")

        layer3_modules = list(self.layer3.children())
        self.seq2 = nn.Sequential(*layer3_modules[: len(layer3_modules) // 2]).to(
            "cuda:2"
        )

        self.seq3 = nn.Sequential(
            nn.Sequential(*layer3_modules[len(layer3_modules) // 2 :]),
            self.layer4,
            self.avgpool,
        ).to("cuda:3")

        self.fc.to("cuda:3")

    def forward(self, x):
        splits = iter(x.split(self.split_size, dim=0))
        s1 = next(splits)
        s2 = self.seq1(s1).to("cuda:2")
        s3 = self.seq2(s2).to("cuda:3")
        ret = []

        for s1 in splits:
            # Run on cuda:3, append result
            res = self.seq3(s3)
            res = self.fc(res.view(res.size(0), -1))
            ret.append(res)

            # Run on cuda:2, move result to cuda:3
            s3 = self.seq2(s2).to("cuda:3")

            # Run on cuda:1, move result to cuda:2
            s2 = self.seq1(s1).to("cuda:2")

        # Finish up last iteration
        s3 = self.seq2(s2).to("cuda:3")
        res = self.seq3(s3)
        res = self.fc(res.view(res.size(0), -1))
        ret.append(res)

        return torch.cat(ret).to("cuda:1")
