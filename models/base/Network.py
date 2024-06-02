import torch
import torch.nn.functional as F
from models.resnet18_encoder import *
from models.resnet20_cifar import *
from models.resnet18_cifar import resnet18_cifar
from utils import identify_importance


class MYNET(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()

        self.mode = mode
        self.args = args
        # self.num_features = 512
        if self.args.dataset in ['cifar100']:
            # self.encoder = resnet20()
            # self.num_features = 64
            self.encoder = resnet18_cifar()
            self.num_features = 512
        if self.args.dataset in ['mini_imagenet']:
            self.encoder = resnet18(False, args)  # pretrained=False
            self.num_features = 512
        if self.args.dataset == 'cub200':
            self.encoder = resnet18(True, args)  # pretrained=True follow TOPIC, models for cub is imagenet pre-trained. https://github.com/xyutao/fscil/issues/11#issuecomment-687548790
            self.num_features = 512
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(self.num_features, self.args.num_classes, bias=False)
        self.fc.is_classifier = True

    def forward_metric(self, x):
        x = self.encode(x)
        if 'cos' in self.mode:
            x = F.linear(F.normalize(x, p=2, dim=-1), F.normalize(self.fc.weight, p=2, dim=-1))
            x = self.args.temperature * x

        elif 'dot' in self.mode:
            x = self.fc(x)

        else:
            base_logits = self.fc(x)[:, :self.args.base_class]
            # new_fc = self.fc.weight[self.args.base_class:, :]
            new_logits = - (torch.unsqueeze(self.fc.weight.t(), dim=0) - torch.unsqueeze(x, dim=-1)).norm(p=2, dim=1)[:,
                         self.args.base_class:]
            x = torch.cat([base_logits, new_logits], dim=-1)

        return x

    def encode(self, x):
        x = self.encoder(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.squeeze(-1).squeeze(-1)
        return x

    def forward(self, input):
        if self.mode != 'encoder':
            input = self.forward_metric(input)
            return input
        elif self.mode == 'encoder':
            input = self.encode(input)
            return input
        else:
            raise ValueError('Unknown mode')

    def update_fc(self,dataloader,class_list,session):
        print("[Session: {}] Update_FC".format(session))
        global data_imgs
        for batch in dataloader:
            data_imgs, label = [_.cuda() for _ in batch]
            data=self.encode(data_imgs).detach()

        if self.args.not_data_init:
            new_fc = nn.Parameter(
                torch.rand(len(class_list), self.num_features, device="cuda"),
                requires_grad=True)
            nn.init.kaiming_uniform_(new_fc, a=math.sqrt(5))
        else:
            new_fc = self.update_fc_avg(data, label, class_list)

        if 'ft' in self.args.new_mode:  # further finetune
            print('further finetune?')
            self.update_fc_ft(new_fc,data_imgs,label,session, class_list)


    def update_fc_avg(self,data,label,class_list):
        new_fc=[]
        for class_index in class_list:
            data_index=(label==class_index).nonzero().squeeze(-1)
            embedding=data[data_index]
            proto=embedding.mean(0)
            new_fc.append(proto)
            self.fc.weight.data[class_index]=proto
        new_fc=torch.stack(new_fc,dim=0)
        return new_fc

    def get_logits(self,x, fc):
        if 'dot' in self.args.new_mode:
            return F.linear(x, fc)

        elif 'cos' in self.args.new_mode:
            return self.args.temperature * F.linear(F.normalize(x, p=2, dim=-1), F.normalize(fc, p=2, dim=-1))

        else:
            base_logits = F.linear(x, fc)[:, :self.args.base_class]
            new_logits = - (torch.unsqueeze(fc.t(), dim=0) - torch.unsqueeze(x, dim=-1)).norm(p=2, dim=1)[:,
                         self.args.base_class:]
            return torch.cat([base_logits, new_logits], dim=-1)


    def update_fc_ft(self, new_fc, data_imgs,label,session, class_list=None):
        self.eval()
        optimizer_embedding = torch.optim.SGD(self.encoder.parameters(), lr=self.args.lr_new, momentum=0.9)

        with torch.enable_grad():
            for epoch in range(self.args.epochs_new):


                fc = self.fc.weight[:self.args.base_class + self.args.way * session, :].detach()
                data = self.encode(data_imgs)
                logits = self.get_logits(data, fc)

                loss = F.cross_entropy(logits, label)
                optimizer_embedding.zero_grad()
                loss.backward()

                optimizer_embedding.step()


        identify_importance(self.args, self, data_imgs.cpu(), batchsize=1,
                                    keep_ratio=self.args.fraction_to_keep, session=session, way=self.args.way,
                                    new_labels=label)



