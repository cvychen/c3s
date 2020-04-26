# -*- coding: utf-8 -*-
#author:Chenxuqian
####################################
# This script is with regularization loss
####################################

import os, sys
# import pickle as pk
# import pdb

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as opt
from torch.optim import lr_scheduler
import torch.utils.model_zoo as model_zoo
from datetime import datetime
from utils import make_print_to_file
# from utils import imdb, myimagefolder, mydataloader
# progpath = os.path.dirname(os.path.realpath(__file__))          # /home/luowei/Codes/feasc-msc
# sys.path.append(progpath)
import learning
import mymodels
# from utils import Logger, create_txt

##################### Dataset path

# datasetname = 'idesigner'
# data = datasetname + 'kaggle-designer'

# rsltparams = dict()

# save_log = os.path.join("./results/log", "osme+c3s-" + datasetname + "-" + datetime.now().strftime('%Y%m%d_%H%M%S'))
# save_pkl = os.path.join("./results/pkl", "osme+c3s-" + datasetname + "-" + datetime.now().strftime('%Y%m%d_%H%M%S'))
# if os.path.exists(save_pkl):
#     raise NameError('model dir exists!')
# os.makedirs(save_pkl)
# if os.path.exists(save_log):
#     raise NameError('model dir exists!')
# os.makedirs(save_log)
# logging = init_log(save_dir)
# logging.info
# log = create_txt(save_log)
# sys.stdout = Logger(log)

#手动设置参数

ver = str(sys.argv[1])
val = str(sys.argv[2])
#nparts = int(sys.argv[2])
gamma1 = float(sys.argv[3])
nparts = 2
#gamma1 = 1

datasetname = "cubbirds"
#datetime.now().strftime('%m%d_%H%M%S')
version =  ver + '-' + val
datasets_path = os.path.expanduser("~/")
datasetpath = os.path.join(datasets_path, datasetname)

print_txt = './result/'+ version +'/'
if not os.path.exists(print_txt):
        os.makedirs(print_txt)

make_print_to_file(print_txt)


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

print("group:{}:, val:{}".format(ver, val))

# organizing data
# assert imdb.creatDataset(datasetpath, datasetname='cubbirds') == True, "Failing to creat train/val/test sets"
data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize((300, 300)),
         transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}

"""
if val == "initial":
    #print("val:{}".format(val))
    data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((600, 600)),
        transforms.RandomCrop((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize((600, 600)),
         transforms.CenterCrop((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}
elif val == "direct_448":
    #print("val:{}".format(val))
    data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}
else:
    #print("val:{}".format(val))
    data_transform = {
    'trainval': transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
            transforms.Resize((300, 300)),
         transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
}
  """
 

  

# using ground truth data
datasplits = {x: datasets.ImageFolder(os.path.join(datasetpath, x), data_transform[x])
              for x in ['trainval', 'test']}

batchsize = 32
dataloader = {x: torch.utils.data.DataLoader(datasplits[x], batch_size=batchsize, shuffle=True, num_workers=32)
              for x in ['trainval', 'test']}

datasplit_sizes = {x: len(datasplits[x]) for x in ['trainval', 'test']}
class_names = datasplits['trainval'].classes
num_classes = len(class_names)

################################### constructing or loading model
# nparts = 2          # number of parts you want to use for your dataset
seflag = True      # True to use the SENet, False for the ResNet
model = mymodels.feasc50(num_classes=num_classes, nparts=nparts, seflag=seflag)

if torch.cuda.device_count() > 0:
    model = nn.DataParallel(model)
    model = model.cuda()

# load pretrained SENet weights
state_dict_path = "pretrained-weights.pkl"
state_params = torch.load(state_dict_path)
state_params['weight'].popitem('module.fc.weight')
state_params['weight'].popitem('module.fc.bias')
model.load_state_dict(state_params['weight'], strict=False)

# creating loss functions
# gamma1 = 1
cls_loss = nn.CrossEntropyLoss()
reg_loss = mymodels.RegularLoss(gamma=gamma1, nparts=nparts)
criterion = [cls_loss, reg_loss]

# creating optimizer
lr = 0.01
optmeth = 'sgd'
optimizer = opt.SGD(model.parameters(), lr=lr, momentum=0.9)

# creating optimization scheduler
# scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
if val== "decay_10-20":
    print("type:{}".format(val))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[11, 20], gamma=0.1)
elif val== "non_decay":
    print("type:{}".format(val))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[31, 40], gamma=0.1)
else:
    print("type:{}".format(val))
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[16, 30], gamma=0.1)

# training the model
epochs = 30
isckpt = False  # if you want to load model params from checkpoint, set it to True
# print parameters
print("{}: {}, gamma: {}, nparts: {}, epochs: {}".format(optmeth, lr, gamma1, nparts, epochs))
model = learning.train(model, dataloader, criterion, optimizer, scheduler, version=version,
                                                         epochs=epochs, dataName=datasetname)

# print("测试集精度-----------------------")
# rsltparams = modellearning2.eval(model=model, dataloader=dataloader['test'])

#### save model
# modelpath = './models'
#modelname = "{}_parts{}-sc{}-{}--SENet50-full-rglz.model".format(datasetname, nparts, gamma1, lr)
#torch.save(model.state_dict(), save_pkl + modelname)
