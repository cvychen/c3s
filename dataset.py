import numpy as np
import scipy.misc
import os
from PIL import Image
from torchvision import transforms
INPUT_SIZE = 448
import matplotlib.pyplot as plt
import torch
import torchvision



# class CUB():
#     def __init__(self, root, is_train=True, data_len=None):
#         self.root = root
#         self.is_train = is_train
#         img_txt_file = open(os.path.join(self.root, 'images.txt'))
#         label_txt_file = open(os.path.join(self.root, 'image_class_labels.txt'))
#         train_val_file = open(os.path.join(self.root, 'train_test_split.txt'))
#         img_name_list = []
#         for line in img_txt_file:
#             img_name_list.append(line[:-1].split(' ')[-1])
#         label_list = []
#         for line in label_txt_file:
#             label_list.append(int(line[:-1].split(' ')[-1]) - 1)
#         train_test_list = []
#         for line in train_val_file:
#             train_test_list.append(int(line[:-1].split(' ')[-1]))
#         train_file_list = [x for i, x in zip(train_test_list, img_name_list) if i]
#         test_file_list = [x for i, x in zip(train_test_list, img_name_list) if not i]
#         if self.is_train:
#             self.train_img = [scipy.misc.imread(os.path.join(self.root, 'images', train_file)) for train_file in
#                               train_file_list[:data_len]]
#             self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
#         if not self.is_train:
#             self.test_img = [scipy.misc.imread(os.path.join(self.root, 'images', test_file)) for test_file in
#                              test_file_list[:data_len]]
#             self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]
#
#     def __getitem__(self, index):
#         if self.is_train:
#             img, target = self.train_img[index], self.train_label[index]
#             lx = 0
#             if len(img.shape) == 2:
#                 lx = img
#                 img = np.stack([img] * 3, 2)
#
#             img = Image.fromarray(img, mode='RGB')
#             img = transforms.Resize((600, 600), Image.BILINEAR)(img)
#             img = transforms.RandomCrop(INPUT_SIZE)(img)
#             img = transforms.RandomHorizontalFlip()(img)
#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#
#         else:
#             img, target = self.test_img[index], self.test_label[index]
#             if len(img.shape) == 2:
#                 img = np.stack([img] * 3, 2)
#             img = Image.fromarray(img, mode='RGB')
#             img = transforms.Resize((600, 600), Image.BILINEAR)(img)
#             img = transforms.CenterCrop(INPUT_SIZE)(img)
#             img = transforms.ToTensor()(img)
#             img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
#
#         return img, target, lx
#
#     def __len__(self):
#         if self.is_train:
#             return len(self.train_label)
#         else:
#             return len(self.test_label)


class Vegfru():
    def __init__(self, root, is_train=True, data_len=None):
        self.root = root
        self.is_train = is_train

        img_txt_file_train = open(os.path.join(self.root, 'vegfru_list/vegfru_train.txt'))
        img_txt_file_test = open(os.path.join(self.root, 'vegfru_list/vegfru_test.txt'))

        self.train_label = []
        self.test_label = []
        train_file_list = []
        test_file_list = []

        for line in img_txt_file_train:
            self.train_label.append(int(line[:-1].split(' ')[-1]))
            train_file_list.append((line[:-1].split(' ')[0]))

        for line in img_txt_file_test:
            self.test_label.append(int(line[:-1].split(' ')[-1]))
            test_file_list.append((line[:-1].split(' ')[0]))

        self.train_img = train_file_list
        self.test_img = test_file_list
        # if self.is_train:
        #     self.train_img = [scipy.misc.imread(os.path.join(self.root, train_file)) for train_file in
        #                       train_file_list[:data_len]]
        #     # self.train_label = [x for i, x in zip(train_test_list, label_list) if i][:data_len]
        # if not self.is_train:
        #     self.test_img = [scipy.misc.imread(os.path.join(self.root, test_file)) for test_file in
        #                      test_file_list[:data_len]]
            # self.test_label = [x for i, x in zip(train_test_list, label_list) if not i][:data_len]

    def __getitem__(self, index):
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            lx = 0
            # if len(img.shape) == 2:
            #     lx = img
            #     img = np.stack([img] * 3, 2)
            img = Image.open(self.root + img).convert('RGB')

            # img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        else:
            img, target = self.test_img[index], self.test_label[index]
            img = Image.open(self.root + img).convert('RGB')
            # if len(img.shape) == 2:
            #     img = np.stack([img] * 3, 2)
            # img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.CenterCrop(INPUT_SIZE)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

        return img, target

    def __len__(self):
        if self.is_train:
            return len(self.train_label)
        else:
            return len(self.test_label)


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(2)  # pause a bit so that plots are updated


if __name__ == '__main__':
    # dataset = Vegfru(root='/home/yelin/python/VegFru')

    dataloader = dict()
    trainset = Vegfru(root='/home/yelin/python/VegFru/', is_train=True, data_len=None)
    dataloader['trainval'] = torch.utils.data.DataLoader(trainset, batch_size=16,
                                                         shuffle=True, num_workers=16, drop_last=False)
    # testset = dataset.Vegfru(root='/home/yelin/python/VegFru', is_train=False, data_len=None)
    # dataloader['test'] = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
    #                                                  shuffle=False, num_workers=16, drop_last=False)
    inputs, classes = next(iter(dataloader['trainval']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out)
    # print(label[0])
    # print(dataset[2])
    # print(len(dataset.train_label))
    # for data in dataset:
    #     print(data[0].size(), data[1])
    # dataset = CUB(root='/home/yelin/python/CUB_200_2011', is_train=False)
    # print(len(dataset.test_img))
    # print(len(dataset.test_label))
    # for data in dataset:
    #     print(data[0].size(), data[1])
