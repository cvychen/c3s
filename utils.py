# -*- coding: utf-8 -*-
#author:Chenxuqian
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def drawPic(dic=None, start=1, epoch=None, DIR=None, title=None):
    length = epoch
    x = range(1, length + 1)
    # read out the dic_dict
    train = np.array(dic['trainval'])
    valid = np.array(dic['test'])
    if 'acc' in title:
        min_train = np.argmax(train)
        min_valid = np.argmax(valid)
    else:
        min_train = np.argmin(train)
        min_valid = np.argmin(valid)

    plt.figure(figsize=(16, 4))
    plt.title(title)
    plt.plot(x, train, label='train_dic', color='r', marker='o', markersize=0)
    plt.plot(x, valid, label='valid_dic', color='b', marker='o', markersize=0)
    plt.xlim((start, epoch+1))
    # 横坐标描述
    plt.xlabel('epoch')
    plt.plot(min_train + 1, train[min_train], 'rs')
    show_max = '[' + str(min_train + 1) + ' , ' + str(train[min_train]) + ']'
    plt.annotate(show_max, xytext=(min_train + 1, train[min_train]), xy=(min_train + 1, train[min_train]))
    plt.plot(min_valid + 1, valid[min_valid], 'bs')
    show_max2 = '[' + str(min_valid + 1) + ' , ' + str(valid[min_valid]) + ']'
    plt.annotate(show_max2, xytext=(min_valid + 1, valid[min_valid]), xy=(min_valid + 1, valid[min_valid]))
    # 纵坐标描述
    plt.ylabel(title)

    plt.legend()
    img_path = DIR + title
    plt.savefig(img_path, dpi=300)
    # plt.show()
    plt.close()


#保存文本
def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    #import config_file as cfg_file
    import sys
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
 
 
 
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
 
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))

#保存精度到csv
def saveTo_csv(acc_list, loss_list, file='./' ):
	# 列表
    df = pd.DataFrame()
    df['acc_trainval'] = acc_list['trainval']
    df['acc_test'] = acc_list['test']

    df['loss_trainval'] = loss_list['trainval']
    df['loss_test'] = loss_list['test']
    # 保存到本地excel
    df.to_csv(file+'list.csv', index=False)
