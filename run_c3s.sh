#!/bin/bash

#层析实验1：对于图像分辨率的比较
#python ./model.py g1 initial    #trans_448
#python ./model.py g1 direct_448
python ./model.py g1 gamma_1 1

#层析实验2：gamma的比较
#python ./model.py g2 gamma_0.5 0.5
#python ./model.py g2 gamma_0.1 0.1

#层析实验3：decay的比较
#python ./model.py g3 non_decay 1
#python ./model.py g3 decay_10-20 1

