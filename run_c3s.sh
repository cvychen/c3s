#!/bin/bash

#����ʵ��1������ͼ��ֱ��ʵıȽ�
#python ./model.py g1 initial    #trans_448
#python ./model.py g1 direct_448
python ./model.py g1 gamma_1 1

#����ʵ��2��gamma�ıȽ�
#python ./model.py g2 gamma_0.5 0.5
#python ./model.py g2 gamma_0.1 0.1

#����ʵ��3��decay�ıȽ�
#python ./model.py g3 non_decay 1
#python ./model.py g3 decay_10-20 1

