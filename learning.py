# -*- coding: utf-8 -*-
#author:Chenxuqian
import copy
import torch
import time
import os
# from utils import modelserial
# from utils.utils import progress_bar
from utils import drawPic
from utils import saveTo_csv
from tqdm import tqdm


def train(model, dataloader, criterion, optimizer, scheduler, version=None, dataName=None, epochs=30):

    # get the size of train and evaluation data
    if isinstance(dataloader, dict):
        dataset_sizes = {x: len(dataloader[x].dataset) for x in dataloader.keys()}
        print(dataset_sizes)
    else:
        dataset_sizes = len(dataloader.dataset)

    if not isinstance(criterion, list):
        criterion = [criterion]

    best_model_params = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    start_epoch = 1

    pic_loss = dict()
    pic_loss['test'] = []
    pic_loss['trainval'] = []
    pic_acc = dict()
    pic_acc['trainval'] = []
    pic_acc['test'] = []

    save_csv = './result/' + version + '/'
    if not os.path.exists(save_csv):
        os.makedirs(save_csv)

    since = time.time()
    for epoch in range(start_epoch, epochs+1):
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)
        for phase in ['trainval', 'test']:
            if phase == 'trainval':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_cls_loss = 0.0
            running_c3s_reg_loss = 0.0
           # running_mnl_reg_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            #for inputs, labels in dataloader[phase]:
            for inputs, labels in tqdm(dataloader[phase]):
                #pdb.set_trace()
                inputs = inputs.cuda()
                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'trainval'):
                    # pdb.set_trace()
                    outputs, parts = model(inputs)
            #        log_softmax_out = F.log_softmax(outputs, dim=-1)
                    _, preds = torch.max(outputs, 1)

                    cls_loss = criterion[0](outputs, labels)
                    c3s_reg_loss = criterion[1](parts)
                   # mnl_reg_loss = criterion[2](log_softmax_out)

                    total_loss = cls_loss + c3s_reg_loss

                    # backward + optimize only if in training phase
                    if phase == 'trainval':
                        # pdb.set_trace()
                        total_loss.backward()
                        optimizer.step()
                        

                # statistics
                running_cls_loss += cls_loss.item() * inputs.size(0)
                running_c3s_reg_loss += c3s_reg_loss.item() * inputs.size(0)
               # running_mnl_reg_loss += mnl_reg_loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'trainval':
                scheduler.step()    

            epoch_loss = (running_cls_loss + running_c3s_reg_loss) / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            pic_loss[phase].append(float('%.4f'%epoch_loss))
            pic_acc[phase].append(float('%.4f'%epoch_acc))
           # with open(log_path, "a") as log_file:
            #    log_file.writelines('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_params = copy.deepcopy(model.state_dict())

            #if phase == 'test' and epoch % 5 == 4:
            #    modelserial.saveCheckpoint({'epoch': epoch,
            #                                'best_epoch': best_epoch,
            #                                'state_dict': model.state_dict(),
            #                                'best_state_dict': best_model_params,
            #                                'best_acc': best_acc}, datasetname+'2')
        saveTo_csv(pic_acc, pic_loss, save_csv)
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))
    print('-----------------------------------------------------------------')
    #with open(log_path, "a") as log_file:
    #    log_file.writelines('Training complete in {:.0f}m {:.0f}s\n'.format(
    #            time_elapsed // 60, time_elapsed % 60))
    #    log_file.writelines('Best test Acc: {:4f}\n'.format(best_acc))

   # rsltparams = dict()
    #rsltparams['val_acc'] = best_acc.item()
    #rsltparams['gamma1'] = criterion[1].gamma
    #rsltparams['lr'] = optimizer.param_groups[0]['lr']
    #rsltparams['best_epoch'] = best_epoch

    # load best model weights
    model.load_state_dict(best_model_params)
    return model



def eval(model, dataloader=None):
    model.eval()
    datasize = len(dataloader.dataset)
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs, _ = model(inputs)
            preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == labels.data)
    acc = torch.div(running_corrects.double(), datasize).item()
    print("Test Accuracy: {}".format(acc))

    rsltparams = dict()
    rsltparams['test_acc'] = acc
    return rsltparams


