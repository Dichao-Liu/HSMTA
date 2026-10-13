from __future__ import print_function
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from my_utils import *
import cv2
import torch
import torchvision
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import bagnets.pytorchnet
from skimage import feature as skif
import torch.nn.functional as F
import random
import shutil
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import timm

from sklearn.metrics import precision_recall_fscore_support as score

import argparse
parser = argparse.ArgumentParser(description='Organize Dataset')
parser.add_argument('--m', type=str, required=True, default='hm', help='model')
parser.add_argument('--fold', type=int,  required=True, default=0)
args, unparsed = parser.parse_known_args()

from hsmta_srr import HSMTA_SRR
from hsmta import HSMTA
from hsmta_reference import ReferenceModel

seed = 0
if args.m == "hm" or args.m == "hms" or args.m == "hmr":
    input_size=228
else:
    input_size = 224
    
seed_everything(seed)
    
def inference(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    if use_cuda:
        device = torch.device("cuda")
    else:
        device = "cpu"
    net.to(device)

    
    if  "mobilenetv4" in args.m:
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    test_set = torchvision.datasets.ImageFolder(root=data_path,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    score_list = []
    target_list = []
    pred_list = []
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)

        score_list.append(outputs.softmax(dim=1).data.cpu())
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        pred_list.append(predicted.data.cpu().unsqueeze(0))
        target_list.append(targets.data.cpu().unsqueeze(0))

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % 50 == 0 or batch_idx == (test_loader.__len__()-1):
            print('Step: %d | Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
            batch_idx, test_loss / (batch_idx + 1), 100. * float(correct) / total, correct, total))

    pred_list = torch.cat(pred_list, axis=-1).squeeze().numpy()
    target_list = torch.cat(target_list, axis=-1).squeeze().numpy()
    score_list = torch.cat(score_list, axis=0).squeeze().numpy()
    
    accuracy = accuracy_score(pred_list, target_list)*100
    f1_micro = f1_score(target_list,pred_list,average='micro')
    f1_macro = f1_score(target_list,pred_list,average='macro')
    
    
    precision_micro, recall_micro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='micro')
    precision_macro, recall_macro, _, _ = score(torch.from_numpy(target_list), torch.from_numpy(pred_list), average='macro')
    
    auc_micro = roc_auc_score(target_list, score_list, multi_class='ovr',average='micro')
    auc_macro = roc_auc_score(target_list, score_list, multi_class='ovr',average='macro')
    
    
    
    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)
    print("Test Accuracy: {}%".format(test_acc))

    return accuracy, f1_micro, f1_macro, auc_micro,auc_macro, precision_micro, precision_macro, recall_micro, recall_macro


      

def test(net, criterion, batch_size, data_path=''):
    net.eval()
    use_cuda = torch.cuda.is_available()
    test_loss = 0
    correct = 0
    total = 0
    idx = 0
    device = torch.device("cuda")

    
    if  "mobilenetv4" in args.m:
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    else:
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    
    test_set = torchvision.datasets.ImageFolder(root=data_path,
                                               transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        idx = batch_idx
        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = Variable(inputs), Variable(targets)
        with torch.no_grad():
            outputs = net(inputs)

            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    test_acc = 100. * float(correct) / total

    test_loss = test_loss / (idx + 1)

    return test_acc, test_loss


def train(nb_epoch, batch_size, num_class, store_name, lr=0.002, data_path='', start_epoch=0, test_folder=''):
    # setup output
    exp_dir = store_name
    try:
        os.stat(exp_dir)
    except:
        os.makedirs(exp_dir)
    
  

    use_cuda = torch.cuda.is_available()
    print(use_cuda)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(input_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_path,'train'), transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    

    if args.m == "hm":
        net = HSMTA(num_class)
    elif args.m == "hms":
        net = HSMTA_SRR(num_class)   
    elif args.m == "hmr":
        net = ReferenceModel(num_class)
    elif args.m == "resnet50":
        net = torchvision.models.resnet50(pretrained=False, num_classes=num_class)
    elif  args.m == "squeezenet1_0":  
        net = torch.hub.load('pytorch/vision:v0.10.0', args.m, pretrained=False)
        net.classifier[1] = nn.Conv2d(512, num_class, kernel_size=1, stride=1)
        net.cuda()
    elif args.m == "shufflenet_v2_x1_0":
        net = torch.hub.load('pytorch/vision:v0.10.0', args.m, pretrained=False).cuda()
        num_ftrs = net.fc.in_features
        net.fc = nn.Linear(num_ftrs, num_class)  
        net.cuda()                          
    else:
        net = timm.create_model(args.m, pretrained=False, num_classes=num_class)
    
    

    netp = torch.nn.DataParallel(net)

    # GPU
    device = torch.device("cuda")
    net.to(device)

    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    max_val_acc = 0

    for epoch in tqdm(range(start_epoch, nb_epoch), ncols=80, unit='s', unit_scale=True):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        idx = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            idx = batch_idx
            if inputs.shape[0] < batch_size:
                continue
            if use_cuda:
                inputs, targets = inputs.to(device), targets.to(device)
            inputs, targets = Variable(inputs), Variable(targets)

            # update learning rater
 
            for nlr in range(len(optimizer.param_groups)):
                optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr)

            optimizer.zero_grad()
            outputs = netp(inputs)
            loss = CELoss(outputs, targets)
            loss.backward()
            optimizer.step()
            

            #  training log
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            train_loss += loss.item() 
            

        train_acc = 100. * float(correct) / total
        train_loss = train_loss / (idx + 1)
        with open(exp_dir + '/results_train.txt', 'a') as file:
            file.write(
                'Iteration %d | train_acc = %.5f | train_loss = %.5f |\n' % (
                epoch, train_acc, train_loss))


        val_acc, val_loss = test(net, CELoss, 4, os.path.join(data_path, test_folder))
        if val_acc > max_val_acc:
            max_val_acc = val_acc
            net.cpu()
            torch.save(net, './' + store_name + '/model.pth')
            net.to(device)
        with open(exp_dir + '/results_test.txt', 'a') as file:
            file.write('Iteration %d, test_acc = %.5f, test_loss = %.6f\n' % (
            epoch, val_acc, val_loss))
            
    trained_model = torch.load('./' + store_name + '/model.pth')
    accuracy, f1_micro, f1_macro, auc_micro, auc_macro, precision_micro, precision_macro, recall_micro, recall_macro = inference(trained_model, CELoss, 3, os.path.join(data_path, 'validation'))
    with open(exp_dir + '/results_test.txt', 'a') as file:
        file.write('Inference Results: Accuracy = %.5f, F1_micro = %.5f, F1_macro = %.5f, Auc_micro = %.5f, Auc_macro = %.5f \n' % (
        accuracy, f1_micro, f1_macro, auc_micro, auc_macro))
        file.write('Inference Results: precision_micro = %.5f, precision_macro = %.5f, recall_micro = %.5f, recall_macro = %.5f \n' % (
        precision_micro, precision_macro, recall_micro, recall_macro))
         
              
 

if __name__ == '__main__':
    seed_everything(seed)
    lr = 0.002
    
    results_path ='Results_DSCC'

    mk_dir(results_path)
    

    task_result_path = os.path.join(results_path, 'classification')
    mk_dir(task_result_path)

    data_path = './datasets/DSCC/splited/seed_0_5fold/{}'.format(args.fold)

    
    pyname = os.path.basename(__file__).replace('.py','').replace('train_models_','')
    experiment_result_path = \
                        os.path.join(task_result_path, "{}_{}_seed_{}_input_size_{}_lr_{}_fold_{}"
                                     .format(pyname,args.m,seed,input_size,lr,args.fold))


    train(nb_epoch=500,             
            batch_size=16,
            num_class=3,
            lr = lr,      
            store_name= experiment_result_path,     
            data_path=data_path,          
            start_epoch=0,
            test_folder='validation')         
