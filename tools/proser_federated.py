import argparse
import os
import _init_paths
import time
import os.path as osp
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import roc_auc_score
from models import *
from models.digit import DigitModel
from utils import ensure_path, get_root_logger, pprint
import copy
import random
from data import data_relabel

def seed_everything(seed: int):  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def prepare_data(options):
    # Prepare data
    transform_mnist = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_svhn = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_usps = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_synth = transforms.Compose([
            transforms.Resize([28,28]),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    transform_mnistm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # MNIST
    mnist_trainset     = data_relabel.DigitsDataset(data_path="/data/cyang/Data/Digits/MNIST", channels=1, percent=options['percent'], setname='train',  transform=transform_mnist, seed=options['seed'], known_class=options['known_class'])
    mnist_testset_close      = data_relabel.DigitsDataset(data_path="/data/cyang/Data/Digits/MNIST", channels=1, percent=options['percent'], setname='testclose', transform=transform_mnist, seed=options['seed'], known_class=options['known_class'])
    mnist_testset_open      = data_relabel.DigitsDataset(data_path="/data/cyang/Data/Digits/MNIST", channels=1, percent=options['percent'], setname='testopen', transform=transform_mnist, seed=options['seed'], known_class=options['known_class'])
    
    # SVHN
    svhn_trainset      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SVHN', channels=3, percent=options['percent'],  setname='train',  transform=transform_svhn, seed=options['seed'], known_class=options['known_class'])
    svhn_testset_close       = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SVHN', channels=3, percent=options['percent'],  setname='testclose', transform=transform_svhn, seed=options['seed'], known_class=options['known_class'])
    svhn_testset_open       = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SVHN', channels=3, percent=options['percent'],  setname='testopen', transform=transform_svhn, seed=options['seed'], known_class=options['known_class'])    
    
    # USPS
    usps_trainset      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/USPS', channels=1, percent=options['percent'],  setname='train',  transform=transform_usps, seed=options['seed'], known_class=options['known_class'])
    usps_testset_close       = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/USPS', channels=1, percent=options['percent'],  setname='testclose', transform=transform_usps, seed=options['seed'], known_class=options['known_class'])
    usps_testset_open       = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/USPS', channels=1, percent=options['percent'],  setname='testopen', transform=transform_usps, seed=options['seed'], known_class=options['known_class'])
    
    # Synth Digits
    synth_trainset     = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SynthDigits/', channels=3, percent=options['percent'],  setname='train',  transform=transform_synth, seed=options['seed'], known_class=options['known_class'])
    synth_testset_close      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SynthDigits/', channels=3, percent=options['percent'],  setname='testclose', transform=transform_synth, seed=options['seed'], known_class=options['known_class'])
    synth_testset_open      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/SynthDigits/', channels=3, percent=options['percent'],  setname='testopen', transform=transform_synth, seed=options['seed'], known_class=options['known_class'])

    # MNIST-M
    mnistm_trainset     = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/MNIST_M/', channels=3, percent=options['percent'],  setname='train',  transform=transform_mnistm, seed=options['seed'], known_class=options['known_class'])
    mnistm_testset_close      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/MNIST_M/', channels=3, percent=options['percent'],  setname='testclose', transform=transform_mnistm, seed=options['seed'], known_class=options['known_class'])
    mnistm_testset_open      = data_relabel.DigitsDataset(data_path='/data/cyang/Data/Digits/MNIST_M/', channels=3, percent=options['percent'],  setname='testopen', transform=transform_mnistm, seed=options['seed'], known_class=options['known_class'])

    print('Known class list: {}'.format(mnist_trainset.known_class_list))
    print('Unknown class list: {}'.format(mnist_trainset.unknown_class_list))


    mnist_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=options['batch_size'], shuffle=True)
    mnist_loader_close = torch.utils.data.DataLoader(mnist_testset_close, batch_size=options['batch_size'], shuffle=False)
    mnist_loader_open = torch.utils.data.DataLoader(mnist_testset_open, batch_size=options['batch_size'], shuffle=False)
    
    svhn_loader = torch.utils.data.DataLoader(svhn_trainset, batch_size=options['batch_size'], shuffle=True)
    svhn_loader_close = torch.utils.data.DataLoader(svhn_testset_close, batch_size=options['batch_size'], shuffle=False)
    svhn_loader_open = torch.utils.data.DataLoader(svhn_testset_open, batch_size=options['batch_size'], shuffle=False)    
    
    usps_loader = torch.utils.data.DataLoader(usps_trainset, batch_size=options['batch_size'], shuffle=True)
    usps_loader_close = torch.utils.data.DataLoader(usps_testset_close, batch_size=options['batch_size'], shuffle=False)
    usps_loader_open = torch.utils.data.DataLoader(usps_testset_open, batch_size=options['batch_size'], shuffle=False)    

    synth_loader = torch.utils.data.DataLoader(synth_trainset, batch_size=options['batch_size'], shuffle=True)
    synth_loader_close = torch.utils.data.DataLoader(synth_testset_close, batch_size=options['batch_size'], shuffle=False)
    synth_loader_open = torch.utils.data.DataLoader(synth_testset_open, batch_size=options['batch_size'], shuffle=False)

    mnistm_loader = torch.utils.data.DataLoader(mnistm_trainset, batch_size=options['batch_size'], shuffle=True)
    mnistm_loader_close = torch.utils.data.DataLoader(mnistm_testset_close, batch_size=options['batch_size'], shuffle=False)
    mnistm_loader_open = torch.utils.data.DataLoader(mnistm_testset_open, batch_size=options['batch_size'], shuffle=False)

    testset_close = [mnist_testset_close, svhn_testset_close, usps_testset_close, synth_testset_close, mnistm_testset_close]
    testset_open = [mnist_testset_open, svhn_testset_open, usps_testset_open, synth_testset_open, mnist_testset_open]

    train_loaders = [mnist_loader, svhn_loader, usps_loader, synth_loader, mnistm_loader]
    test_loaders_close = [mnist_loader_close, svhn_loader_close, usps_loader_close, synth_loader_close, mnistm_loader_close]
    test_loaders_open = [mnist_loader_open, svhn_loader_open, usps_loader_open, synth_loader_open, mnistm_loader_open]

    return train_loaders, test_loaders_close, test_loaders_open, testset_close, testset_open, [mnist_trainset.known_class_list, mnist_trainset.unknown_class_list]

################# Key Function ########################
def communication(args, server_model, models, client_weights):
    with torch.no_grad():
        # aggregate params
        if args.mode.lower() == 'fedbn':
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(len(models)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(models)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
        else:
            for key in server_model.state_dict().keys():
                # num_batches_tracked is a non trainable LongTensor and
                # num_batches_tracked are the same for all clients for the given datasets
                if 'num_batches_tracked' in key:
                    server_model.state_dict()[key].data.copy_(models[0].state_dict()[key])
                else:
                    temp = torch.zeros_like(server_model.state_dict()[key])
                    for client_idx in range(len(client_weights)):
                        temp += client_weights[client_idx] * models[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(len(client_weights)):
                        models[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])
    return server_model, models

def train(args, trainloader, net, optimizer, criterion):
    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # if args.shmode==False:
        #     progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1), 100.*correct/total

def traindummy(args, trainloader, net, optimizer, criterion):

    # print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    alpha=args.alpha
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()

        totallenth=len(inputs)
        halflenth=int(len(inputs)/2)
        beta=torch.distributions.beta.Beta(alpha, alpha).sample([]).item()
        
        prehalfinputs=inputs[:halflenth]
        prehalflabels=targets[:halflenth]
        laterhalfinputs=inputs[halflenth:]
        laterhalflabels=targets[halflenth:]

        index = torch.randperm(prehalfinputs.size(0)).cuda()
        pre2embeddings=pre2block(net,prehalfinputs)
        mixed_embeddings = beta * pre2embeddings + (1 - beta) * pre2embeddings[index]

        dummylogit=dummypredict(net,laterhalfinputs)
        lateroutputs=net(laterhalfinputs)
        latterhalfoutput=torch.cat((lateroutputs,dummylogit),1)
        prehalfoutput=torch.cat((latter2blockclf1(net,mixed_embeddings),latter2blockclf2(net,mixed_embeddings)),1)
        
        maxdummy,_=torch.max(dummylogit.clone(),dim=1)
        maxdummy=maxdummy.view(-1,1)
        dummpyoutputs=torch.cat((lateroutputs.clone(),maxdummy),dim=1)
        for i in range(len(dummpyoutputs)):
            nowlabel=laterhalflabels[i]
            dummpyoutputs[i][nowlabel]=-1e9
        dummytargets=torch.ones_like(laterhalflabels)*args.known_class
        

        outputs=torch.cat((prehalfoutput,latterhalfoutput),0)
        loss1= criterion(prehalfoutput, (torch.ones_like(prehalflabels)*args.known_class).long().cuda()) 
        loss2=criterion(latterhalfoutput,laterhalflabels )
        loss3= criterion(dummpyoutputs, dummytargets)
        loss=0.01*loss1+args.lamda1*loss2+args.lamda2*loss3
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # if args.shmode==False:
        #     progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) L1 %.3f, L2 %.3f'\
        #     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total , loss1.item(), loss2.item(), ))
    
    # if args.log:
    #     logger.info('==> Epoch {}  Loss: {:.3f}\t | Acc: {:.3f} ({}/{}) L1 {:.3f}, L2 {:.3f} \n'
    #     .format(epoch, train_loss/(batch_idx+1), 100.*correct/total, correct, total , loss1.item(), loss2.item()))

    return train_loss/(batch_idx+1), 100.*correct/total

def valdummy(net, closeset, openset, closeloader, openloader):
    net.eval()
    CONF_AUC=False
    CONF_DeltaP=True
    auclist1=[]
    auclist2=[]
    correct = 0
    total = 0
    linspace=[0]
    closelogits=torch.zeros((len(closeset),args.known_class+1)).cuda()
    openlogits=torch.zeros((len(openset),args.known_class+1)).cuda()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(closeloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batchnum=len(targets)
            logits=net(inputs)
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            closelogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
        acc = 100. * correct / total
        for batch_idx, (inputs, targets) in enumerate(openloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            batchnum=len(targets)
            logits=net(inputs)
            dummylogit=dummypredict(net,inputs)
            maxdummylogit,_=torch.max(dummylogit,1)
            maxdummylogit=maxdummylogit.view(-1,1)
            totallogits=torch.cat((logits,maxdummylogit),dim=1)
            openlogits[batch_idx*batchnum:batch_idx*batchnum+batchnum,:]=totallogits
    Logitsbatchsize=200
    maxauc=0
    maxaucbias=0
    for biasitem in linspace:
        if CONF_AUC:
            for temperature in [1024.0]:
                closeconf=[]
                openconf=[]
                closeiter=int(len(closelogits)/Logitsbatchsize)
                openiter=int(len(openlogits)/Logitsbatchsize)
                for batch_idx  in range(closeiter):
                    logitbatch=closelogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    closeconf.append(conf.cpu().numpy())
                closeconf=np.reshape(np.array(closeconf),(-1))
                closelabel=np.ones_like(closeconf)
                for batch_idx  in range(openiter):
                    logitbatch=openlogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    conf=embeddings[:,-1]
                    openconf.append(conf.cpu().numpy())
                openconf=np.reshape(np.array(openconf),(-1))
                openlabel=np.zeros_like(openconf)
                totalbinary=np.hstack([closelabel,openlabel])
                totalconf=np.hstack([closeconf,openconf])
                auc1=roc_auc_score(1-totalbinary,totalconf)
                auc2=roc_auc_score(totalbinary,totalconf)
                # if args.log:
                #     logger.info('Temperature:',temperature,'bias',biasitem,'AUC_by_confidence',auc2)
                # print('Temperature:',temperature,'bias',biasitem,'AUC_by_confidence',auc2)
                auclist1.append(np.max([auc1,auc2]))
        if CONF_DeltaP:
            for temperature in [1024.0]:
                closeconf=[]
                openconf=[]
                closeiter=int(len(closelogits)/Logitsbatchsize)
                openiter=int(len(openlogits)/Logitsbatchsize)
                for batch_idx  in range(closeiter):
                    logitbatch=closelogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    dummyconf=embeddings[:,-1].view(-1,1)
                    maxknownconf,_=torch.max(embeddings[:,:-1],dim=1)
                    maxknownconf=maxknownconf.view(-1,1)
                    conf=dummyconf-maxknownconf
                    closeconf.append(conf.cpu().numpy())
                closeconf=np.reshape(np.array(closeconf),(-1))
                closelabel=np.ones_like(closeconf)
                for batch_idx  in range(openiter):
                    logitbatch=openlogits[batch_idx*Logitsbatchsize:batch_idx*Logitsbatchsize+Logitsbatchsize,:]
                    logitbatch[:,-1]=logitbatch[:,-1]+biasitem
                    embeddings=nn.functional.softmax(logitbatch/temperature,dim=1)
                    dummyconf=embeddings[:,-1].view(-1,1)
                    maxknownconf,_=torch.max(embeddings[:,:-1],dim=1)
                    maxknownconf=maxknownconf.view(-1,1)
                    conf=dummyconf-maxknownconf
                    openconf.append(conf.cpu().numpy())
                openconf=np.reshape(np.array(openconf),(-1))
                openlabel=np.zeros_like(openconf)
                totalbinary=np.hstack([closelabel,openlabel])
                totalconf=np.hstack([closeconf,openconf])
                auc1=roc_auc_score(1-totalbinary,totalconf)
                auc2=roc_auc_score(totalbinary,totalconf)
                # print('Temperature:',temperature,'bias',biasitem,'AUC_by_Delta_confidence',auc1)
                auclist1.append(np.max([auc1,auc2]))
    return acc, 100. * np.max(np.array(auclist1))


def dummypredict(net,x):
    if  args.backbone=="WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        out = net.layer3(out)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out
    elif args.backbone == 'DigitModel':
        out = F.relu(net.bn1(net.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(net.bn2(net.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = F.relu(net.bn3(net.conv3(out)))
        out = out.view(out.shape[0], -1)
        out = net.fc1(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn4(out)
        out = F.relu(out)
        out = net.fc2(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn5(out)
        out = F.relu(out)
        out = net.clf2(out)
        return out



def pre2block(net,x):
    if  args.backbone=="WideResnet":
        out = net.conv1(x)
        out = net.layer1(out)
        out = net.layer2(out)
        return out
    elif args.backbone=='DigitModel':
        out = F.relu(net.bn1(net.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = F.relu(net.bn2(net.conv2(out)))
        out = F.max_pool2d(out, 2)
        return out

def latter2blockclf1(net,x):
    if  args.backbone=="WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.linear(out)
        return out
    elif args.backbone == 'DigitModel':
        out = F.relu(net.bn3(net.conv3(x)))
        out = out.view(out.shape[0], -1)
        out = net.fc1(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn4(out)
        out = F.relu(out)
        out = net.fc2(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn5(out)
        out = F.relu(out)
        out = net.fc3(out)
        return out

def latter2blockclf2(net,x):
    if args.backbone=="WideResnet":
        out = net.layer3(x)
        out = F.relu(net.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = net.clf2(out)
        return out
    elif args.backbone == 'DigitModel':
        out = F.relu(net.bn3(net.conv3(x)))
        out = out.view(out.shape[0], -1)
        out = net.fc1(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn4(out)
        out = F.relu(out)
        out = net.fc2(out)
        if out.size()[0] ==1:
            out = out
        else:
            out = net.bn5(out)
        out = F.relu(out)
        out = net.clf2(out)
        return out

def getmodel(args):
    print('==> Building model..')
    if args.backbone=='WideResnet':
        net=Wide_ResNet(28, 10, args.known_class)
    elif args.backbone == 'DigitModel':
        net = DigitModel(args.known_class)
    net=net.cuda()
    return net


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--lr_open', default=0.1, type=float, help='learning rate')
    parser.add_argument('--com_iter', default=1,type=int,help='communication iters')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--model_type', default='Proser', type=str, help='Recognition Method')
    parser.add_argument('--backbone', default='DigitModel', type=str, help='Backbone type.')
    parser.add_argument('--dataset', default='cifar10_relabel',type=str,help='dataset configuration')
    parser.add_argument('--percent', type = float, default= 0.1, help ='percentage of dataset to train')
    parser.add_argument('--mode', default='fedavg',type=str,help='federated mode')
    parser.add_argument('--known_class', default=6,type=int,help='number of known class')
    parser.add_argument('--seed', default='66',type=int,help='random seed for dataset generation.')
    parser.add_argument('--batch_size', default='32',type=int,help='batch size for dataloader.')
    parser.add_argument('--lamda1', default='1',type=float,help='trade-off between loss')
    parser.add_argument('--lamda2', default='1',type=float,help='trade-off between loss')
    parser.add_argument('--alpha', default='1',type=float,help='alpha value for beta distribution')
    parser.add_argument('--dummynumber', default=1,type=int,help='number of dummy label.')
    parser.add_argument('--shmode',action='store_true')
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--pretrain_epoch', default=50, type=int, help='resnet pretrain stage')
    parser.add_argument('--max_epoch', default=100, type=int, help='total epoches')

    args = parser.parse_args()
    options = vars(args)

    pprint(vars(args))
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    seed_everything(args.seed)

    best_acc = 0  
    start_epoch = 0  
    
    print('==> Preparing data..')

    Digit_data = prepare_data(options)
    knownlist, unknownlist = Digit_data[5][0], Digit_data[5][1]

    trainloaders, closeloader, openloader, closeset, openset = Digit_data[0], Digit_data[1], Digit_data[2], Digit_data[3], Digit_data[4]

    datasets = ['MNIST', 'SVHN', 'USPS', 'Synth', 'MNIST-M']
    client_num = len(datasets)
    client_weights = [1/client_num for i in range(client_num)]


    log = args.log
    if log:
        log_path = os.path.join('/data/cyang/Code/Experiment/fedpd_exp', args.model_type + '_' + args.dataset)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
            os.makedirs(os.path.join(log_path, 'models'))
        logfile = os.path.join(log_path,'{}_{}_{}_{}_{}.log'.format(args.mode, str(args.pretrain_epoch), str(args.com_iter), str(args.lr_open), str(args.known_class)))
        logger = get_root_logger(logfile)
        logger.info('==={}==='.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logger.info('===Setting===')
        logger.info('method: {}'.format(args.model_type))    
        logger.info('dataset: {}'.format(args.dataset))
        logger.info('backbone: {}'.format(args.backbone))
        logger.info('com_iter: {}'.format(str(args.com_iter)))
        logger.info('lr: {}'.format(str(args.lr)))
        logger.info('pretrain epoch: {}'.format(str(args.pretrain_epoch)))
        logger.info('known class: {}'.format(knownlist))
        logger.info('unknown class: {}'.format(unknownlist))
    
    save_path1 = osp.join('/data/cyang/Code/Experiment/fedpd_exp','D{}-M{}-B{}'.format(args.dataset,args.model_type, args.backbone,))
    model_path = osp.join('/data/cyang/Code/Experiment/fedpd_exp','D{}-M{}-B{}'.format(args.dataset,'softmax', args.backbone,))
    save_path2 = 'LR{}-K{}-U{}-Seed{}'.format(str(args.lr), knownlist,unknownlist,str(args.seed))
    args.save_path = osp.join(save_path1, save_path2)
    ensure_path(save_path1, remove=False)
    ensure_path(args.save_path, remove=False)

    server_model = getmodel(args)
    server_model.clf2=nn.Linear(512,args.dummynumber)
    server_model=server_model.cuda()

    models = [copy.deepcopy(server_model).cuda() for idx in range(client_num)]

    criterion = nn.CrossEntropyLoss()
    optimizers = [optim.SGD(models[idx].parameters(), lr=args.lr,momentum=0.9, weight_decay=5e-4) for idx in range(client_num)]
    globalacc=0
    for i in range(args.max_epoch):
        if args.log:
            if i == 0:
                logger.info('Begin close set training')
            elif i == args.pretrain_epoch:
                logger.info('Begin open set training')
            logger.info("==> Epoch {}/{}".format(i, args.max_epoch))
        for client_idx in range(client_num):
            if i < args.pretrain_epoch:
                train_loss, _ = train(args, trainloaders[client_idx], models[client_idx], optimizers[client_idx], criterion)
            else:
                optimizers = [optim.SGD(models[idx].parameters(), lr=args.lr * args.lr_open, momentum=0.9, weight_decay=5e-4) for idx in range(client_num)]
                train_loss, _ = traindummy(args, trainloaders[client_idx], models[client_idx], optimizers[client_idx], criterion)
            acc, auc = valdummy(models[client_idx], closeset[client_idx], openset[client_idx], closeloader[client_idx], openloader[client_idx])    
            if args.log:
                logger.info('{:<7s}  Loss: {:.3f} | Acc: {:.3f} | Auc: {:.3f}'
                .format(datasets[client_idx], train_loss, acc, auc))
        if (i+1) % args.com_iter == 0:
            server_model, models = communication(args, server_model, models, client_weights)
            for client_idx in range(client_num):
                acc, auc = valdummy(models[client_idx], closeset[client_idx], openset[client_idx], closeloader[client_idx], openloader[client_idx])
                if args.log:
                    logger.info('{:<7s} | Acc: {:.3f} | Auc: {:.3f}'
                    .format(datasets[client_idx], acc, auc))