import torch
import torch.nn as nn 

from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from torchvision.datasets import CIFAR10
from torchvision import transforms
import argparse
import os, sys

import utils
import numpy as np
import random
from utils import Visualizer

from torch.utils import data

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from SyncBN import SyncBatchNorm2d
from SyncBN.utils import convert_sync_batchnorm
from  tqdm import tqdm

def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./data', 
                        help="path to Dataset")
    
    # Train Options
    parser.add_argument("--model", type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'])
    parser.add_argument("--epochs", type=int, default=40,
                        help="epoch number (default: 40)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="learning rate (default: 0.1)")
    
    parser.add_argument("--batch_size", type=int, default=128,
                        help='batch size (default: 128)')
    parser.add_argument("--lr_decay_step", type=int, default=150,
                        help='batch size (default: 150)')
    parser.add_argument("--ckpt", default=None, type=str,
                        help="path to trained model. Leave it None if you want to retrain your model")
    parser.add_argument("--gpu_id", type=str, default='0', 
                        help="GPU ID")
    
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='momentum for SGD (default: 0.9)')
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help='weight decay (default: 5e-4)')

    parser.add_argument("--num_workers", type=int, default=4,
                        help='number of workers (default: 4)')
    parser.add_argument("--val_on_trainset", action='store_true', default=False ,
                        help="enable validation on train set (default: False)")
    parser.add_argument("--random_seed", type=int, default=23333,
                        help="random seed (default: 23333)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=1,
                        help="epoch interval for eval (default: 1)")
    parser.add_argument("--ckpt_interval", type=int, default=1,
                        help="saving interval (default: 1)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    parser.add_argument("--sync_bn", action='store_true', default=False,
                        help="sync batchnorm")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='15555',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--trace_name", type=str, default=None)
    return parser



def train( cur_epoch, criterion, model, optim, train_loader, device, scheduler=None, print_interval=10, vis=None, trace_name=None):
    """Train and return epoch loss"""
    
    if scheduler is not None:
        scheduler.step()
    print("Epoch %d, lr = %f"%(cur_epoch, optim.param_groups[0]['lr']))
    epoch_loss = 0.0
    interval_loss = 0.0

    for cur_step, (images, labels) in enumerate( train_loader ):
        
        images = images.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.long)

        # N, C, H, W
        optim.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optim.step()

        np_loss = loss.detach().cpu().numpy()
        epoch_loss+=np_loss
        interval_loss+=np_loss

        if (cur_step+1)%print_interval==0:
            interval_loss = interval_loss/print_interval
            print("Epoch %d, Batch %d/%d, Loss=%f"%(cur_epoch, cur_step+1, len(train_loader), interval_loss))
            if vis is not None:
                x = cur_epoch*len(train_loader) + cur_step + 1
                vis.vis_scalar('Loss', trace_name, x, interval_loss )
            interval_loss=0.0
    return epoch_loss / len(train_loader)


def validate( model, loader, device, metrics):
    """Do validation and return specified samples"""
    metrics.reset()
    with torch.no_grad():
        for i, (images, labels) in  enumerate( tqdm( loader ) ):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)
        score = metrics.get_results()
    return score


def main():
    opts = get_argparser().parse_args()
    # Set up visualization
    vis = Visualizer(port=opts.vis_port, env=opts.vis_env)
    if vis is not None: # display options
        vis.vis_table( "%s opts"%opts.trace_name, vars(opts) )
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
    print("Device: %s"%device)

    # Set up random seed
    torch.manual_seed(opts.random_seed)
    torch.cuda.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Set up dataloader
    train_dst = CIFAR10(root='./data', train=True, 
                        transform=transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225] )]),
                        download=opts.download )
    
    val_dst = CIFAR10(root='./data', train=False, 
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize( mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225] )]),
                        download=False )
    
    train_loader = data.DataLoader(train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)
    val_loader = data.DataLoader(val_dst, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers)
    print("Dataset: CIFAR10, Train set: %d, Val set: %d"%(len(train_dst), len(val_dst)))
   
    model = {"resnet18": resnet18,
             "resnet34": resnet34,
             "resnet50": resnet50,
             "resnet101": resnet101,
             "resnet152": resnet152 }[opts.model]()
     
    trace_name = opts.trace_name
    if opts.sync_bn==True:
        print("Use sync batchnorm")
        model = convert_sync_batchnorm(model)
    print(model)

    if torch.cuda.device_count()>1: # Parallel
        print("%d GPU parallel"%(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
        model_ref = model.module # for ckpt
    else:
        model_ref = model
    model = model.to(device)
    
    # Set up metrics
    metrics = utils.StreamClsMetrics(10)
    
    # Set up optimizer
    group_decay, group_no_decay = utils.group_params(model_ref)
    assert(len(group_decay)+len(group_no_decay) == len(list(model_ref.parameters())))
    optimizer = torch.optim.SGD([  {'params': group_decay, 'weight_decay': opts.weight_decay},
                                   {'params': group_no_decay}], 
                                   lr=opts.lr, momentum=opts.momentum )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=0.1)
    print("optimizer:\n%s"%(optimizer))
    
    utils.mkdir('checkpoints')

    # Restore
    best_score = 0.0
    cur_epoch = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt)
        model_ref.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        cur_epoch = checkpoint["epoch"]+1
        best_score = checkpoint['best_score']
        print("Model restored from %s"%opts.ckpt)
        del checkpoint # free memory
    else:
        print("[!] Retrain")
    
    def save_ckpt(path):
        """ save current model
        """
        state = {
                    "epoch": cur_epoch,
                    "model_state": model_ref.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_score": best_score,
        }
        torch.save(state, path)
        print( "Model saved as %s"%path )

    # Set up criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')
    #==========   Train Loop   ==========#
    while cur_epoch < opts.epochs:
        # =====  Train  =====
        model.train()
        epoch_loss = train(cur_epoch=cur_epoch, criterion=criterion, model=model, optim=optimizer, train_loader=train_loader, device=device, scheduler=scheduler, vis=vis, trace_name=trace_name)
        print("End of Epoch %d/%d, Average Loss=%f"%(cur_epoch, opts.epochs, epoch_loss))

        if opts.enable_vis:
            vis.vis_scalar("Epoch Loss", trace_name, cur_epoch, epoch_loss )
        
        # =====  Save Latest Model  =====
        if (cur_epoch+1)%opts.ckpt_interval==0:
            save_ckpt( 'checkpoints/latest_resnet34_cifar10.pkl' )

        # =====  Validation  =====
        if (cur_epoch+1)%opts.val_interval==0:
            print("validate on val set...")
            model.eval()
            val_score = validate(model=model, loader=val_loader, device=device, metrics=metrics)
            print(metrics.to_str(val_score))
            
            # =====  Save Best Model  =====
            if val_score['Mean IoU']>best_score: # save best model
                best_score = val_score['Overall Acc']
                save_ckpt( 'checkpoints/latest_resnet34_cifar10.pkl')
            
            if vis is not None: # visualize validation score and samples
                vis.vis_scalar("[Val] Overall Acc",trace_name, cur_epoch, val_score['Overall Acc'] )
            
            if opts.val_on_trainset==True: # validate on train set
                print("validate on train set...")
                model.eval()
                train_score = validate(model=model, loader=train_loader, device=device, metrics=metrics)
                print(metrics.to_str(train_score))
                if vis is not None:
                    vis.vis_scalar("[Train] Overall Acc", trace_name, cur_epoch, train_score['Overall Acc'] )
                    
        cur_epoch+=1

if __name__=='__main__':
    main()


    



