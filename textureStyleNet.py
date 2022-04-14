import os
import torch
import argparse
from LinearStyleTransfer.libs.Criterion import LossCriterion
from LinearStyleTransfer.libs.Loader import Dataset
from LinearStyleTransfer.libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from LinearStyleTransfer.libs.utils import print_options
from LinearStyleTransfer.libs.models import encoder3,encoder4, encoder5
from LinearStyleTransfer.libs.models import decoder3,decoder4, decoder5
from LinearStyleTransfer.libs.models import encoder5 as loss_network
import torch.optim as optim
import torch.nn as nn
import pywavefront
import torchvision.transforms as transforms
from PIL import Image
import renderUtil

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth',
                    help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth',
                    help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth',
                    help='pre-trained decoder path')
parser.add_argument("--stylePath", default="/home/xtli/DATA/wikiArt/train/images/",
                    help='path to wikiArt dataset')
parser.add_argument("--contentPath", default="/home/xtli/DATA/MSCOCO/train2014/images/",
                    help='path to MSCOCO dataset')
parser.add_argument("--outf", default="trainingOutput/",
                    help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r41",
                    help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41",
                    help='layers for style')
parser.add_argument("--batchSize", type=int,default=1,
                    help='batch size')
parser.add_argument("--niter", type=int,default=1000,
                    help='iterations to train the model')
parser.add_argument('--loadSize', type=int, default=300,
                    help='scale image size')
parser.add_argument('--fineSize', type=int, default=256,
                    help='crop image size')
parser.add_argument("--lr", type=float, default=1e-2,
                    help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0,
                    help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02,
                    help='style loss weight')
parser.add_argument("--log_interval", type=int, default=500,
                    help='log interval')
parser.add_argument("--gpu_id", type=int, default=0,
                    help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=1000,
                    help='checkpoint save interval')
parser.add_argument("--layer", default="r41",
                    help='which features to transfer, either r31 or r41')
parser.add_argument("--matrixPath", default='models/r41.pth',
                    help='pre-trained model path')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.cuda = torch.cuda.is_available()
print_options(opt)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True

################# DATA #################
#content_dataset = Dataset(opt.contentPath,opt.loadSize,opt.fineSize,test=True)
#content_loader = torch.utils.data.DataLoader(dataset=content_dataset,
#                                             batch_size = opt.batchSize,
#                                             shuffle = False,
#                                             num_workers = 1)
#style_dataset = Dataset(opt.stylePath,opt.loadSize,opt.fineSize,test=True)
#style_loader = torch.utils.data.DataLoader(dataset=style_dataset,
#                                           batch_size = opt.batchSize,
#                                           shuffle = False,
#                                           num_workers = 1)



################# MODEL #################

vgg = encoder4()
dec = decoder4()
matrix = MulLayer(opt.layer)
vgg5 = loss_network()

vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))
vgg5.load_state_dict(torch.load(opt.loss_network_dir))


## Fix encoder and lost net weight
for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False

for param in dec.parameters():
    param.requires_grad = True
for param in matrix.parameters():
    param.requires_grad = True

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers.split(','),
                          opt.content_layers.split(','),
                          opt.style_weight,
                          opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GLOBAL VARIABLE #################
contentV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)
styleV = torch.Tensor(opt.batchSize,3,opt.fineSize,opt.fineSize)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    matrix.cuda()
    contentV = contentV.cuda()
    styleV = styleV.cuda()



transform = transforms.Compose([
            transforms.Resize([opt.fineSize,opt.fineSize]),
            transforms.RandomVerticalFlip(1),
            transforms.ToTensor()])

content = Image.open("Horse.tga").convert("RGB") #renderUtil.textureLoader("Horse.tga",opt.fineSize)
content = transform(content).unsqueeze(0)
style = Image.open("0001.png").convert("RGB")
style = transform(style).unsqueeze(0)
model = renderUtil.modelLoader("horse2.obj")
################# TRAINING #################
def adjust_learning_rate(optimizer, iteration):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = opt.lr / (1+iteration*1e-5)


for iteration in range(1,opt.niter+1):
    optimizer.zero_grad()

    contentV.resize_(content.size()).copy_(content)
    styleV.resize_(style.size()).copy_(style)

    # forward

    #style transfer texture
    sF = vgg(styleV)
    cF = vgg(contentV)
    
    if(opt.layer == 'r41'):
        feature,transmatrix = matrix(cF[opt.layer],sF[opt.layer])
    else:
        feature,transmatrix = matrix(cF,sF)

    transfer = dec(feature)
    texture = transfer.clamp(0,1)
    #print(texture.shape)
    texture = torch.swapaxes(texture,1,2)
    texture = torch.swapaxes(texture,2,3)
    cb,cc,ch,cw = texture.size()
    texture = texture.view(cc,ch,cw)
    #print(texture.shape)
    texture = texture.contiguous()
    #print(texture.shape)
    #render
    views = renderUtil.randomViewRender(model[0],model[1],model[2],model[3],texture,size=opt.fineSize)


    train_losses = 0
    styleLosses = 0
    contentLosses = 0
    #calculate loss
    for view in views:
        view = torch.swapaxes(view,3,2)
        view = torch.swapaxes(view,2,1)
        view.unsqueeze(0)
        view = view.contiguous()

        #print(styleV.shape)
        #print(contentV.shape)
        #print(view.shape)
        sF_loss = vgg5(styleV)
        cF_loss = vgg5(contentV)
        tF = vgg5(view)
        loss,styleLoss,contentLoss = criterion(tF,sF_loss,cF_loss)
        train_losses += loss
        styleLosses += styleLoss
        contentLosses += contentLoss

    train_losses /= len(views)
    styleLosses /= len(views)
    contentLosses /= len(views)
        
    # backward & optimization
    train_losses.backward()
    optimizer.step()
    print(f"Iteration: [{iteration}/{opt.niter}] Loss: {train_losses} contentLoss: {contentLoss} styleLoss: {styleLoss} Learng Rate is {optimizer.param_groups[0]['lr']}")
    adjust_learning_rate(optimizer,iteration)