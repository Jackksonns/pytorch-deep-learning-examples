import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
import os
import cv2
import timm

import albumentations
from albumentations import pytorch as AT

#below are all from https://github.com/seefun/TorchUtils, thanks seefun to provide such useful tools
import torch_utils as tu

#load data
CLEAN_DATASET = 0
FOLD = 5
csv = pd.read_csv('clean_train_v4.csv') if CLEAN_DATASET else pd.read_csv('train.csv') #cleaned data has no obvious improvement
#tratifiedKFold ：
    # 是scikit-learn库中的一个交叉验证分割器。
    # “Stratified”表示分层，即在划分每个fold时会保证每个子集中的各类别样本比例与原始数据集一致（常用于分类任务，防止某一类被分得太少）。
    # n_splits=FOLD ：
    # 指定将数据集分成多少个fold（通常是5或10）。
    # FOLD 是你自己定义的变量，比如 FOLD=5 代表5折交叉验证。
    # random_state=709 ：
    # 随机种子，保证每次划分的结果可复现。(随便选一个整数作为种子就行，只要用同样的数据和同样的random_state，每次划分结果都一样。注意：如果你把709换成123，划分顺序就会变；但只要一直用709，你的代码、实验结果都是可复现的。)
    # shuffle=True ：
    # 在分层之前先把样本随机打乱，提升随机性和泛化能力。
sfolder = StratifiedKFold(n_splits=FOLD,random_state=709,shuffle=True)
#3. 生成每一折的训练/验证索引（记录每一折的训练集和验证集的行号索引，方便后续训练循环用）
train_folds = []
val_folds = []
for train_idx, val_idx in sfolder.split(csv['image'], csv['label']):
  train_folds.append(train_idx)
  val_folds.append(val_idx)
  print(len(train_idx), len(val_idx))
#4. 创建标签到数字的映射（把所有类别标签映射成连续的数字编号（并保证顺序一致），方便深度学习训练）
labelmap_list = sorted(list(set(csv['label']))) #sorting is necessary to reproduce the order of the labelmap
labelmap = dict()
for i, label in enumerate(labelmap_list):
  labelmap[label] = i
print(labelmap)

#define dataset根据csv文件读取图片，并返回图片与对应的标签
class LeavesDataset(Dataset):
  def __init__(self, csv, transform=None):
    self.csv = csv
    self.transform = transform
  
  def __len__(self):
    return len(self.csv['image'])
  
  def __getitem__(self, idx):
    img = cv2.imread(self.csv['image'][idx])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    label = labelmap[self.csv['label'][idx]]
    if self.transform:
      img = self.transform(image = img)['image']
    return img, torch.tensor(label).type(torch.LongTensor)

#批量创建训练集和测试集数据加载器（DataLoader）函数（其中参数就要求传入数据transform操作）
def create_dls(train_csv, test_csv, train_transform, test_transform, bs, num_workers):
  train_ds = LeavesDataset(train_csv, train_transform)
  test_ds = LeavesDataset(test_csv, test_transform)
  train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, drop_last=True)
  test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, drop_last=False)
  return train_dl, test_dl, len(train_ds), len(test_ds)

#data augument——这是先封装好这些增强和预处理的操作，后面把这个变量作为参数传入函数
train_transform1 = albumentations.Compose([
    albumentations.Resize(112, 112, interpolation=cv2.INTER_AREA),
    albumentations.RandomRotate90(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.3),
    albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.0625, rotate_limit=45, border_mode=1, p=0.5),
    #tu.randAugment(N=2,M=6,p=1,cut_out=True),    
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])
    
test_transform1 = albumentations.Compose([
    albumentations.Resize(112, 112, interpolation=cv2.INTER_AREA),
    albumentations.Normalize(),
    AT.ToTensorV2(),
    ])

#define testdataset
class LeavesTestDataset(Dataset):
  def __init__(self, csv, transform=None):
    self.csv = csv
    self.transform = transform
  
  def __len__(self):
    return len(self.csv['image'])
  
  def __getitem__(self, idx):
    img = Image.open(self.csv['image'][idx])
    if self.transform:
      img = self.transform(img)
    return img

#把之前定义的数据增强or预处理操作变量传入以下函数
def create_testdls(test_csv, test_transform, bs):
  test_ds = LeavesTestDataset(test_csv, test_transform)
  test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2)
  return test_dl

transform_test = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

#把经过标准化和张量处理的图片还原成可以直接显示的PIL图片，用于可视化训练数据
def show_img(x):
  trans = transforms.ToPILImage()
  mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
  std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
  x = (x*std)+mean
  x_pil = trans(x)
  return x_pil

train_csv = csv.iloc[train_folds[0]].reset_index()
val_csv = csv.iloc[val_folds[0]].reset_index()

#创建训练集与验证集的DataLoader，也就是调用上面定义好的函数
train_dl, val_dl, n_train, n_val = create_dls(train_csv, val_csv, train_transform=train_transform1, test_transform=test_transform1, bs=64, num_workers=4)

#mixup_fn = Mixup(prob=1., switch_prob=0.0, onehot=True, label_smoothing=0.05, num_classes=176)
for x, y in train_dl:
  #imgs_train, labels_train = mixup_fn(x, y)
  break

#use pretrained model

#model = torchvision.models.resnet50(pretrained=True)
#model = torchvision.models.resnext101_32x8d(pretrained=True)
#model = timm.create_model('seresnext50_32x4d', pretrained=True)
model = timm.create_model('resnet50d', pretrained=True)
#model = timm.create_model('resnest50d', pretrained=True)

#model = timm.create_model('tf_efficientnetv2_l_in21ft1k', pretrained=True)

#改最后一层，使其数据集要分类的类别数一致。
model.fc = nn.Linear(model.fc.in_features, len(labelmap_list))
nn.init.xavier_uniform_(model.fc.weight);

model.cuda()
device = 'cuda'

#Use label smoothing in loss function.--label smoothing 的思想是：把真实标签的概率从1变成1-ε（比如0.9），其他类别分摊ε（比如0.1）。这样模型不会对某个类别太自信，有助于泛化。
# Use AdamW as our optimizer, which is marginally better than Adam. Note that we make 2 param_groups in the optimizer because we try to update the CNN layers much slower than the last fc layer.
#LabelSmoothing自定义损失函数——在训练时对标签做平滑处理，而不是直接用one-hot标签 避免太自信
#在训练时，只用这个 LabelSmoothing 损失函数就可以了，不需要再另外用别的分类损失函数 
class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
params_1x = [param for name, param in model.named_parameters()
             if name not in ["fc.weight", "fc.bias"]]
lr = 5e-4

#优化器用来AdamW，同时底层和顶层用了不同的params(params_groups实现)
optimizer = torch.optim.AdamW([{'params': params_1x},
                                   {'params': model.fc.parameters(),
                                    'lr': lr * 10}],
                                lr=lr, weight_decay=0.001) #finetuning
'''
from optim import RangerLars
optimizer = RangerLars([{'params': params_1x},
                        {'params': model.fc.parameters(),
                                    'lr': lr * 10}], lr=lr, weight_decay=0.001)
'''
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.001)

loss_fn = LabelSmoothing(0.1) #smoothing：平滑参数，通常设置为0.1或0.05

import math
import matplotlib.pyplot as plt
import numpy as np

#学习率查找器函数。训练前快速找到一个合适的学习率（lr），为后续正式训练做准备
def find_lr(model, factor, train_dl, optimizer, loss_fn, device, init_lr=1e-8, final_lr=1e-1, beta=0.98, plot=True, save_dir=None):
    num = len(train_dl) - 1
    mult = (final_lr / init_lr) ** (1/num)
    lr = init_lr
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    scaler = torch.cuda.amp.GradScaler() # for AMP training 

    if 1:
      for x, y in train_dl:
          x, y = x.to(device), y.to(device)
          batch_num += 1
          optimizer.zero_grad()
          with torch.cuda.amp.autocast():
            out = model(x)
            loss = loss_fn(out, y)
          #smoothen the loss
          avg_loss = beta * avg_loss + (1-beta) * loss.data.item() #check
          smoothed_loss = avg_loss / (1 - beta**batch_num) #bias correction
          #stop if loss explodes
          if batch_num > 1 and smoothed_loss > 4 * best_loss: #prevents explosion
              break
          #record the best loss
          if smoothed_loss < best_loss or batch_num == 1:
              best_loss = smoothed_loss
          #store the values
          losses.append(smoothed_loss)
          log_lrs.append(math.log10(lr))
          #do the sgd step
          #loss.backward()
          #optimizer.step()
          scaler.scale(loss).backward()
          scaler.step(optimizer)
          scaler.update()
          #update the lr for the next step
          lr *= mult
          optimizer.param_groups[0]['lr'] = lr
    #Suggest a learning rate
    log_lrs, losses = np.array(log_lrs), np.array(losses)
    idx_min = np.argmin(losses)
    min_log_lr = log_lrs[idx_min]
    lr_auto = (10 ** (min_log_lr)) /factor
    if plot:
        selected = [np.argmin(np.abs(log_lrs - (min_log_lr-1)))] #highlight the suggested lr
        plt.figure()
        plt.plot(log_lrs, losses,'-gD', markevery=selected)
        plt.xlabel('log_lrs')
        plt.ylabel('loss')
        plt.title('LR Range Test')
        if save_dir is not None:
            plt.savefig(f'{save_dir}/lr_range_test.png')
        else:
            plt.savefig(f'lr_range_test.png')
    return lr_auto

#重新初始化模型。get_learner函数一次性组装好用于训练深度学习模型所需的模型（且最后一层自动适配类别数）、优化器（参数分组，不同层不同学习率）、损失函数（可选 mixup/label smoothing）、测试损失、学习率调度器
# 、Mixup 增强函数。
def get_learner(lr, nb, epochs, model_name='resnet50d', MIXUP=0.1):
  mixup_fn = tu.Mixup(prob=MIXUP, switch_prob=0.0, onehot=True, label_smoothing=0.05, num_classes=len(labelmap_list))
  model = timm.create_model(model_name, pretrained=True)
  model.fc = nn.Linear(model.fc.in_features, len(labelmap_list))
  nn.init.xavier_uniform_(model.fc.weight)
  model.cuda()

  params_1x = [param for name, param in model.named_parameters()
              if name not in ["fc.weight", "fc.bias"]]

  optimizer = torch.optim.AdamW([{'params': params_1x},
                                    {'params': model.fc.parameters(),
                                      'lr': lr*10}],
                                  lr=lr, weight_decay=2e-4)

  loss_fn = tu.SoftTargetCrossEntropy() if MIXUP else LabelSmoothing(0.1)
  loss_fn_test = F.cross_entropy
  '''
  import math
  def warmup_one_cycle(y1=0.0, y2=1.0, steps=100, warmup_steps=0): #no warmup is better experimentally
          #sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
          return lambda x: x / warmup_steps if x < warmup_steps \
                          else ((1 - math.cos((x-warmup_steps) * math.pi / steps)) / 2) * (y2 - y1) + y1

  lf = warmup_one_cycle(1, 0.2, epochs*nb, 3*nb)
  #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf) 
    #lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5*nb, eta_min=lr_suggested/100)
  '''
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*nb, eta_min=lr/20)
  return model, optimizer, loss_fn, loss_fn_test, lr_scheduler, mixup_fn

#train
#Start training
# #Copilot said: 这段代码是深度学习模型训练与推理的主流程，适用于Kaggle或者类似竞赛的数据集。它实现了K折交叉验证下的训练、验证、模型保存和测试集推理。
#注意：每个fold都独立训练、验证，并根据验证集表现保存最优模型（所以会有多个最优模型出现）

'''device = 'cuda'：用GPU训练。
save_dir：模型权重保存路径。
labelmap_inverse：从类别索引到类别名的映射，便于结果反查。
EPOCHS, MIXUP：分别为训练轮数和Mixup数据增强概率。
scaler：用于AMP（自动混合精度），加速和节省显存。
'''

#Start training
import time
import ttach as tta
device = 'cuda'
save_dir = './'
#map from idx to string
labelmap_inverse = dict()
for key_ in labelmap.keys():
  labelmap_inverse[labelmap[key_]] = key_
EPOCHS = 50
MIXUP = 0.1
 
scaler = torch.cuda.amp.GradScaler() # for AMP training 

for fold in range(FOLD):
  print(f'Start Fold{fold}...')
  train_csv = csv.iloc[train_folds[2]].reset_index()
  val_csv = csv.iloc[val_folds[2]].reset_index()
  train_dl, val_dl, n_train, n_val = create_dls(train_csv, val_csv, train_transform=train_transform1, test_transform=test_transform1, bs=64, num_workers=4)
  model, optimizer, loss_fn, loss_fn_test, lr_scheduler, mixup_fn = get_learner(3e-4, len(train_dl), EPOCHS, model_name='resnet50d', MIXUP=MIXUP)
  model_name = f'5fold_test_fold{fold}'
  train_losses = [] 
  val_losses = []
  train_accus = []
  val_accus = []
  best_accu = 0
  best_loss = float('inf')
  lrs = []
  for epoch in range(EPOCHS):
          t1 = time.time()
          val_accu = 0
          train_accu = 0
          train_losses_tmp = []
          #Train
          model.train()
          t_inf = 0
          for x, y in train_dl:
              if MIXUP:
                x, y = mixup_fn(x, y)
              x, y = x.to(device), y.to(device)
              #Forward
              with torch.cuda.amp.autocast():
                pred = model(x)
                loss = loss_fn(pred, y)
              #Backward
              #loss.backward()
              #optimizer.step()
              scaler.scale(loss).backward()
              scaler.step(optimizer)
              scaler.update()
              lr_scheduler.step()
              optimizer.zero_grad()
              #Statistics
              lrs.append(optimizer.param_groups[0]['lr']) #group 0,1,2 share the learning rate
              train_losses_tmp.append(loss.data.item())
              pred_labels = torch.argmax(pred.data, dim=1)
              y_labels = torch.argmax(y.data, dim=1) if MIXUP else y.data
              train_accu += (pred_labels==y_labels).float().sum()
          t_inf /= len(train_dl)
          train_losses.append(np.mean(np.array(train_losses_tmp)))
          train_accu /= n_train
          train_accus.append(train_accu.data.item())

          t2 = time.time()
          #Validation
          val_losses_tmp = []
          model.eval()
          with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logit = model(x)
                val_loss = loss_fn_test(logit, y) 
                val_losses_tmp.append(val_loss.data.item())
                pred = torch.argmax(logit.data, dim=1)
                val_accu += (pred==y.data).float().sum()
          t3 = time.time()
          val_loss = np.mean(np.array(val_losses_tmp))
          val_losses.append(val_loss)
          val_accu /= n_val
          val_accus.append(val_accu.data.item())
          print('fold', fold, 'epoch', epoch, 'train_loss', train_losses[epoch], 'val_loss', val_losses[epoch], 'val_accu', val_accu, 'train_accu', train_accu, 'train time', t2-t1, 'val time', t3-t2, 'lr[0]', lrs[-1])
          if save_dir is not None:
              if val_accu == best_accu:
                  if val_loss < best_loss: #never satisfied
                      checkpoint = {"model": model.state_dict()}
                      torch.save(checkpoint, os.path.join(save_dir,f'{model_name}_best.pth'))
                      print(f'Stored a new best model in {save_dir}')
                      best_loss = val_loss
              elif val_accu > best_accu:
                  checkpoint = {"model": model.state_dict()}
                  torch.save(checkpoint, os.path.join(save_dir,f'{model_name}_best.pth'))
                  print(f'Stored a new best model in {save_dir}')
                  best_accu = val_accu
              '''
              if epoch == EPOCHS - 1:
                  checkpoint = {"model": model.state_dict()}
                  torch.save(checkpoint, os.path.join(save_dir,f'{model_name}_last.pth'))
                  print(f'Stored the last model in {save_dir}')
              '''
  #test time对测试集做预测，使用了TTA（Test Time Augmentation） 增强推理结果的稳健性
  test_csv = pd.read_csv('test.csv')
  test_dl = create_testdls(test_csv, transform_test, bs=8)
  model.eval()
  tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.flip_transform(),  merge_mode='mean')
  tta_model.eval()
  res = []
  for x in test_dl:
      x = x.to(device)
      logit = tta_model(x)
      pred = torch.argmax(logit.data, dim=1).cpu().numpy()
      for i in range(len(pred)):
          res.append(labelmap_inverse[pred[i]])
  test_csv.insert(1, 'label', res)
  test_csv.to_csv((f'submission_e50{model_name}_fold{fold}.csv'), index=False)
  print('test cvs is saved')


#画学习曲线

plt.plot(train_accus)
plt.plot(val_accus)
plt.plot(train_losses)
plt.plot(val_losses)
plt.ylim(0, 1.3)
plt.legend(['train_accus', 'val_accus', 'train_losses', 'val_losses'])
#plt.show()
plt.title('Learning Curve')
plt.xlabel('epochs')

plt.savefig(f'{model_name}_acc98d34.png')

#对5折交叉验证得到的多个模型预测结果进行集成（ensemble），采用多数投票法（majority voting）得到最终预测结果，并保存为Kaggle提交文件。
#1. 收集5折模型的预测结果（CSV）
files = sorted(os.listdir('./'))
cvss_label = []
for file in files:
    if file.endswith('.csv') and file not in ['test.csv', 'train.csv', 'sample_submission.csv']:
        labels = pd.read_csv(file)['label'].to_numpy()
        cvss_label.extend(labels)  # 用 extend 而不是 append

cvss_label = np.array(cvss_label)

#2. 多数投票融合（majority voting）
#对于每一个测试样本，收集所有模型的预测结果，取出现次数最多的那个类别作为最终预测（即“投票”）。
# 最终final_label就是融合后的预测标签。
#也就是说，他最后的预测结果是五个模型中同时预测，得票最高的那一个标签为最终预测结果，进一步增强其鲁棒性。

#注意：下面这个没跑通，也就是提取各个模型对于同一个样本的预测结果的综合概率有报错，但上面整体的实现都没问题。
from scipy import stats
final_label = []
for i in range(cvss_label.shape[1]):
    majority_label = stats.mode(cvss_label[:,i])[0][0]
    final_label.append(majority_label)

#3. 生成Kaggle提交文件
test_csv = pd.read_csv('test.csv')
test_csv.insert(1, 'label', final_label)
test_csv.to_csv(os.path.join(save_dir,'submit_ensemble5_e50.csv'), index=False)
