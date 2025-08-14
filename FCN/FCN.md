# FCN

（注意，我当时运行时数据集的路径有点抽象，是VOCdevkit/VOCdevkik（之后一样），所以改了d2l的load函数以及fcn代码中的voc_folder；复现时文件夹结构不同需要修改回来）

```bash
mkdir -p VOCdevkit
cd VOCdevkit
wget -c http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar
```

wget -c断点续传指令



加速下载指令：

```bash
sudo apt update
sudo apt install -y aria2

aria2c -c -x 16 -s 16 -k 1M -o VOCtrainval_11-May-2012.tar \
"http://data.brainchip.com/dataset-mirror/voc/VOCtrainval_11-May-2012.tar"

```



```bash
root@autodl-container-cbfe4794e2-d8b71a74:~/autodl-tmp# python fcn.py
/root/miniconda3/lib/python3.12/site-packages/torchvision/models/_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
  warnings.warn(
/root/miniconda3/lib/python3.12/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet18_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet18_Weights.DEFAULT` to get the most up-to-date weights.
  warnings.warn(msg)
input image shape: torch.Size([561, 728, 3])
output image shape: torch.Size([1122, 1456, 3])
read 1114 examples
read 1078 examples
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
<Figure size 350x250 with 1 Axes>
loss 0.420, train acc 0.869, test acc 0.853
431.2 examples/sec on [device(type='cuda', index=0)]
root@autodl-container-cbfe4794e2-d8b71a74:~/autodl-tmp#
```

