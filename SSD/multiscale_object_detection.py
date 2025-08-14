import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torchvision.transforms.functional as TF
from PIL import Image

#SSD就是做了这个多尺度的目标检测（底层拟合小物体，顶层拟合大物体）
# 读取图片并返回 PIL 图像和其尺寸（高 h, 宽 w）
def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    return img, h, w

#特征图可理解为每一个卷积层的输出。
# 在特征图上生成锚框（anchor boxes），每个位置产生 len(sizes)*len(ratios) 个框（每个单位or像素作为锚框的中心）
def multibox_prior(data, sizes, ratios):
    """
    以每个像素为中心，生成不同形状的锚框。

    参数：
    - data: 输入特征图，形状为 (batch_size, channels, height, width)
    - sizes: 每个中心点生成锚框的尺度（相对比例）
    - ratios: 每个尺度下的长宽比

    返回：
    - anchors: 所有锚框的坐标，形状为 (batch_size, height * width * num_anchors, 4)，
               坐标格式为 (xmin, ymin, xmax, ymax)，值都在 [0,1] 区间。
    """
    _, _, in_height, in_width = data.shape
    device = data.device
    num_sizes, num_ratios = len(sizes), len(ratios)
    # 每个像素点上的锚框数： sizes + ratios - 1（按 SSD 原文）
    num_anchors = num_sizes + num_ratios - 1

    # 步长（一格在原图中占比）
    step_y = 1.0 / in_height
    step_x = 1.0 / in_width
    # 中心点坐标
    center_y = (torch.arange(in_height, device=device) + 0.5) * step_y
    center_x = (torch.arange(in_width,  device=device) + 0.5) * step_x
    shift_y, shift_x = torch.meshgrid(center_y, center_x, indexing='ij')
    shift_y = shift_y.reshape(-1)
    shift_x = shift_x.reshape(-1)

    # 计算每个比例下的宽高
    ws, hs = [], []
    for i, size in enumerate(sizes):
        for j, ratio in enumerate(ratios):
            if i == 0 or j == 0:
                r = torch.tensor(ratio, dtype=torch.float32, device=device)
                ws.append(size * torch.sqrt(r))
                hs.append(size / torch.sqrt(r))
    ws = torch.tensor(ws, device=device) * 0.5  # 半宽
    hs = torch.tensor(hs, device=device) * 0.5  # 半高

    # 所有中心点重复 num_anchors 次
    centers = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)
    centers = centers.repeat_interleave(num_anchors, dim=0)  # (H*W*num_anchors, 4)

    # 每种锚框尺寸也要在所有点上重复
    box_sizes = torch.stack((ws, hs, ws, hs), dim=1)
    box_sizes = box_sizes.repeat(len(shift_x), 1)            # (H*W*num_anchors, 4)

    # xmin, ymin, xmax, ymax
    mins = centers[:, :2] - box_sizes[:, :2]
    maxs = centers[:, 2:] + box_sizes[:, 2:]
    anchors = torch.cat((mins, maxs), dim=1)

    return anchors.unsqueeze(0)  # (1, H*W*num_anchors, 4)

# 可视化锚框
def show_bboxes(ax, bboxes, colors=None):
    """
    在图像上显示边界框

    参数：
    - ax: matplotlib 的图像坐标轴
    - bboxes: 边界框列表，格式为 [[xmin, ymin, xmax, ymax], ...]
    - colors: 边框颜色列表（可选）
    """
    def get_color(i):
        cmap = plt.cm.get_cmap("tab10")
        return cmap(i % 10)

    colors = colors or [get_color(i) for i in range(len(bboxes))]
    for bbox, c in zip(bboxes, colors):
        rect = patches.Rectangle(
            (bbox[0], bbox[1]),
            bbox[2] - bbox[0],
            bbox[3] - bbox[1],
            fill=False,
            edgecolor=c,
            linewidth=1.5
        )
        ax.add_patch(rect)

# 主函数：显示指定特征图尺寸下的锚框
def display_anchors(fmap_w, fmap_h, s, img_path='./catdog.jpg'):
    """
    在图像上可视化给定特征图尺寸和尺度 s 下的锚框。

    参数：
    - fmap_w: 特征图宽度
    - fmap_h: 特征图高度
    - s: 锚框尺度列表（相对比例），如 [0.15]
    - img_path: 输入图像路径
    """
    img, h, w = load_image(img_path)
    fig = plt.figure(figsize=(6, 6))
    ax = plt.imshow(img).axes

    # 伪特征图：batch=1, channels=3（任意）, H=fmap_h, W=fmap_w
    fmap = torch.zeros((1, 3, fmap_h, fmap_w))
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    # 将归一化坐标映射回像素坐标
    scale = torch.tensor((w, h, w, h), dtype=torch.float32)
    boxes = (anchors[0] * scale).detach().cpu().numpy()

    show_bboxes(ax, boxes)
    plt.show()

# 示例调用
if __name__ == '__main__':
    #s可理解为锚框大小
    #探测小目标（s=0.15指的是探测（占）整个图片15%大小的区域；同时因为高宽均为4，所以画出来是16个像素px
    display_anchors(fmap_w=4, fmap_h=4, s=[0.15])
    #将特征图的高宽均减半，然后使用较大的锚框来检测较大的目标
    display_anchors(fmap_w=2, fmap_h=2, s=[0.4])
    #更大的锚框
    display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
