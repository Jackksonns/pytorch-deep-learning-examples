import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import matplotlib.image as mpimg

#这份代码仅仅给定物体的左上xy右下xy（或者左上xy+宽高），用代码绘图工具在图片上画bbox而已。
#不具备识别物体类别的功能
# 设置图像显示的尺寸
def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图像大小"""
    plt.rcParams['figure.figsize'] = figsize

set_figsize()

# 读取图像
img = mpimg.imread('./catdog.jpg')

# 显示图像
plt.imshow(img)

# 边界框——两种表示法
def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

# bbox是边界框的英文缩写，定义猫和狗的边界框
dog_bbox, cat_bbox = [60.0, 45.0, 378.0, 516.0], [400.0, 112.0, 655.0, 493.0]

# 我们可以通过转换两次来验证边界框转换函数的正确性
boxes = torch.tensor((dog_bbox, cat_bbox))
assert torch.all(box_center_to_corner(box_corner_to_center(boxes)) == boxes)

# 辅助函数：将边界框表示成matplotlib的边界框格式
def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y), 宽, 高)
    return patches.Rectangle(
        (bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)

# 获取当前图像的坐标轴
ax = plt.gca()
# 添加边界框
ax.add_patch(bbox_to_rect(dog_bbox, 'blue'))
ax.add_patch(bbox_to_rect(cat_bbox, 'red'))

plt.show()
