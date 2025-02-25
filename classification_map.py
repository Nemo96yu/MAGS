import scipy.io as io
import numpy as np
import matplotlib.pyplot as plt

data_path ='./classification_maps/Houston3_parse_gt.mat'
data = io.loadmat(data_path)
# y = data['Indian_GT_SLAP']
y = data["test_gt"]

def Draw_Classification_Map(label, name: str, scale: float = 4.0, dpi: int = 400):
    label = label
    if name == 'indian' or name == 'salinas':
        colors = [  # 自定义颜色列表，每个类别对应一个颜色
            (0, 0, 0),  # 类别0：黑色
            (255, 255, 95),  # 类别1：浅黄色
            (0, 0, 255),  # 类别2：蓝色
            (255, 85, 0),  # 类别3：橘红色
            (0, 255, 128),  # 类别4：深青色
            (255, 0, 255),  # 类别5：粉色
            (83, 0, 255),  # 类别6：蓝紫色
            (68, 162, 255),  # 类别7：天蓝色
            (0, 255, 64),  # 类别8：明绿色
            (150, 150, 75),  # 类别9：土黄色
            (170, 72, 146),  # 类别10：粉紫色
            (106, 181, 255),  # 类别11：浅蓝色
            (0, 72, 72),  # 类别12：深青色
            (0, 255, 255),  # 类别13：绿色
            (128, 64, 0),  # 类别14：棕色
            (151, 255, 177),  # 类别15：浅绿色
            (255, 255, 0),  # 类别16：明黄色
            # 可根据实际需要继续添加其他类别的颜色
        ]
    elif name == 'Botswana':
        colors = [  # 自定义颜色列表，每个类别对应一个颜色
            # (0, 0, 0),  # 类别0：黑色
            (255, 255, 95),  # 类别1：浅黄色
            (0, 0, 255),  # 类别2：蓝色
            (255, 85, 0),  # 类别3：橘红色
            (0, 255, 128),  # 类别4：深青色
            (255, 0, 255),  # 类别5：粉色
            (83, 0, 255),  # 类别6：蓝紫色
            (68, 162, 255),  # 类别7：天蓝色
            (0, 255, 64),  # 类别8：明绿色
            (150, 150, 75),  # 类别9：土黄色
            (170, 72, 146),  # 类别10：粉紫色
            (106, 181, 255),  # 类别11：浅蓝色
            (0, 72, 72),  # 类别12：深青色
            (0, 255, 128),  # 类别13：绿色
            (128, 64, 0),  # 类别14：棕色
            # 可根据实际需要继续添加其他类别的颜色
        ]

    # 创建一个空白的RGB图像，形状与label相同
    rgb_label = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)

    # 将颜色值赋给每个类别对应的像素
    for i, color in enumerate(colors):
        rgb_label[label == i] = color

    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.imshow(rgb_label)  # 显示彩色图像
    fig.set_size_inches(label.shape[1]*2.0/dpi, label.shape[0]*2.0/dpi)

    foo_fig = plt.gcf()
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    foo_fig.savefig('./classification_maps/Houston_parse.png',transparent=True, dpi=dpi, pad_inches=0)
    plt.show()

Draw_Classification_Map(y,'indian')