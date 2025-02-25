import numpy as np
import torch
import matplotlib.pyplot as plt

def get_classification_map(y_pred, y):

    height = y.shape[0]
    width = y.shape[1]
    k = 0
    cls_labels = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            target = int(y[i, j])
            if target == 0:
                continue
            else:
                cls_labels[i][j] = y_pred[k]+1
                k += 1

    return cls_labels

def list_to_colormap(x_list):
    y = np.zeros((x_list.shape[0], 3))
    # colors = [  # 自定义颜色列表，每个类别对应一个颜色
    #     (0, 0, 0),  # 类别0：黑色
    #     (255, 255, 95),  # 类别1：浅黄色
    #     (0, 0, 255),  # 类别2：蓝色
    #     (255, 85, 0),  # 类别3：橘红色
    #     (0, 255, 128),  # 类别4：深青色
    #     (255, 0, 255),  # 类别5：粉色
    #     (83, 0, 255),  # 类别6：蓝紫色
    #     (68, 162, 255),  # 类别7：天蓝色
    #     (0, 255, 64),  # 类别8：明绿色
    #     (150, 150, 75),  # 类别9：土黄色
    #     (170, 72, 146),  # 类别10：粉紫色
    #     (106, 181, 255),  # 类别11：浅蓝色
    #     (0, 72, 72),  # 类别12：深青色
    #     (0, 255, 255),  # 类别13：绿色
    #     (128, 64, 0),  # 类别14：棕色
    #     (151, 255, 177),  # 类别15：浅绿色
    #     (255, 255, 0),  # 类别16：明黄色
    #     # 可根据实际需要继续添加其他类别的颜色
    # ]
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 255, 95]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 85, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 128]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([83, 0, 255]) / 255.
        if item == 7:
            y[index] = np.array([68, 162, 255]) / 255.
        if item == 8:
            y[index] = np.array([0, 255, 64]) / 255.
        if item == 9:
            y[index] = np.array([150, 150, 75]) / 255.
        if item == 10:
            y[index] = np.array([170, 72, 146]) / 255.
        if item == 11:
            y[index] = np.array([106, 181, 255]) / 255.
        if item == 12:
            y[index] = np.array([0, 72, 72]) / 255.
        if item == 13:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 14:
            y[index] = np.array([128, 64, 0]) / 255.
        if item == 15:
            y[index] = np.array([151, 255, 177]) / 255.
        if item == 16:
            y[index] = np.array([255, 255, 0]) / 255.

    return y

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1]*2.0/dpi, ground_truth.shape[0]*2.0/dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def test(device, net, test_loader,lable_embed):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    y_test = 0
    for spe, pca, pad, labels in test_loader:
        spe, pca, pda = spe.to(device), pca.to(device), pad.to(device)
        # outputs = net(spe.type(torch.cuda.FloatTensor), pca.type(torch.cuda.FloatTensor),
        #                  pda.type(torch.cuda.FloatTensor), False)
        outputs = net(pca.type(torch.cuda.FloatTensor), spe.type(torch.cuda.FloatTensor), lable_embed,0,True)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test

def get_cls_map(net, device, all_data_loader, y, class_vectors):

    y_pred, y_new = test(device, net, all_data_loader,class_vectors)
    cls_labels = get_classification_map(y_pred, y)
    x = np.ravel(cls_labels)
    gt = y.flatten()

    y_list = list_to_colormap(x)
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (y.shape[0], y.shape[1], 3))
    gt_re = np.reshape(y_gt, (y.shape[0], y.shape[1], 3))
    # classification_map(y_re, y, 300,
    #                    'classification_maps/' + 'IP_predictions.eps')
    classification_map(y_re, y, 300,
                       'classification_maps/' + 'Houston_MAGS.png')
    # classification_map(gt_re, y, 300,
    #                    'classification_maps/' + 'Houston_gt.png')
    print('------Get classification maps successful-------')

