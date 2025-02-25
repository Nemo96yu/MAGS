import scipy.io as sio
import torch.utils.data
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from transformers import BertModel, BertTokenizer, logging
import torch.optim as optim
from operator import truediv
import get_cls_map
import time
from loss import *
from Autoencoder import AutoEncoder
import torch.backends.cudnn as cudnn
import os
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def setup_seed(seed):
    """
    setup random seed to fix the result
    Args:
        seed: random seed
    Returns: None
    """
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.enable = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)


def loadData(data_name):
    # 读入数据 Indian
    data = sio.loadmat('../data/Indian_pines_corrected.mat')['indian_pines_corrected']
    labels = sio.loadmat('../data/Indian_pines_gt.mat')['indian_pines_gt']
    if data_name == "Pavia":
        data = sio.loadmat('../data/PaviaU.mat')['paviaU']
        labels = sio.loadmat('../data/PaviaU_gt.mat')['paviaU_gt']
    if data_name == "Houston":
        data = sio.loadmat('../data/HoustonU.mat')['HoustonU']
        labels = sio.loadmat('../data/HoustonU.mat')['HoustonU_GT']
    return data, labels


# 对高光谱数据 X 应用 PCA 变换
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))

    return newX


# 对单个像素周围提取 patch 时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X

    return newX


# 在每个像素周围提取 patch ，然后创建成符合 keras 处理的格式
def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels


def splitTrainTestSet(X, y, testRatio, randomState=12):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=testRatio,
                                                        random_state=randomState,
                                                        stratify=y)

    return X_train, X_test, y_train, y_test


def generate_candidate_label(train_labels, partial_rate, i):
    k = int(torch.max(train_labels) - torch.min(train_labels) + 1)
    n = len(train_labels)
    train_labels = torch.Tensor(train_labels).long()
    partialY = torch.zeros(n, k)
    partialY[torch.arange(n), train_labels] = 1.0
    transition_matrix = np.eye(k)
    transition_matrix[np.where(~np.eye(transition_matrix.shape[0], dtype=bool))] = partial_rate
    np.random.seed(i)
    random_n = np.random.uniform(0, 1, size=(n, k))

    for j in range(n):  # for each instance
        partialY[j, :] = torch.from_numpy((random_n[j, :] < transition_matrix[train_labels[j], :]) * 1)

    print("Finish Generating Candidate Label Sets!\n")
    return partialY


BATCH_SIZE_TRAIN = 128


# indain =128
# Pavia = 64
def create_data_loader(data, i):
    # 地物类别
    # class_num = 16
    # 读入数据
    X, y = loadData(data)
    # 用于测试样本的比例
    test_ratio = 0.95
    # 每个像素周围提取 patch 的尺寸
    patch_size = 13
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 32

    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y_all = createImageCubes(X_pca, y, windowSize=patch_size)
    # 数据归一化
    img = np.reshape(X, (-1, X.shape[2]))
    img = StandardScaler().fit_transform(img)
    img_std = np.reshape(img, (X.shape[0], X.shape[1], X.shape[2]))
    X_pad, _ = createImageCubes(img_std, y, windowSize=patch_size)
    # X_spe, _ = get_pixel(X, y)
    X = np.concatenate([X_pca, X_pad], axis=3)
    print('Data cube X shape: ', X.shape)
    print('Data cube y shape: ', y.shape)
    print('\n... ... create train & test data ... ...')
    # train, test= splitTrainTestSetDF(X_data, test_ratio)
    # X = X.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X, y_all, test_ratio, i)
    partial_rate = 0.3
    partial_labels = generate_candidate_label(torch.tensor(ytrain), partial_rate, i)

    # 创建train_loader和 test_loader
    X = TestDS(X, y_all)
    trainset = TrainDS(Xtrain, ytrain, partial_labels)
    testset = TestDS(Xtest, ytest)

    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=BATCH_SIZE_TRAIN,
                                               shuffle=True,
                                               num_workers=0,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=BATCH_SIZE_TRAIN,
                                              shuffle=False,
                                              num_workers=0,
                                              )
    all_data_loader = torch.utils.data.DataLoader(dataset=X,
                                                  batch_size=BATCH_SIZE_TRAIN,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  )
    return train_loader, test_loader, all_data_loader, y, partial_labels


""" Training dataset"""


class TrainDS(torch.utils.data.Dataset):

    def __init__(self, Xtrain, ytrain, partial_labels):
        self.spe = Xtrain[:, 6, 6, 32:]
        self.pca = Xtrain[:, :, :, 0:32]
        self.pad = Xtrain[:, :, :, 32:]
        self.len = len(ytrain)
        self.y_data = torch.LongTensor(ytrain)
        self.y_partial = torch.FloatTensor(partial_labels)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.spe[index], self.pca[index], self.pad[index], self.y_data[index], self.y_partial[index], index

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):

    def __init__(self, Xtest, ytest):
        self.spe = Xtest[:, 6, 6, 32:]
        self.pca = Xtest[:, :, :, 0:32]
        self.pad = Xtest[:, :, :, 32:]
        self.len = len(ytest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.spe[index], self.pca[index], self.pad[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


def train(train_loader, epochs, confidence, lable_embed, num):
    # 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    # 网络放到GPU上
    net = AutoEncoder(num_class=num).to(device)
    # 分类损失函数
    criterion = partial_loss(confidence, 0.95)
    mse = nn.MSELoss()  # MSE 损失
    # gap loss
    gap_loss = semantic_loss()
    con_loss = ContrastiveLoss()
    # 初始化优化器
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # 开始训练
    train_loss_save = []
    for epoch in range(epochs):
        net.train()
        criterion.set_conf_ema_m(epoch)
        total_loss = 0
        loss_g = 0
        loss_m = 0
        loss_j = 0
        n = len(train_loader)
        for i, (spe, pca, pad, target, partialY, index) in enumerate(train_loader):
            spe, pca, pad, target = spe.to(device), pca.to(device), pad.to(device), target.to(device)
            # 正向传播 +　反向传播 + 优化
            # 通过输入得到预测的输出
            # mix the input
            # Lambda = np.random.beta(8.0, 8.0)
            # idx_rp = torch.randperm(128)
            # X_1_rp = spe[idx_rp]
            # X_2_rp = pca[idx_rp]
            # X_1_mix = Lambda * spe + (1 - Lambda) * X_1_rp
            # X_2_mix = Lambda * pca + (1 - Lambda) * X_2_rp
            outputs, img1, img2, f2, score = net(pca.type(torch.cuda.FloatTensor), spe.type(torch.cuda.FloatTensor),
                                          lable_embed, target, False)
            # 计算损失函数
            loss1 = criterion(outputs, index)
            loss2 = gap_loss(outputs, partialY)
            loss3 = mse(img1.type(torch.cuda.FloatTensor), pca.type(torch.cuda.FloatTensor)) + mse(
                img2.type(torch.cuda.FloatTensor), spe.type(torch.cuda.FloatTensor).unsqueeze(2))
            loss4 = con_loss(outputs, f2)
            loss_match = criterion(score, index)
            # loss_match = loss_match.mean()
            # w1 = epoch / 300 * (1 - 0.1) + 0.5
            # w2 = epoch / 300 * (1 - 0.5) + 0.5
            w1 = 1
            w2 = 0.5
            # 标签消歧
            if epoch >= 30: # warm up for Indian pines data set
                criterion.confidence_update(epoch, outputs, index, partialY)

            loss = loss1 + w1 * loss2 + w2 * loss3 + loss4 + 0.1*loss_match
            # 优化器梯度归零
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()
            total_loss += loss1.item()
            loss_g += loss2
            loss_m += loss3
            loss_j += loss4
        print('[Epoch: %d]   [loss mse: %.4f]   [cls loss: %.4f],  [MaxGap loss: %.4f], [Alignment loss: %.4f]' % (
        epoch + 1,
        loss_m / n, total_loss / n, loss_g / n, loss_j / n))
        loss_p = loss_g.cpu().detach().numpy()
        train_loss_save.append(loss_p / n)

    print('Finished Training')
    # 显示train_loss,train_acc 曲线图
    # x1 = range(len(train_loss_save))
    # y1 = train_loss_save
    # x_smooth = np.linspace(min(x1), max(x1), 300)
    # y_smooth = make_interp_spline(x1, y1)(x_smooth)
    # plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)
    # plt.plot(x_smooth, y_smooth, color="steelblue", label="Max-Gap Loss", linewidth=2)
    # plt.legend(prop={"size": 12})
    # font_dict = dict(fontsize=12)
    # plt.ylabel('Loss Value', fontdict=font_dict)
    # plt.xlabel('Epoch', fontdict=font_dict)
    # plt.savefig('./loss_png/Houston_loss.png')
    # plt.show()

    return net, device


def test(device, net, test_loader, lable_embed):
    count = 0
    # 模型测试
    net.eval()
    y_pred_test = 0
    # y_prot = 0
    y_test = 0
    for spe, pca, pad, labels in test_loader:
        spe, pca, pda = spe.to(device), pca.to(device), pad.to(device)
        # outputs = net(spe.type(torch.cuda.FloatTensor), pca.type(torch.cuda.FloatTensor), pda.type(torch.cuda.FloatTensor), False)
        outputs = net(pca.type(torch.cuda.FloatTensor), spe.type(torch.cuda.FloatTensor), lable_embed, 0, True)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        # prot = np.argmax(prot.detach().cpu().numpy(), axis=1)
        if count == 0:
            y_pred_test = outputs
            # y_prot = prot
            y_test = labels
            count = 1
        else:
            y_pred_test = np.concatenate((y_pred_test, outputs))
            # y_prot = np.concatenate((y_prot, prot))
            y_test = np.concatenate((y_test, labels))

    return y_pred_test, y_test


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def acc_reports(y_test, y_pred_test, name):
    target_names = []
    if name == "Indian":
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
    if name == "Pavia":
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    if name == "Houston":
        target_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    classification = classification_report(y_test, y_pred_test, digits=4, target_names=target_names)
    oa = accuracy_score(y_test, y_pred_test)
    # oa_prot = accuracy_score(y_test, y_prot)
    confusion = confusion_matrix(y_test, y_pred_test)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred_test)

    # return classification, oa*100, oa_prot*100, confusion, each_acc*100, aa*100, kappa*100
    return classification, oa * 100, confusion, each_acc * 100, aa * 100, kappa * 100


def get_class_embedding(class_name=[], embed_dim=768):
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    logging.set_verbosity_error()
    tokenizer = BertTokenizer.from_pretrained('/home/students/doctor/2023/tianxy/Pycharm-Remote/bert-base-uncased')
    model = BertModel.from_pretrained('/home/students/doctor/2023/tianxy/Pycharm-Remote/bert-base-uncased')
    all_class_embedding = []
    for label in class_name:
        word_list = label.split(' ')  # a label with multiple words
        label_embedding = torch.zeros(embed_dim)
        for word in word_list:
            text_dict = tokenizer.encode_plus(word, add_special_tokens=True, return_attention_mask=True)
            input_ids = torch.tensor(text_dict['input_ids']).unsqueeze(0)
            token_type_ids = torch.tensor(text_dict['token_type_ids']).unsqueeze(0)
            attention_mask = torch.tensor(text_dict['attention_mask']).unsqueeze(0)
            res = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            words_vectors = res[0].detach().squeeze(0)
            wordvec = words_vectors[1]
            label_embedding += wordvec.detach().clone()
        all_class_embedding.append(label_embedding)
    class_embedding = torch.stack(all_class_embedding).to(device)
    # class_embedding = torch.nn.functional.normalize(class_embedding, dim=1).float()
    return class_embedding


if __name__ == '__main__':
    n = 1
    OA = 0
    AA = 0
    KAppa = 0
    Each_acc = []
    for i in range(n):
        setup_seed(6)  # 6 3
        data_name = "Indian"
        train_loader, test_loader, all_data_loader, y_all, partialY = create_data_loader(data_name, 6)
        # calculate confidence
        tempY = partialY.sum(dim=1).unsqueeze(1).repeat(1, partialY.shape[1])
        confidence = partialY.float() / tempY
        confidence = confidence.cuda()
        class_name = []
        num_classes = 0
        if data_name == "Indian":
            class_name = ["Alfalfa", "Corn notill", "Corn mintill", "Corn", "Grass pasture", "Grass-trees",
                          "Grass pasture mowed", "Hay windrowed", "Oats", "Soybean notill", "Soybean mintill",
                          "Soybean clean",
                          "Wheat", "Woods", "Buildings Grass Trees Drives", "Stone Steel Towers"]
            num_classes = 16
        if data_name == "Pavia":
            class_name = ["Asphalt", "Meadows", "Gravel", "Trees", "Metal sheets", "Bare soil", "Bitumen", "Bricks",
                          "Shadows"]
            num_classes = 9
        if data_name == "Houston":
            class_name = ["Healthy grass", "Stressed grass", "Synthetic grass", "Trees", "Soil", "Water", "Residential",
                          "Commercial", "Road", "Highway", "Railway", "Parking Lot 1", "Parking Lot 2",
                          "Tennis Court", "Running Track"]
            num_classes = 15

        class_vectors = get_class_embedding(class_name).cuda(6)
        tic1 = time.perf_counter()
        net, device = train(train_loader, epochs=100, confidence=confidence, lable_embed=class_vectors, num=num_classes)
        # 只保存模型参数
        # torch.save(net.state_dict(), 'cls_params/params.pth')
        toc1 = time.perf_counter()
        tic2 = time.perf_counter()
        y_pred_test, y_test = test(device, net, test_loader, lable_embed=class_vectors)
        toc2 = time.perf_counter()
        # 评价指标
        classification, oa, confusion, each_acc, aa, kappa = acc_reports(y_test, y_pred_test, name=data_name)
        OA += oa
        if i == 0:
            Each_acc = each_acc
        else:
            Each_acc = Each_acc + each_acc
        KAppa = KAppa + kappa
        AA += aa
        # classification = str(classification)
        Training_Time = toc1 - tic1
        Test_time = toc2 - tic2
    file_name = "cls_result/correct/classification.txt"
    with open(file_name, 'w') as x_file:
        x_file.write('{} Training_Time (s)'.format(Training_Time))
        x_file.write('\n')
        x_file.write('{} Test_time (s)'.format(Test_time))
        x_file.write('\n')
        x_file.write('{} Kappa accuracy (%)'.format(KAppa / n))
        x_file.write('\n')
        x_file.write('{} Overall accuracy (%)'.format(OA / n))
        x_file.write('\n')
        # x_file.write('{} Prototype accuracy (%)'.format(oa_prot))
        # x_file.write('\n')
        x_file.write('{} Average accuracy (%)'.format(AA / n))
        x_file.write('\n')
        x_file.write('{} Each accuracy (%)'.format(Each_acc / n))
        x_file.write('\n')
        # x_file.write('{}'.format(classification))
        # x_file.write('\n')
        # x_file.write('{}'.format(confusion))

    # get_cls_map.get_cls_map(net, device, all_data_loader, y_all, class_vectors)
