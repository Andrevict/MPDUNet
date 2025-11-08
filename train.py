from util.evaluate import *
from util.loss import BCEDiceLoss

import os
import random
import time
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch.optim
import torch.nn as nn

# 设置代码训练种子
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def train_val(model, train_loader, val_loader, device, epochs, optimizer, scheduler, loss, save_path,
              save_current_model=False, save_evaluation=False, early_stop=-1):
    """
    train model and testify model
    :param model: pytorch model
    :param train_loader: train loader
    :param val_loader: val loader or test loader
    :param device: 'cpu' or 'cuda', string
    :param epochs:
    :param optimizer:
    :param scheduler: it can reduce the lr of optim
    :param loss: loss function
    :param save_path: best model save path
    :param save_current_model: save current model, saved model path is save_path
    :param save_evaluation: True or False, bool
    :param early_stop: if the loss don't decrease by [early_stop] epochs, training.html will stop to avoid overfit
                        if early_stop < 0 then early stop method don't work
    :return: None
    """
    model = model.to(device)

    count = early_stop

    # 记录指标
    metric = dict()

    # 记录最小的val loss
    min_val_loss = 65535

    # 引入评估指标
    train_metric = Metrics()
    test_metric = Metrics()

    # 开始训练
    for epoch in range(epochs):
        # 评估参数
        train_loss = 0
        val_loss = 0
        # 评估指标重置
        train_metric.reset()
        test_metric.reset()

        # 记录时间
        start_time = time.time()
        model.train()  # 模型调整为训练模式
        for x_batch, y_batch in tqdm(train_loader, desc='训练中: '):
            # 将数据存储到GPU上
            x, y = x_batch.to(device), y_batch.to(device)
            y_pred = model(x)

            train_batch_loss = loss(y_pred, y)

            # 反向传播
            train_batch_loss.backward()
            optimizer.step()  # 更新模型
            optimizer.zero_grad()  # 优化器梯度清零

            # 模型评价
            train_loss += train_batch_loss.cpu().item()
            train_metric.update(y, y_pred > 0.5)            # threshold > 0.5 -> 1

        train_loss = train_loss / train_loader.__len__()

        if scheduler:
            # 跟据测试集修改学习率
            scheduler.step(train_loss)

        # 记录每轮批次计算指标
        metric[f"train[{epoch + 1:03}/{epochs:03}]"] = {"Loss": train_loss, "metric": train_metric.get_metrics()}

        # 验证模型
        model.eval()  # 调整为验证模式
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc='测试中: '):
                x, y = x_batch.to(device), y_batch.to(device)
                y_pred = model(x)

                # 模型衡量记录
                val_batch_loss = loss(y_pred, y)

                # 模型评价
                val_loss += val_batch_loss.cpu().item()
                test_metric.update(y, y_pred > 0.5)         # threshold > 0.5 -> 1

        val_loss = val_loss / val_loader.__len__()
        metric[f"test[{epoch + 1:03}/{epochs:03}]"] = {"Loss": val_loss, "metric": test_metric.get_metrics()}

        if save_current_model:
            torch.save(model, os.path.join(save_path, f"Current_Model.pth"))

        # 保存模型
        if min_val_loss > val_loss:
            # 保存模型的定义
            torch.save(model, os.path.join(save_path, f"Best_Model.pth"))
        else:
            count -= 1
            if count >= 0:
                print(f"早停于第{epoch}轮（验证损失{early_stop}轮无提升）")
                break  # 停止训练

        if save_evaluation:
            import json
            with open(os.path.join(save_path, "evaluation.json"), 'a') as f:
                json.dump(metric, f)

        print(f"\n[{epoch + 1:03}/{epochs:03}]  Time {time.time() - start_time:.2f} sec(s)")
        print("训练集评估: " + f"损失函数: {train_loss:.8}" + train_metric.__str__())
        print("测试集评估: " + f"损失函数: {val_loss:.8}" + test_metric.__str__())
        print("训练信息已保存至: " + save_path)


def load_model(model_path: str) -> nn.Module:
    """
    resume to train model
    :param model_path: model path(model file name *.pth)
    :param epoch: epoch when the model training.html stop
    :param epochs: epoch sum
    :return: nn.Module
    """
    return torch.load(model_path)


def predict(model, image):
    """
    predict image by using the model
    :param model:
    :param image:
    :return:
    """
    return model(image)


# 修改超参数，只修改以下内容，不要修改之上的内容
if __name__ == "__main__":
    from model.wnetplus import WnetPlus
    from dataset.ISIC2018 import load_ISIC2018

    seed_everything(22)

    # 图形增广 训练集
    train_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    # 测试集 一般不做增广 转换为 torch.Tensor 变量即可
    test_transforms = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
    ])
    # 加载数据集
    train_dataset, test_dataset = load_TN3K("/data/公共数据集/TN3K", train_transforms, test_transforms)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    save_path = "/home/zlh/save/model/WNetPlus"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    learning_rate = 0.0001
    epochs = 100
    model = WnetPlus(3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
    # 学习率退火 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=15,
                                                           threshold=0.0001, min_lr=0)  # 固定步长衰减
    loss = BCEDiceLoss()
    print(f"开始训练：{model.__class__.__name__}\n")
    train_val(model, train_loader, test_loader, device, epochs, optimizer, scheduler, loss, save_path,
              save_current_model=True, save_evaluation=True)
