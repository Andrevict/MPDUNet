from typing import Tuple

import torch

class Metrics:
    def __init__(self):
        """初始化二分类指标计算器"""
        self.FN = 0
        self.TN = 0
        self.FP = 0
        self.TP = 0
        self.total_samples = 0
        self.reset()  # 初始化混淆矩阵元素

    def reset(self) -> None:
        """重置所有累积的指标，为新的计算周期做准备"""
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        self.total_samples = 0

    def update(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> None:
        """
        更新批次数据，计算当前批次的混淆矩阵并累加

        参数:
            y_true: 真实标签，形状为(batch_size,)或(batch_size, 1)，值为0或1
            y_pred: 预测结果，可以是:
                    - 概率值，形状为(batch_size,)或(batch_size, 1)
                    - 类别标签，形状为(batch_size,)或(batch_size, 1)，值为0或1
                    - 概率分布，形状为(batch_size, 2)
        """
        # 确保输入是张量并移至CPU
        if not isinstance(y_true, torch.Tensor):
            y_true = torch.tensor(y_true, dtype=torch.float32)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = torch.tensor(y_pred, dtype=torch.float32)

        self.total_samples += len(y_true)

        y_true = y_true.cpu().flatten()
        y_pred = y_pred.cpu().flatten()

        # 计算当前批次的混淆矩阵元素
        batch_TP = torch.sum((y_pred == 1) & (y_true == 1)).item()
        batch_FP = torch.sum((y_pred == 1) & (y_true == 0)).item()
        batch_TN = torch.sum((y_pred == 0) & (y_true == 0)).item()
        batch_FN = torch.sum((y_pred == 0) & (y_true == 1)).item()

        # 累加至总计数
        self.TP += batch_TP
        self.FP += batch_FP
        self.TN += batch_TN
        self.FN += batch_FN


    def compute_matrix(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """返回当前累积的混淆矩阵"""
        return (
            (self.TP, self.FN),
            (self.FP, self.TN)
        )

    def compute_accuracy(self) -> float:
        """计算准确率"""
        return (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)

    def compute_precision(self) -> float:
        """计算精确率"""
        return self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0

    def compute_recall(self) -> float:
        """计算召回率"""
        return self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0

    def compute_f1(self) -> float:
        """计算F1分数"""
        precision = self.compute_precision()
        recall = self.compute_recall()
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    def compute_dice(self) -> float:
        """计算Dice系数"""
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN) if (2 * self.TP + self.FP + self.FN) > 0 else 0.0

    def compute_iou(self) -> float:
        """计算IoU（交并比）"""
        return self.TP / (self.TP + self.FP + self.FN) if (self.TP + self.FP + self.FN) > 0 else 0.0

    def get_metrics(self) -> dict:
        """返回所有指标的字典"""
        return {
            'confusion_matrix': self.compute_matrix(),
            'accuracy': self.compute_accuracy(),
            'precision': self.compute_precision(),
            'recall': self.compute_recall(),
            'f1': self.compute_f1(),
            'dice': self.compute_dice(),
            'iou': self.compute_iou(),
            'total_samples': self.total_samples
        }

    def __str__(self) -> str:
        """返回格式化的指标字符串"""
        metrics = self.get_metrics()

        # 构建指标字符串
        metrics_str = (
            f"  样本总数: {metrics['total_samples']}"
            f"  准确率:   {metrics['accuracy']:.4f}"
            f"  精确率:   {metrics['precision']:.4f}"
            f"  召回率:   {metrics['recall']:.4f}"
            f"  F1分数:   {metrics['f1']:.4f}"
            f"  Dice系数: {metrics['dice']:.4f}"
            f"  IoU:     {metrics['iou']:.4f}"
        )

        return f"{metrics_str}"

    def __repr__(self) -> str:
        """返回对象的正式表示"""
        return f"Metrics(total_samples={self.total_samples})"

