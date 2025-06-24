import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    lovasz_hinge = None  # 确保即使未安装也能运行

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss', 'FocalDiceLoss', 'TverskyLoss', 'ComboLoss', 'BoundaryAwareDiceLoss',
           'AdaptiveDiceLoss']


class BCEDiceLoss(nn.Module):
    # 保持原有实现不变
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    # 保持原有实现不变
    def __init__(self):
        super().__init__()
        if lovasz_hinge is None:
            raise ImportError("需要安装LovaszSoftmax库: https://github.com/bermanmaxim/LovaszSoftmax")

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        return lovasz_hinge(input, target, per_image=True)


class FocalDiceLoss(nn.Module):
    # 保持原有实现不变
    def __init__(self, alpha=0.25, gamma=2, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        prob = torch.sigmoid(input)
        p_t = prob * target + (1 - prob) * (1 - target)
        focal_loss = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss

        focal_loss = focal_loss.mean()

        input_sigmoid = torch.sigmoid(input)
        num = target.size(0)
        input_flat = input_sigmoid.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (input_flat * target_flat).sum(1)
        dice_loss = 1 - ((2. * intersection + self.smooth) /
                         (input_flat.sum(1) + target_flat.sum(1) + self.smooth)).mean()

        return focal_loss + dice_loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        """
        Tversky损失（Dice变体，重点惩罚假阳性/假阴性）
        适合皮肤病小目标分割（控制漏检/误检比例）

        :param alpha: 假阳性惩罚权重（FP）
        :param beta: 假阴性惩罚权重（FN）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, input, target):
        input = torch.sigmoid(input)
        target = target.float()

        # 计算TP, FP, FN
        tp = (input * target).sum()
        fp = (input * (1 - target)).sum()
        fn = ((1 - input) * target).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky


class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, ce_ratio=0.5, smooth=1e-5):
        """
        组合损失（交叉熵+改进Dice）
        适合类别极度不平衡的皮肤病数据

        :param alpha: Dice部分的权重
        :param ce_ratio: 交叉熵的权重系数
        """
        super().__init__()
        self.alpha = alpha
        self.ce_ratio = ce_ratio
        self.smooth = smooth

    def forward(self, input, target):
        # Dice部分
        input_sigmoid = torch.sigmoid(input)
        intersection = (input_sigmoid * target).sum()
        union = input_sigmoid.sum() + target.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # 交叉熵部分（带权重）
        bce = F.binary_cross_entropy_with_logits(input, target)
        balanced_bce = bce * (self.ce_ratio + (1 - self.ce_ratio) * target.mean())

        return self.alpha * (1 - dice) + (1 - self.alpha) * balanced_bce


class BoundaryAwareDiceLoss(nn.Module):
    def __init__(self, kernel_size=3, smooth=1e-5):
        """
        边界感知Dice损失（增强边缘区域优化）
        适合皮肤病边界模糊的场景（如湿疹、银屑病）

        :param kernel_size: 边界检测核大小
        """
        super().__init__()
        # 拉普拉斯边缘检测核
        self.kernel = torch.tensor([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).view(1, 1, kernel_size, kernel_size)

    def forward(self, input, target):
        input_sigmoid = torch.sigmoid(input)
        target = target.float()

        # 提取边界
        input_edge = F.conv2d(input_sigmoid.unsqueeze(1), self.kernel.to(input.device), padding=1)
        target_edge = F.conv2d(target.unsqueeze(1), self.kernel.to(target.device), padding=1)

        # 计算边界区域的Dice
        edge_intersection = (input_edge * target_edge).sum()
        edge_union = input_edge.sum() + target_edge.sum()
        edge_dice = (2. * edge_intersection + 1e-5) / (edge_union + 1e-5)

        # 计算整体Dice
        intersection = (input_sigmoid * target).sum()
        union = input_sigmoid.sum() + target.sum()
        overall_dice = (2. * intersection + 1e-5) / (union + 1e-5)

        # 边界增强（边界损失占30%权重）
        return 0.7 * (1 - overall_dice) + 0.3 * (1 - edge_dice)


class AdaptiveDiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        """
        自适应Dice损失（根据类别频率调整权重）
        适合病变区域占比动态变化的皮肤病数据集
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, input, target):
        input_sigmoid = torch.sigmoid(input)
        target = target.float()

        # 计算正类频率
        pos_freq = target.mean()
        neg_freq = 1 - pos_freq

        # 自适应权重（病变区域越小权重越高）
        w = 1 / (pos_freq ** 2 + self.smooth)

        # 带权重的Dice计算
        intersection = (w * input_sigmoid * target).sum()
        union = (w * input_sigmoid).sum() + (w * target).sum()
        return 1 - (2. * intersection + self.smooth) / (union + self.smooth)
