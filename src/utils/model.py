import torch
from torch import nn
import torch.nn.functional as F
from utils.sam import SAM
from efficientnet_pytorch import EfficientNet
import pdb
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        :param alpha: Weighting factor for positive/negative classes.
        :param gamma: Focusing parameter to reduce the impact of easy samples.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convert inputs to probabilities using softmax
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = torch.zeros_like(probs).scatter_(1, targets.view(-1, 1), 1)

        # Calculate the focal loss
        focal_weight = (1 - probs).pow(self.gamma)  # (1 - p_t)^gamma
        focal_loss = -self.alpha * focal_weight * targets_one_hot * probs.log()

        if self.reduction == 'mean':
            return focal_loss.sum() / inputs.size(0)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# ArcFace Loss implementation
class ArcFaceLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, margin=0.5, scale=30):
        super(ArcFaceLoss, self).__init__()
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        theta = torch.acos(cosine.clamp(-1 + 1e-7, 1 - 1e-7))
        target_logit = torch.cos(theta + self.margin)

        # One-hot encoding for the target class
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Combine target logits with the non-target logits
        output = one_hot * target_logit + (1.0 - one_hot) * cosine
        output *= self.scale
        return output


# MixStyle implementation
class MixStyle(nn.Module):
    def __init__(self, p=0.0000001, alpha=0):
        super(MixStyle, self).__init__()
        self.p = p
        self.alpha = alpha

    def forward(self, x):

        if not self.training or torch.rand(1).item() > self.p:
            return x
        pdb.set_trace()
        batch_size = x.size(0)
        # Calculate mean and variance
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma = torch.sqrt(x.var(dim=[2, 3], keepdim=True) + 1e-6)
        # Shuffle within the batch
        perm = torch.randperm(batch_size)
        mu_perm = mu[perm]
        sigma_perm = sigma[perm]
        # Mix the statistics
        mix_factor = torch.rand(batch_size, 1, 1, 1, device=x.device).uniform_(0, 1) * self.alpha
        mu_mixed = mu * (1 - mix_factor) + mu_perm * mix_factor
        sigma_mixed = sigma * (1 - mix_factor) + sigma_perm * mix_factor
        return (x - mu) / sigma * sigma_mixed + mu_mixed


#Detector Model with ArcFaceLoss and MixStyle
class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        self.net = EfficientNet.from_pretrained("efficientnet-b4", advprop=True)

        # MixStyle layer
        self.mixstyle = MixStyle()

        # Embedding and classifier
        self.embedding_layer = nn.Linear(1792, 512)  # EfficientNet-B4 outputs 1792 features
        self.classifier = nn.Linear(512, 2)  # Classifier for final output

        # Losses
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0)
        self.cel = nn.CrossEntropyLoss()
        self.arcface_loss = ArcFaceLoss(embedding_size=512, num_classes=2, margin=0.5, scale=30)

        # SAM optimizer
        self.optimizer = SAM(self.parameters(), torch.optim.SGD, lr=0.001, momentum=0.9)

    def forward(self, x):
        # Extract features using EfficientNet
        x = self.net.extract_features(x)
        x = self.mixstyle(x)  # Apply MixStyle to intermediate feature maps
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)  # Global average pooling
        embeddings = self.embedding_layer(x)  # Get embeddings
        logits = self.classifier(embeddings)  # Final classifier
        return logits, embeddings

    def training_step(self, x, target):
        for i in range(2):
            logits, embeddings = self(x)  # Forward pass to get both logits and embeddings

            if i == 0:
                pred_first = logits  # Save the first prediction for SAM

            # CrossEntropy Loss on logits
            loss_cls = self.cel(logits, target)
            loss_focalloss = self.focal_loss(logits, target)

            # ArcFace Loss on embeddings
            arcface_logits = self.arcface_loss(embeddings, target)
            loss_arcface_cls = self.cel(arcface_logits, target)  # Apply cross-entropy on ArcFace logits

            # Total loss: combine classification loss and ArcFace loss
            loss = loss_cls + loss_arcface_cls+0.0000001*loss_focalloss
            # loss = loss_cls
            self.optimizer.zero_grad()
            loss.backward()

            if i == 0:
                self.optimizer.first_step(zero_grad=True)
            else:
                self.optimizer.second_step(zero_grad=True)

        return pred_first




