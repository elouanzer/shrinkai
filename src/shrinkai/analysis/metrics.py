from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class BaseFeatureMetric(ABC):
    """Abstract base class for all feature alignment metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Display name of the metric (e.g., 'CKA', 'RSA')."""
        pass

    @abstractmethod
    def compute(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> float:
        """Computes the alignment score between two feature maps.

        Args:
            student_features: Tensor of shape [Batch, ...].
            teacher_features: Tensor of shape [Batch, ...].

        Returns:
            float: A score indicating representation similarity.
        """
        pass


class CKAMetric(BaseFeatureMetric):
    """Linear Centered Kernel Alignment (CKA).

    Measures the similarity of representations across models with different
    architectures or channel dimensions. A score of 1.0 means identical
    representational geometry; 0.0 means completely orthogonal.
    """

    @property
    def name(self) -> str:
        return "CKA (Linear)"

    def compute(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> float:
        if student_features.dim() > 2:
            student_features = student_features.view(student_features.size(0), -1)
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.view(teacher_features.size(0), -1)

        s_centered = student_features - student_features.mean(dim=0, keepdim=True)
        t_centered = teacher_features - teacher_features.mean(dim=0, keepdim=True)

        s_gram = s_centered @ s_centered.t()
        t_gram = t_centered @ t_centered.t()

        hsic = torch.sum(s_gram * t_gram)
        norm_s = torch.sqrt(torch.sum(s_gram * s_gram))
        norm_t = torch.sqrt(torch.sum(t_gram * t_gram))

        if norm_s == 0 or norm_t == 0:
            return 0.0

        cka = hsic / (norm_s * norm_t)
        return cka.item()


class RSAMetric(BaseFeatureMetric):
    """Representational Similarity Analysis (RSA) using Pearson correlation.

    Measures if the relative distances between samples in a batch are preserved
    between the teacher and the student, regardless of their hidden dimension sizes.
    """

    @property
    def name(self) -> str:
        return "RSA (Pearson)"

    def compute(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> float:
        # flatten [B, C, H, W] -> [B, Features]
        s_flat = student_features.view(student_features.size(0), -1)
        t_flat = teacher_features.view(teacher_features.size(0), -1)

        s_sim = F.cosine_similarity(s_flat.unsqueeze(1), s_flat.unsqueeze(0), dim=-1)
        t_sim = F.cosine_similarity(t_flat.unsqueeze(1), t_flat.unsqueeze(0), dim=-1)

        # top of matrix
        idx = torch.triu_indices(s_sim.size(0), s_sim.size(1), offset=1)
        s_pdist = s_sim[idx[0], idx[1]]
        t_pdist = t_sim[idx[0], idx[1]]

        # pearson correlation
        s_mean, t_mean = s_pdist.mean(), t_pdist.mean()
        s_centered, t_centered = s_pdist - s_mean, t_pdist - t_mean

        cov = (s_centered * t_centered).sum()
        var_s = (s_centered**2).sum()
        var_t = (t_centered**2).sum()

        if var_s == 0 or var_t == 0:
            return 0.0

        pearson_corr = cov / torch.sqrt(var_s * var_t)
        return pearson_corr.item()


class SpatialAttentionMetric(BaseFeatureMetric):
    """Spatial Attention Transfer similarity.

    Collapses the channel dimension to measure if the student and teacher
    activate on the same spatial regions of the input (e.g., the foreground object).
    """

    @property
    def name(self) -> str:
        return "Spatial Attention"

    def compute(self, student_features: torch.Tensor, teacher_features: torch.Tensor) -> float:
        # only 4D tensors (CNNs: Batch, Channel, Height, Width)
        if student_features.dim() != 4 or teacher_features.dim() != 4:
            return 0.0

        # spatial attention map
        s_att = torch.sum(torch.abs(student_features), dim=1, keepdim=True)
        t_att = torch.sum(torch.abs(teacher_features), dim=1, keepdim=True)

        # align spatial dimensions
        if s_att.shape[2:] != t_att.shape[2:]:
            s_att = F.adaptive_avg_pool2d(s_att, output_size=t_att.shape[2:])

        # normalization
        s_att = F.normalize(s_att.view(s_att.size(0), -1), p=2, dim=1)
        t_att = F.normalize(t_att.view(t_att.size(0), -1), p=2, dim=1)

        cos_sim = (s_att * t_att).sum(dim=1).mean()
        return cos_sim.item()


FEATURE_METRICS = {
    "cka": CKAMetric(),
    "rsa": RSAMetric(),
    "attention": SpatialAttentionMetric(),
}
