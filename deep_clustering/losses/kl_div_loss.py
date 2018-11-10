"""
custom loss implementation for KL-divergence loss
"""
import torch


class ClusterAssignmentHardeningLoss(torch.nn.Module):
    """
        KL-divergence loss implementaion used for training
    """
    NU = 1

    def __init__(self):
        super(ClusterAssignmentHardeningLoss, self).__init__()

    def forward(self, encode_output, centroids):

        # I don't use norm. Norm is more memory-efficient, but possibly less numerically stable in backward

        q_raw = (1 + ((encode_output.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(2) / self.NU) ** (-(self.NU + 1) / 2)  # i, j
        q_sum = q_raw.sum(1, keepdim=True)  # i, 1 --> will be broadcast
        q = q_raw / q_sum  # i, j

        p_raw = q ** 2 / q.sum(0, keepdim=True)  # i, j
        p_sum = p_raw.sum(1, keepdim=True)  # 1, j --> will be broadcast
        p = p_raw / p_sum

        kl_div = (p * (p.clamp(min=1e-7).log() - q.clamp(min=1e-7).log())).sum()

        return kl_div
