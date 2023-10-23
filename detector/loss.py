import torch as tr, torch.nn as nn

class AsymmetricLoss(nn.Module):
    """https://arxiv.org/abs/2009.14119"""

    def __init__(
        self,
        gamma_neg=4,
        gamma_pos=1,
        clip=0.05,
        eps=1e-8,
        no_grad=True,
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.no_grad = no_grad

    def forward(self, x, y):
        xs_pos = tr.sigmoid(x)
        xs_neg = 1 - xs_pos

        if self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        loss_pos = y * tr.log(xs_pos.clamp(min=self.eps))
        loss_neg = (1 - y) * tr.log(xs_neg.clamp(min=self.eps))
        loss = loss_pos + loss_neg

        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.no_grad: tr.set_grad_enabled(False)
            pt = (xs_pos * y) + (xs_neg * (1 - y))
            one_sided_gamma = (self.gamma_pos * y) + (self.gamma_neg * (1 - y))
            one_sided_w = tr.pow(1 - pt, one_sided_gamma)
            if self.no_grad: tr.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum() / 1000
