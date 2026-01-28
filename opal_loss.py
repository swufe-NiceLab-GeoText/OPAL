from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_eif_variance(
    logit: torch.Tensor,
    Q0: torch.Tensor,
    Q1: torch.Tensor,
    T: torch.Tensor,
    Y: torch.Tensor,
    delta: float = 0.25,
    eps_g: float = 1e-3,
    eps: float = 1e-8,
    return_diagnostics: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor | float]]:
    logit = logit.float().view(-1)
    Q0 = Q0.float().view(-1)
    Q1 = Q1.float().view(-1)
    T = T.float().view(-1)
    Y = Y.float().view(-1)

    g = torch.sigmoid(logit).clamp(eps_g, 1 - eps_g)
    tau_i = Q1 - Q0

    ipw_treated = T / g * (Y - Q1)
    ipw_control = (1 - T) / (1 - g) * (Y - Q0)
    psi = tau_i + ipw_treated - ipw_control

    w_raw = torch.min(g / delta, (1 - g) / delta)
    w = torch.clamp(w_raw, max=1.0).detach()

    w_sum = w.sum() + eps
    psi_bar_w = ((w * psi).sum() / w_sum).detach()

    loss_eif_var = (w * (psi - psi_bar_w) ** 2).sum() / w_sum

    if not return_diagnostics:
        return loss_eif_var

    ess = (w.sum() ** 2) / ((w**2).sum() + eps)
    diagnostics: Dict[str, torch.Tensor | float] = {
        "psi_mean": float(psi_bar_w.item()),
        "psi_var": float(loss_eif_var.item()),
        "w_mean": float(w.mean().item()),
        "ess": float(ess.item()),
        "g_mean": float(g.mean().item()),
        "g_min": float(g.min().item()),
        "g_max": float(g.max().item()),
        "tau_mean": float(tau_i.mean().item()),
        "tau_var": float(tau_i.var().item()),
    }
    return loss_eif_var, diagnostics


class DTCOIFLoss(nn.Module):
    def __init__(
        self,
        alpha: float = 1.0,
        lambda_eif: float = 1.0,
        delta: float = 0.25,
        eps_g: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.lambda_eif = lambda_eif
        self.delta = delta
        self.eps_g = eps_g
        self.eps = eps
        self.mse = nn.MSELoss()

    def forward(
        self,
        logit: torch.Tensor,
        Q0: torch.Tensor,
        Q1: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> Dict[str, torch.Tensor]:
        logit = logit.view(-1)
        Q0 = Q0.view(-1)
        Q1 = Q1.view(-1)
        T = T.float().view(-1)
        Y = Y.float().view(-1)

        Y_pred = T * Q1 + (1 - T) * Q0
        loss_outcome = self.mse(Y_pred, Y)
        loss_propensity = F.binary_cross_entropy_with_logits(logit, T)

        if return_diagnostics:
            loss_eif_var, diagnostics = compute_eif_variance(
                logit,
                Q0,
                Q1,
                T,
                Y,
                delta=self.delta,
                eps_g=self.eps_g,
                eps=self.eps,
                return_diagnostics=True,
            )
        else:
            loss_eif_var = compute_eif_variance(
                logit,
                Q0,
                Q1,
                T,
                Y,
                delta=self.delta,
                eps_g=self.eps_g,
                eps=self.eps,
                return_diagnostics=False,
            )
            diagnostics = None

        loss_total = loss_outcome + self.alpha * loss_propensity + self.lambda_eif * loss_eif_var

        out = {
            "loss_total": loss_total,
            "loss_outcome": loss_outcome,
            "loss_propensity": loss_propensity,
            "loss_eif_var": loss_eif_var,
        }

        if return_diagnostics and diagnostics is not None:
            out["diagnostics"] = diagnostics

        return out

    @staticmethod
    def compute_ate(Q0: torch.Tensor, Q1: torch.Tensor) -> torch.Tensor:
        return (Q1 - Q0).view(-1).mean()


