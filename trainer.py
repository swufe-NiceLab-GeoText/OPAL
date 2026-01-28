from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .opal_loss import compute_eif_variance


def _flatten_grads(grads: Tuple[Optional[torch.Tensor], ...], params: List[torch.nn.Parameter]) -> torch.Tensor:
    flats: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            flats.append(torch.zeros_like(p).view(-1))
        else:
            flats.append(g.contiguous().view(-1))
    return torch.cat(flats)


def _unflatten_grads(flat: torch.Tensor, params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    offset = 0
    for p in params:
        numel = p.numel()
        out.append(flat[offset : offset + numel].view_as(p))
        offset += numel
    return out


def fill_none_grads(grads: Tuple[Optional[torch.Tensor], ...], params: List[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]


def orthogonalize_with_clamp(
    g_causal: Tuple[Optional[torch.Tensor], ...],
    g_prop: Tuple[Optional[torch.Tensor], ...],
    params: List[torch.nn.Parameter],
    s_max: float = 5.0,
    lambda_orth: float = 1.0,
    eps: float = 1e-8,
    tiny: float = 1e-12,
) -> Tuple[List[torch.Tensor], Dict[str, float]]:
    g_c_flat = _flatten_grads(g_causal, params)
    g_p_flat = _flatten_grads(g_prop, params)

    norm_c = g_c_flat.norm()
    norm_p = g_p_flat.norm()

    if norm_p < tiny:
        stats = {
            "cos_before": 0.0,
            "cos_after": 0.0,
            "norm_before": float(norm_c.item()),
            "norm_after": float(norm_c.item()),
            "scale": 1.0,
            "clamp_triggered": False,
            "skipped": True,
            "lambda_orth": float(lambda_orth),
        }
        return fill_none_grads(g_causal, params), stats

    dot = (g_c_flat * g_p_flat).sum()
    cos_before = dot / (norm_c * norm_p + eps)

    proj_coef = dot / (norm_p**2 + eps)
    g_c_orth = g_c_flat - lambda_orth * proj_coef * g_p_flat

    norm_c_orth = g_c_orth.norm()
    cos_after = (g_c_orth * g_p_flat).sum() / (norm_c_orth * norm_p + eps)

    if norm_c_orth < eps:
        scale = 1.0
        clamp_triggered = False
    else:
        raw_scale = norm_c / (norm_c_orth + eps)
        scale = min(float(raw_scale.item()), float(s_max))
        clamp_triggered = float(raw_scale.item()) > float(s_max)
        g_c_orth = g_c_orth * scale

    stats = {
        "cos_before": float(cos_before.item()),
        "cos_after": float(cos_after.item()),
        "norm_before": float(norm_c.item()),
        "norm_after": float(norm_c_orth.item()),
        "scale": float(scale),
        "clamp_triggered": bool(clamp_triggered),
        "skipped": False,
        "lambda_orth": float(lambda_orth),
    }

    return _unflatten_grads(g_c_orth, params), stats


class OPALTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: torch.device | str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device

        self.alpha = float(config.get("alpha", 1.0))
        self.lambda_eif = float(config.get("lambda_eif", 1.0))
        self.delta = float(config.get("delta", 0.25))
        self.lambda_orth = float(config.get("lambda_orth", 1.0))
        self.s_max = float(config.get("s_max", 5.0))
        self.warmup_epochs = int(config.get("warmup_epochs", 10))
        self.eps_g = float(config.get("eps_g", 1e-3))
        self.eps = float(config.get("eps", 1e-8))

        self._setup_param_groups()

    def _setup_param_groups(self) -> None:
        self.trunk_params: List[nn.Parameter] = []
        self.g_head_params: List[nn.Parameter] = []
        self.q_head_params: List[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if "g_head" in name or "propensity" in name:
                self.g_head_params.append(param)
            elif "Q0" in name or "Q1" in name or "outcome" in name:
                self.q_head_params.append(param)
            else:
                self.trunk_params.append(param)

        if not self.trunk_params:
            self.trunk_params = list(self.model.parameters())
            self.g_head_params = []
            self.q_head_params = []

    def _set_grad(self, params: List[nn.Parameter], grads: List[torch.Tensor]) -> None:
        for p, g in zip(params, grads):
            if p.grad is None:
                p.grad = g.clone()
            else:
                p.grad.copy_(g)

    def _get_logit(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "logit" in outputs:
            return outputs["logit"].view(-1)
        if "g" in outputs:
            g = outputs["g"].view(-1).clamp(1e-6, 1 - 1e-6)
            return torch.log(g / (1 - g))
        raise ValueError("Model must output 'logit' or 'g'")

    def train_step_warmup(
        self,
        textual: torch.Tensor,
        visual: torch.Tensor,
        structured: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
    ) -> Dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(textual, visual, structured)
        logit = self._get_logit(outputs)
        Q0 = outputs["Q0"].view(-1)
        Q1 = outputs["Q1"].view(-1)

        T = T.float().view(-1)
        Y = Y.float().view(-1)

        Y_pred = T * Q1 + (1 - T) * Q0
        loss_outcome = nn.functional.mse_loss(Y_pred, Y)
        loss_propensity = nn.functional.binary_cross_entropy_with_logits(logit, T)

        loss_total = loss_outcome + self.alpha * loss_propensity
        loss_total.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return {
            "loss_total": float(loss_total.item()),
            "loss_outcome": float(loss_outcome.item()),
            "loss_propensity": float(loss_propensity.item()),
            "loss_eif_var": 0.0,
        }

    def train_step_full(
        self,
        textual: torch.Tensor,
        visual: torch.Tensor,
        structured: torch.Tensor,
        T: torch.Tensor,
        Y: torch.Tensor,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(textual, visual, structured)
        logit = self._get_logit(outputs)
        Q0 = outputs["Q0"].view(-1)
        Q1 = outputs["Q1"].view(-1)

        T = T.float().view(-1)
        Y = Y.float().view(-1)

        Y_pred = T * Q1 + (1 - T) * Q0
        loss_outcome = nn.functional.mse_loss(Y_pred, Y)
        loss_propensity = nn.functional.binary_cross_entropy_with_logits(logit, T)

        loss_eif_var, _ = compute_eif_variance(
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

        loss_causal = loss_outcome + self.lambda_eif * loss_eif_var

        if self.trunk_params:
            g_prop_trunk = torch.autograd.grad(
                self.alpha * loss_propensity,
                self.trunk_params,
                retain_graph=True,
                allow_unused=True,
            )
            g_causal_trunk = torch.autograd.grad(
                loss_causal,
                self.trunk_params,
                retain_graph=True,
                allow_unused=True,
            )

            g_causal_trunk_orth, grad_stats = orthogonalize_with_clamp(
                g_causal_trunk,
                g_prop_trunk,
                self.trunk_params,
                s_max=self.s_max,
                lambda_orth=self.lambda_orth,
                eps=self.eps,
            )

            g_prop_trunk_filled = fill_none_grads(g_prop_trunk, self.trunk_params)
            g_trunk_final = [gp + gc for gp, gc in zip(g_prop_trunk_filled, g_causal_trunk_orth)]
            self._set_grad(self.trunk_params, g_trunk_final)
        else:
            grad_stats = {
                "cos_before": 0.0,
                "cos_after": 0.0,
                "norm_before": 0.0,
                "norm_after": 0.0,
                "scale": 1.0,
                "clamp_triggered": False,
                "skipped": True,
                "lambda_orth": float(self.lambda_orth),
            }

        if self.g_head_params:
            g_ghead = torch.autograd.grad(
                self.alpha * loss_propensity + self.lambda_eif * loss_eif_var,
                self.g_head_params,
                retain_graph=True,
                allow_unused=True,
            )
            self._set_grad(self.g_head_params, fill_none_grads(g_ghead, self.g_head_params))

        if self.q_head_params:
            g_qhead = torch.autograd.grad(
                loss_causal,
                self.q_head_params,
                retain_graph=False,
                allow_unused=True,
            )
            self._set_grad(self.q_head_params, fill_none_grads(g_qhead, self.q_head_params))

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        loss_dict = {
            "loss_total": float((loss_outcome + self.alpha * loss_propensity + self.lambda_eif * loss_eif_var).item()),
            "loss_outcome": float(loss_outcome.item()),
            "loss_propensity": float(loss_propensity.item()),
            "loss_eif_var": float(loss_eif_var.item()),
        }
        return loss_dict, grad_stats

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        is_warmup = epoch < self.warmup_epochs

        metrics: Dict[str, float] = {
            "loss_total": 0.0,
            "loss_outcome": 0.0,
            "loss_propensity": 0.0,
            "loss_eif_var": 0.0,
        }

        n = 0
        for batch in loader:
            if len(batch) == 5:
                textual, visual, structured, T, Y = batch
            elif len(batch) == 4:
                textual, visual, T, Y = batch
                structured = torch.zeros(textual.size(0), 1)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            textual = textual.to(self.device)
            visual = visual.to(self.device)
            structured = structured.to(self.device)
            T = T.to(self.device)
            Y = Y.to(self.device)

            if is_warmup:
                step = self.train_step_warmup(textual, visual, structured, T, Y)
            else:
                step, _ = self.train_step_full(textual, visual, structured, T, Y)

            for k in metrics:
                metrics[k] += float(step[k])
            n += 1

        if n > 0:
            for k in metrics:
                metrics[k] /= n

        return metrics

    def fit(self, loader: DataLoader, epochs: int) -> List[Dict[str, float]]:
        history: List[Dict[str, float]] = []
        for epoch in range(int(epochs)):
            history.append(self.train_epoch(loader, epoch))
        return history

    @torch.no_grad()
    def evaluate_ate(self, loader: DataLoader) -> float:
        self.model.eval()
        all_tau: List[torch.Tensor] = []
        for batch in loader:
            if len(batch) == 5:
                textual, visual, structured, _, _ = batch
            elif len(batch) == 4:
                textual, visual, _, _ = batch
                structured = torch.zeros(textual.size(0), 1)
            else:
                raise ValueError(f"Unexpected batch size: {len(batch)}")

            textual = textual.to(self.device)
            visual = visual.to(self.device)
            structured = structured.to(self.device)

            outputs = self.model(textual, visual, structured)
            Q0 = outputs["Q0"].view(-1)
            Q1 = outputs["Q1"].view(-1)
            all_tau.append(Q1 - Q0)

        self.model.train()
        return torch.cat(all_tau).mean().item()


