# OPAL (Core Release)

OPAL is a multimodal causal effect estimation model that predicts the propensity score and potential outcomes from multimodal covariates. The core components include: (i) modality-specific encoders, (ii) bidirectional cross-modal attention for interaction modeling, and (iii) a doubly robust objective with an EIF-variance term optimized via orthogonal gradient updates to reduce task interference.

This repository intentionally includes **only the core model / loss / training logic**. Datasets, full experiment pipelines, and generated outputs are excluded.

## Requirements

Install minimal dependencies:

```bash
pip install -r requirements.txt
```

## Dataset (Expected Format)

Our paper uses semi-synthetic multimodal datasets constructed from Weibo (e.g., LFC / NZ / TGD2). This release **does not** distribute data. To use OPAL, prepare inputs in the following format:

- **Covariates**
  - `textual_features`: `FloatTensor` of shape `(B, d_t)` (e.g., CLIP text embeddings)
  - `visual_features`: `FloatTensor` of shape `(B, d_p)` (e.g., CLIP image embeddings)
  - `structured_features` (optional): `FloatTensor` of shape `(B, d_s)` (e.g., user/post metadata)
- **Treatment / Outcome**
  - `T`: `FloatTensor` of shape `(B,)`, binary treatment in `{0, 1}`
  - `Y`: `FloatTensor` of shape `(B,)`, continuous outcome

## Minimal Usage

Forward pass:

```python
import torch
from opal_model import OPAL

model = OPAL(textual_dim=512, visual_dim=512, structured_dim=20)
text = torch.randn(8, 512)
image = torch.randn(8, 512)
struct = torch.randn(8, 20)

outputs = model(text, image, struct)
g, Q0, Q1 = outputs["g"], outputs["Q0"], outputs["Q1"]
```

Compute the DTCO-IF loss (including EIF-variance):

```python
import torch
from opal_loss import DTCOIFLoss

T = torch.randint(0, 2, (8,)).float()
Y = torch.randn(8)

g_safe = g.view(-1).clamp(1e-6, 1 - 1e-6)
logit = torch.log(g_safe / (1 - g_safe))

loss_fn = DTCOIFLoss(alpha=1.0, lambda_eif=0.9, delta=0.15, eps_g=1e-3)
losses = loss_fn(logit, Q0, Q1, T, Y)
loss = losses["loss_total"]
```

## Project Structure

```
.
├── __init__.py          # public API exports
├── encoders.py          # ModalityEncoder, CrossModalAttention
├── opal_model.py        # OPAL model (propensity + potential outcomes)
├── opal_loss.py         # DTCO-IF / EIF-variance objective
├── trainer.py           # orthogonal dual-task training core
├── config.py            # default hyperparameters
└── requirements.txt     # minimal dependencies
```


