# Estimating Causal Effects with Orthogonal Prediction-Aware Learning


## Requirements

```
pip install -r requirements.txt
```

## Datasets

We use semi-synthetic multimodal datasets constructed from Weibo:

- LFC (Fallen City)
- NZ (Nezha)
- TGD2 (Special Forces 2)

Dataset:https://pan.quark.cn/s/8627c70ae337?pwd=84ck
## Project Structure

```
├── encoders.py          # ModalityEncoder, CrossModalAttention
├── opal_model.py        # OPAL model
├── opal_loss.py         # DTCO-IF loss
├── trainer.py           # Orthogonal training
├── config.py            # Hyperparameters
└── requirements.txt
```

