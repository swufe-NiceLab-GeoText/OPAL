# Estimating Causal Effects with Orthogonal Prediction-Aware Learning

## Requirements

```
pip install -r requirements.txt
```

## Datasets

We use semi-synthetic multimodal datasets constructed from Weibo and Yelp:

- **LFC** (Fallen City) — `data/lfc.pkl`
- **NZ** (Nezha) — `data/nz.pkl`
- **TGD2** (Special Forces 2) — `data/tgd2.pkl`
- **Yelp** — `data/yelp.pkl.gz`

Each file contains 512-d CLIP text/image embeddings, structured user attributes, binary treatment, observed outcome, and ground-truth ATE. Yelp stores large feature matrices as float16 inside a gzip archive (promote to float32 after loading if needed).

## Project Structure

```
├── encoders.py          # ModalityEncoder, CrossModalAttention
├── opal_model.py        # OPAL model
├── opal_loss.py         # DTCO-IF loss
├── trainer.py           # Orthogonal training
├── config.py            # Hyperparameters
├── __init__.py
├── requirements.txt
└── data/
    ├── lfc.pkl
    ├── nz.pkl
    ├── tgd2.pkl
    └── yelp.pkl.gz
```
