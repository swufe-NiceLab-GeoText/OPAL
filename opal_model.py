from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoders import CrossModalAttention, ModalityEncoder


class CausalHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 100, output_type: str = "propensity"):
        super().__init__()
        self.output_type = output_type
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

        if output_type == "propensity":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.head(x))


class OPAL(nn.Module):
    def __init__(
        self,
        textual_dim: int = 512,
        visual_dim: int = 512,
        structured_dim: int = 0,
        hidden_dim: int = 256,
        representation_dim: int = 200,
        num_attention_heads: int = 4,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.textual_dim = textual_dim
        self.visual_dim = visual_dim
        self.structured_dim = structured_dim

        self.text_encoder = ModalityEncoder(textual_dim, hidden_dim, hidden_dim, dropout_rate)
        self.image_encoder = ModalityEncoder(visual_dim, hidden_dim, hidden_dim, dropout_rate)

        if structured_dim > 0:
            self.struct_encoder = ModalityEncoder(structured_dim, hidden_dim // 2, hidden_dim // 2, dropout_rate)
        else:
            self.struct_encoder = None

        self.text_to_image_attn = CrossModalAttention(hidden_dim, num_attention_heads, dropout_rate)
        self.image_to_text_attn = CrossModalAttention(hidden_dim, num_attention_heads, dropout_rate)

        fusion_input_dim = hidden_dim * 2 + (hidden_dim // 2 if structured_dim > 0 else 0)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.ELU(),
            nn.Dropout(dropout_rate),
        )

        self.g_head = CausalHead(representation_dim, 100, "propensity")
        self.Q0_head = CausalHead(representation_dim, 100, "outcome")
        self.Q1_head = CausalHead(representation_dim, 100, "outcome")

        self.modality_contribution = nn.Sequential(
            nn.Linear(representation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        textual_features: torch.Tensor,
        visual_features: torch.Tensor,
        structured_features: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        return_modality_contrib: bool = False,
    ) -> Dict[str, torch.Tensor]:
        text_encoded = self.text_encoder(textual_features)
        image_encoded = self.image_encoder(visual_features)

        text_attended, text_attn = self.text_to_image_attn(
            text_encoded, image_encoded, image_encoded, return_attention
        )
        image_attended, image_attn = self.image_to_text_attn(
            image_encoded, text_encoded, text_encoded, return_attention
        )

        if self.struct_encoder is not None and structured_features is not None:
            struct_encoded = self.struct_encoder(structured_features)
            fused = torch.cat([text_attended, image_attended, struct_encoded], dim=-1)
        else:
            fused = torch.cat([text_attended, image_attended], dim=-1)

        representation = self.fusion(fused)

        g = self.g_head(representation)
        Q0 = self.Q0_head(representation)
        Q1 = self.Q1_head(representation)

        outputs: Dict[str, torch.Tensor] = {
            "g": g,
            "Q0": Q0,
            "Q1": Q1,
            "representation": representation,
        }

        if return_attention:
            outputs["text_attn"] = text_attn
            outputs["image_attn"] = image_attn

        if return_modality_contrib:
            outputs["modality_contrib"] = self.modality_contribution(representation)

        return outputs

    @staticmethod
    def estimate_ate(Q0: torch.Tensor, Q1: torch.Tensor) -> torch.Tensor:
        return (Q1 - Q0).mean()

    @staticmethod
    def estimate_individual_te(Q0: torch.Tensor, Q1: torch.Tensor) -> torch.Tensor:
        return Q1 - Q0


