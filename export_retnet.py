import torch
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

config = RetNetConfig(vocab_size=64000)
retnet = RetNetDecoder(config)

prev_output_tokens = torch.ones(2, 10)
token_embeddings = torch.rand(2, 10, config.decoder_embed_dim)

print(retnet)

class M(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, prev_output_tokens, token_embeddings):
        o1, o2 = self.m(
            prev_output_tokens=prev_output_tokens,
            token_embeddings=token_embeddings,
            features_only=True,
        )
        return o1
        # return o1, o2["inner_states"], o2["l_aux"], o2["attn"]

M(retnet)(prev_output_tokens, token_embeddings)
torch.onnx.export(
    M(retnet),
    (prev_output_tokens, token_embeddings),
    "retnet.onnx",
    opset_version=14,
)
