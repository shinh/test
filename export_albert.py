import torch
from transformers import AlbertConfig, AlbertModel

# Initializing an ALBERT-xxlarge style configuration
#albert_xxlarge_configuration = AlbertConfig()

# Initializing an ALBERT-base style configuration
albert_base_configuration = AlbertConfig(
    hidden_size=768,
    num_attention_heads=12,
    intermediate_size=3072,
)

# Initializing a model (with random weights) from the ALBERT-base style configuration
model = AlbertModel(albert_base_configuration)

# Accessing the model configuration
configuration = model.config

print(model)

# torch.save(model, "albert.pt")

x = torch.zeros(1, 3, dtype=torch.int32)
torch.onnx.export(model, x, "albert.onnx")
