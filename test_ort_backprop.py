# Ref: https://onnxruntime.ai/docs/get-started/training-pytorch.html
# $ pip install torch-ort
# $ python -m torch_ort.configure

import numpy as np
import onnx
import onnxruntime
import torch


class Model(torch.nn.Module):
    def forward(self, a, b, c):
        return a * b + c


a = torch.tensor(2.).requires_grad_(True)
b = torch.tensor(3.).requires_grad_(True)
c = torch.tensor(5.).requires_grad_(True)


model = Model()
y = model(a, b, c)
y.backward()

# 3, 2, and 1.
print(a.grad, b.grad, c.grad)

torch.onnx.export(Model(), (a, b, c), "mul_add.onnx",
                  input_names=["a", "b", "c"])

onnx_model = onnx.load("mul_add.onnx")
onnxruntime.training.artifacts.generate_artifacts(
    onnx_model,
    requires_grad=["a", "b", "c"],
    artifact_directory="mul_add",
)

sess = onnxruntime.InferenceSession("mul_add/training_model.onnx", providers=["CPUExecutionProvider"])

ga = np.zeros(1).astype(np.float32)
gb = np.zeros(1).astype(np.float32)
gc = np.zeros(1).astype(np.float32)
outputs = sess.run(
    [
        "a_grad.accumulation.out",
        "b_grad.accumulation.out",
        "c_grad.accumulation.out",
    ],
    {
        "a": a.detach().numpy(),
        "b": b.detach().numpy(),
        "c": c.detach().numpy(),
        "a_grad.accumulation.buffer": ga,
        "b_grad.accumulation.buffer": gb,
        "c_grad.accumulation.buffer": gc,
        "lazy_reset_grad": np.array([True]),
    }
)

print(ga, gb, gc)
