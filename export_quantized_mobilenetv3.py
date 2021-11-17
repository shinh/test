import torch
import torchvision

from torch.hub import load_state_dict_from_url


qm = torchvision.models.quantization.mobilenet_v3_large(pretrained=True)

x = torch.zeros(1,3,224,224)

quant_url = "https://download.pytorch.org/models/quantized/mobilenet_v3_large_qnnpack-5bcacf28.pth"
state_dict = load_state_dict_from_url(quant_url, progress=True)

qm.fuse_model()
qm.qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
torch.quantization.prepare_qat(qm, inplace=True)

qm.load_state_dict(state_dict)


OPSET_VERSION = 13

def register_quant(opset):
    @torch.onnx.symbolic_helper.parse_args("v")
    def symbolic_fake_quantize_per_tensor_affine(g, x):
        return g.op("QuantizeLinear", x)

    torch.onnx.symbolic_registry.register_op(
        "fake_quantize_per_channel_affine",
        symbolic_fake_quantize_per_tensor_affine,
        "",
        opset
    )


def register_hswish(opset):
    @torch.onnx.symbolic_helper.parse_args("v")
    def symbolic_hardswish(g, input):
        return g.op("HardSwish", input)

    torch.onnx.symbolic_registry.register_op(
        "hardswish", symbolic_hardswish, "", opset
    )


register_quant(OPSET_VERSION)
register_hswish(OPSET_VERSION)


torch.onnx.export(qm, x, "quantized_mobilenetv3_large.onnx",
                  opset_version=OPSET_VERSION,
                  input_names=["input"],
                  output_names=["output"],
                  enable_onnx_checker=False
                  )
