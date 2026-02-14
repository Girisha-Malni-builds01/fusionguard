import torch

def dynamic_int8(model):
    if torch.backends.quantized.engine == "none":
        if torch.backends.quantized.supported_engines:
            torch.backends.quantized.engine = torch.backends.quantized.supported_engines[0]

    return torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
