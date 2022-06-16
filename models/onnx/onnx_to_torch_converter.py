import torch
import onnx
from onnx2torch import convert
import argparse

parser = argparse.ArgumentParser(description='ONNX to PyTorch Converter')
parser.add_argument('-i', '--input', type=str, help='ONNX file', required=True)
parser.add_argument('-o', '--output', type=str,
          help='Output/ PyTorch file', required=True)

args = parser.parse_args()

onnx_model_path = args.input
pytorch_model_path = args.output

# torch_model_1 = convert(onnx_model_path)
# onnx_model = onnx.load(onnx_model_path)

onnx_model = onnx.load(onnx_model_path)
assert isinstance(onnx_model, onnx.onnx_ml_pb2.ModelProto)

torch_model_2 = convert(onnx_model)

torch.save(torch_model_2, pytorch_model_path)