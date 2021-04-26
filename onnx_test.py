import onnx
import tvm
from tvm import relay


model = onnx.load("mb1-ssd.onnx")

ishape = (1, 3, 300, 300)
shape_dict = {"input.1": ishape}

mod, params = relay.frontend.from_onnx(model, shape_dict, freeze_params=True)
mod = relay.transform.DynamicToStatic()(mod)

print(mod)
