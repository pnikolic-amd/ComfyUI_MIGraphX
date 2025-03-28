import os
import sys
import torch
from typing import Union, Dict, Type
import comfy.model_patcher
import comfy.model_management
import comfy.model_base
import migraphx as mgx


DATA_TYPES = {"fp16": torch.float16, "int8": torch.int8, "float32": torch.float32}

_MGX_TO_TORCH_DTYPE_DICT = {
    "bool_type": torch.bool,
    "uint8_type": torch.uint8,
    "int8_type": torch.int8,
    "int16_type": torch.int16,
    "int32_type": torch.int32,
    "int64_type": torch.int64,
    "float_type": torch.float32,
    "double_type": torch.float64,
    "half_type": torch.float16,
}

_TORCH_TO_MGX_DTYPE_DICT = {
    value: key
    for (key, value) in _MGX_TO_TORCH_DTYPE_DICT.items()
}


class MgxTransformer:
    def __init__(self, model, dtype):
        self.model = model
        self.dtype = dtype
        self.model_data = torch.zeros([1])
        self._allocate_torch_tensors()
    
    def __call__(self, x, timesteps, context, y=None, control=None, transformer_options=None, **kwargs):
        self._copy_tensor_sync(self.model_data["x"], x)
        self._copy_tensor_sync(self.model_data["timesteps"], timesteps)
        self._copy_tensor_sync(self.model_data["context"], context)
        if y is not None:
            self._copy_tensor_sync(self.model_data["y"], y)

        mgx_data = self._tensors_to_args()
        self.model.run(mgx_data)
        mgx.gpu_sync()

        return self.model_data[self._get_output_name(0)]
    
    def load_state_dict(self, sd, strict=False):
        pass

    def state_dict(self):
        return {}
    
    @staticmethod
    def _tensor_to_arg(tensor):
        return mgx.argument_from_pointer(
            mgx.shape(
                **{
                    "type": _TORCH_TO_MGX_DTYPE_DICT[tensor.dtype],
                    "lens": list(tensor.size()),
                    "strides": list(tensor.stride())
                }), tensor.data_ptr())

    def _tensors_to_args(self):
        return {name: self._tensor_to_arg(tensor) for name, tensor in self.model_data.items()}

    @staticmethod
    def _get_output_name(idx):
        return f"main:#output_{idx}"

    def _allocate_torch_tensors(self):
        input_shapes = self.model.get_parameter_shapes()
        self.model_data = {
            name: torch.zeros(shape.lens()).to(
                _MGX_TO_TORCH_DTYPE_DICT[shape.type_string()]).to(device="cuda")
            for name, shape in input_shapes.items()
        }

    @staticmethod
    def _copy_tensor_sync(tensor, data):
        tensor.copy_(data)
        torch.cuda.synchronize()

def create_path(path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    root_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.join(root_path, path)        

def create_dir(save_dir_path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path, exist_ok=True)  
        print(f"Directory '{save_dir_path}' created successfully.")  
    return save_dir_path

def convert_model_to_ONNX(onnx_tmp_dir: Union[str, os.PathLike], 
                           model: comfy.model_patcher.ModelPatcher,
                           input_shapes: Dict,
                           dtype: Type):
    onnx_file_path = os.path.join(onnx_tmp_dir, "model.onnx")

    comfy.model_management.unload_all_models()
    comfy.model_management.load_models_gpu([model], force_patch_weights=True, force_full_load=True)
    transformer = model.model.diffusion_model

    inputs = {}
    input_names = []
    output_names = ["h"]
    for name in input_shapes:
        input_names.append(name)
        inputs[name] = torch.zeros(
                input_shapes[name],
                device=comfy.model_management.get_torch_device(),
                dtype=dtype)
        
    dynamic_axes = {
                "x": {0: "batch", 1: "num_channels", 2: "height", 3: "width"},
                "timesteps": {0: "batch"},
                "context": {0: "batch", 1: "num_embeds"},
                "y": {0: "batch"},
                "h": {0: "batch", 1: "num_channels", 2: "height", 3: "width"},
            }

    print("Export model to ONNX.")
    torch.onnx.export(
        transformer,
        (inputs,),
        onnx_file_path,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()


def convert_model_with_mgx(onnx_tmp_dir: Union[str, os.PathLike], 
                            mxr_file_path: Union[str, os.PathLike], 
                            input_shapes: Dict):
    onnx_file = os.path.join(onnx_tmp_dir, "model.onnx")
    
    print("Load model from ONNX.")
    model = mgx.parse_onnx(onnx_file, map_input_dims=input_shapes) 

    print("Compile model with mgx.")
    model.compile(mgx.get_target("gpu"),
                          exhaustive_tune=False,
                          offload_copy=False)
        
    print("Save compiled model.")
    mgx.save(model, mxr_file_path, format="msgpack")

    os.system(f"rm -rf {onnx_tmp_dir}")

    return model

def load_from_mxr(mxr_file_path: Union[str, os.PathLike]):
    print("Loading transformer model...")
    model = mgx.load(mxr_file_path, format="msgpack")
    print("Model loaded successfully.")
    return model

def load_MGX_transformer_model(model: comfy.model_base.BaseModel, force_compile: bool, mxr_file_name: str,
                               input_shapes: Dict, data_type: str) -> MgxTransformer:
    transformer_mgx = None
    mxr_file_path = create_path(f"mgx_files/{mxr_file_name}")
    if force_compile or not os.path.isfile(mxr_file_path):
        create_dir(create_path("mgx_files"))
        onnx_tmp_dir = create_dir(create_path("mgx_files/tmp"))

        convert_model_to_ONNX(onnx_tmp_dir, model, input_shapes, DATA_TYPES[data_type])
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        transformer_mgx = convert_model_with_mgx(onnx_tmp_dir, mxr_file_path, input_shapes)
    else:
        transformer_mgx = load_from_mxr(mxr_file_path)
        
    return MgxTransformer(transformer_mgx, DATA_TYPES[data_type])