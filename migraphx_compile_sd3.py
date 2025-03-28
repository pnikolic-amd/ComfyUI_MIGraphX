import comfy.model_patcher
import comfy.model_management
import comfy.supported_models
from .migraphx_utils import load_MGX_transformer_model


class CompileSD3MIGraphX:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL", ),
                "force_compile": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "When set on false, it will try to load model from mxr file, if file exists.",},),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "height": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 8,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "width": ("INT", {
                    "default": 512, "min": 256, "max": 2048, "step": 8,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "context_len": ([77, 154], {
                    "tooltip": "Depends on text encoders that are used. If t5 enabled set to 154, if not set to 77.",
                },),
                "data_type": (["fp16", "int8", "fp32"], {
                    "tooltip": "Depends on SD3 model that is loaded.",
                },),
            }
        }
        return inputs
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "compile_on_MIGraphX"
    CATEGORY = "advanced/migraphx"

    def compile_on_MIGraphX(self, model, force_compile, batch_size, height, width, context_len, data_type):
        input_shapes = {
                "x": [2*batch_size, 16, height // 8, width // 8],
                "timesteps": [2*batch_size],
                "context": [2*batch_size, context_len, 4096],
                "y": [2*batch_size, 2048],
            }
        mxr_file_name = f"model_{batch_size}_{height}_{width}_{context_len}_{data_type}.mxr"
    
        conf = comfy.supported_models.SD3({})
        conf.unet_config["disable_unet_model_creation"] = True
        comfy_model = conf.get_model({})        

        comfy_model.diffusion_model = load_MGX_transformer_model(model, force_compile, mxr_file_name, 
                                                                 input_shapes, data_type)
        comfy_model.memory_required = lambda *args, **kwargs: 0

        return (comfy.model_patcher.ModelPatcher(comfy_model,
                                                load_device=comfy.model_management.get_torch_device(),
                                                offload_device=comfy.model_management.unet_offload_device()),)



NODE_CLASS_MAPPINGS = {
    "CompileSD3MIGraphX": CompileSD3MIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileSD3MIGraphX": "Compile SD3 model on migraphx"
}