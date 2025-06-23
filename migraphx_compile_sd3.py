import comfy.model_patcher
import comfy.model_management
import comfy.supported_models
from .migraphx_utils import load_MGX_transformer_model


class CompileDiffusersMIGraphX:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "model": ("MODEL", ),
                "force_compile": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "When set on false, it will try to load model from mxr file, if file exists.",},),
                #"model_type": (["sd3", "sd3.5", "flux_dev", "flux_schnell"], ),
                "model_type": (["sd1.5", "sd3", "sd3.5"], ),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Based on value for latent.",
                },),
                "width": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 8,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "height": ("INT", {
                    "default": 1024, "min": 256, "max": 2048, "step": 8,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "context_len": ([77, 154, 256], {
                    "tooltip": "Depends on text encoders that are used.",
                },),
                "data_type": (["fp16", "int8", "fp32", "bfp16"], {
                    "tooltip": "Depends on diffusion model that is loaded.",
                },),
            }
        }
        return inputs
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "compile_on_MIGraphX"
    CATEGORY = "advanced/migraphx"

    def compile_on_MIGraphX(self, model, force_compile, model_type, batch_size, height, width, context_len, data_type):    
        if model_type == "sd1.5":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd3" or  model_type == "sd3.5":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = conf.get_model({}) 
        #elif model_type == "flux_dev":
        #    conf = comfy.supported_models.Flux({})
        #elif model_type == "flux_schnell":
        #    conf = comfy.supported_models.FluxSchnell({})  
        else:
            print("ERROR: model not supported.")
            return ()  

        mxr_file_name = f"{model_type}_{batch_size}_{width}_{height}_{context_len}_{data_type}.mxr"       

        comfy_model.diffusion_model = load_MGX_transformer_model(model, force_compile, mxr_file_name, 
                                                                batch_size, height, width, context_len,
                                                                data_type)
        comfy_model.memory_required = lambda *args, **kwargs: 0

        return (comfy.model_patcher.ModelPatcher(comfy_model,
                                                load_device=comfy.model_management.get_torch_device(),
                                                offload_device=comfy.model_management.unet_offload_device()),)



NODE_CLASS_MAPPINGS = {
    "CompileDiffusersMIGraphX": CompileDiffusersMIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileDiffusersMIGraphX": "Compile diffusion model on migraphx"
}