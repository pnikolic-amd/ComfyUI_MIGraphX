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
                "model_type": (["sd1.x", "sd2.x-768v", "sd3.x", "sdxl_base", "sdxl_refiner", "flux_dev", "flux_schnell"], ),
                "batch_size": ("INT", {
                    "default": 1, "min": 1, "max": 32, "step": 1,
                    "tooltip": "Based on value for latent.",
                },),
                "width": ("INT", {
                    "default": 1024, "min": 128, "max": 4096, "step": 128,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "height": ("INT", {
                    "default": 1024, "min": 128, "max": 4096, "step": 128,
                    "tooltip": "Must be same value as value for latent.",
                },),
                "data_type": (["fp16", "int8", "fp32", "bfp16"], {
                    "tooltip": "Depends on diffusion model that is loaded.",
                },),
            },
            "optional": {
                "context": ("BOOLEAN", {
                    "default": False, 
                    "tooltip": "True if T5 encoder is used for SD3.x models",
                },),
            }
        }
        return inputs
    
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_NODE = True
    FUNCTION = "compile_on_MIGraphX"
    CATEGORY = "advanced/migraphx"

    def compile_on_MIGraphX(self, model, force_compile, model_type, batch_size, height, width, data_type, context):    
        if model_type == "sd1.x":
            conf = comfy.supported_models.SD15({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = comfy.model_base.BaseModel(conf)
        elif model_type == "sd2.x-768v":
            conf = comfy.supported_models.SD20({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = comfy.model_base.BaseModel(conf, model_type=comfy.model_base.ModelType.V_PREDICTION)
        elif model_type == "sd3.x":
            conf = comfy.supported_models.SD3({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = conf.get_model({}) 
        elif model_type == "sdxl_base":
            conf = comfy.supported_models.SDXL({"adm_in_channels": 2816})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = comfy.model_base.SDXL(conf)
        elif model_type == "sdxl_refiner":
            conf = comfy.supported_models.SDXLRefiner(
                {"adm_in_channels": 2560})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = comfy.model_base.SDXLRefiner(conf) 
        elif model_type == "flux_dev":
            conf = comfy.supported_models.Flux({})
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = conf.get_model({})
        elif model_type == "flux_schnell":
            conf = comfy.supported_models.FluxSchnell({}) 
            conf.unet_config["disable_unet_model_creation"] = True
            comfy_model = conf.get_model({})
        else:
            print("ERROR: model not supported.")
            return ()  

        mxr_file_name = f"{model_type}_{batch_size}_{width}_{height}_{data_type}.mxr"
        
        if model_type in ["sd1.x", "sd2.x-768v", "sd3.x", "sdxl_base", "sdxl_refiner"]:
            batch_size = batch_size * 2

        comfy_model.diffusion_model = load_MGX_transformer_model(model, force_compile, mxr_file_name, 
                                                                batch_size, height, width, context,
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