from .migraphx_compile_sd3 import CompileDiffusersMIGraphX
from .migraphx_utils import MgxTransformer

NODE_CLASS_MAPPINGS = {
    "CompileDiffusersMIGraphX": CompileDiffusersMIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileDiffusersMIGraphX": "Compile diffusion model on migraphx"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']