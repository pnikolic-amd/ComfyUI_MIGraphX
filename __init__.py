from .migraphx_compile_sd3 import CompileSD3MIGraphX
from .migraphx_utils import MgxTransformer

NODE_CLASS_MAPPINGS = {
    "CompileSD3MIGraphX": CompileSD3MIGraphX
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CompileSD3MIGraphX": "Compile SD3 model on migraphx"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']