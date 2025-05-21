from m4.models.custom_modules import DecoupledEmbedding, DecoupledLinear
from m4.models.idefics.configuration_idefics import IdeficsConfig
from m4.models.idefics.modeling_idefics import IdeficsForCausalLM
from m4.models.vgpt2.configuration_vgpt2 import VGPT2Config
from m4.models.vgpt2.modeling_vgpt2 import VGPT2LMHeadModel
from m4.models.vllama3.configuration_vllama3 import VLlama3Config
from m4.models.vllama3.modeling_vllama3 import VLlama3ForCausalLM
from m4.models.vmistral.configuration_vmistral import VMistralConfig
from m4.models.vmistral.modeling_vmistral import VMistralForCausalLM


_SUPPORTED_MODELS = {
    "vgpt2": VGPT2Config,
    # "vllama": IdeficsConfig,
    "idefics": IdeficsConfig,
    "vmistral": VMistralConfig,
    "vllama3": VLlama3Config,
}

model_type_to_modeling_class = {
    "vgpt2": VGPT2LMHeadModel,
    # "vllama": IdeficsForCausalLM,
    "idefics": IdeficsForCausalLM,
    "vmistral": VMistralForCausalLM,
    "vllama3": VLlama3ForCausalLM,
}
