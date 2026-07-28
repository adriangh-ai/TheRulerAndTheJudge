import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


def load_model(
        model_id:str, 
        device:str="auto", 
        attn_implementation='eager', 
        multimodal=False
    ) -> AutoModelForCausalLM:
    """
    Load a model.

    Args:
        model_id (str): Model identifier.

    Returns:
        AutoModelForCausalLM: Model.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype='auto',
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )

    # Compatibility shim for multimodal configurations exposing text_config.
    if multimodal:
        cfg = model.config
        tc = getattr(cfg, "text_config", None)

        if tc is not None:
            for a in (
                "num_attention_heads", 
                "num_hidden_layers", 
                "num_key_value_heads",
                "head_dim", 
                "hidden_size"
            ):
                if not hasattr(cfg, a) and hasattr(tc, a):
                    setattr(cfg, a, getattr(tc, a))

    return model


def load_quantized_model(model_id:str, device:str='auto') -> AutoModelForCausalLM:
    """
    Load a quantized model using BitsAndBytes.

    Args:
        model_id (str): Model identifier.

    Returns:
        AutoModelForCausalLM: Quantized model.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',  # Normalized float 4 
        bnb_4bit_use_double_quant=True,  # Double quantization for better efficincy
        bnb_4bit_compute_dtype=torch.float16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        dtype='auto',
        device_map=device,
        quantization_config=bnb_config,
    )
    return model


def load_model_tokenizer(
        model_id:str, 
        quant=False,
        device='auto',
        **kwargs
    ) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load a model and tokenizer.

    Args:
        model_id (str): Model identifier.

    Returns:
        tuple[AutoModelForCausalLM, AutoTokenizer]: Model and tokenizer.
    """
    attn_implementation = kwargs.get('attn_implementation', 'eager')

    if quant:
        model = load_quantized_model(model_id, device=device)
    else:
        model = load_model(model_id, device=device, attn_implementation=attn_implementation)
           
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # In case pad_token is not set
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer