import argparse
import torch
import pandas as pd

from typing import Union, Literal

from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.notebook import tqdm

from ..utils import extract_generated_score_info
from ..prompt_templates import PromptTemplate
from ..data_preprocess import load_all_human_judgments
from .models import load_model_tokenizer


UncertaintyMethods = Union[Literal["semantic_entropy"], Literal["sar"], Literal["mahalanobis"]]

def generate_judge_responses(
        model_id:str,
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        template_cls: PromptTemplate,
        examinee: list[str],
        item_id: list[str],
        item_question: list[str], 
        examinee_response: list[str], 
        max_new_tokens:int=5000,
    ) -> pd.DataFrame:
    """
    Given the full generated IDs from the model (including prompt), identify the score value
    and its token index based on the anchor string.

    Args:
        model: The language model used for generation.
        tokenizer: The tokenizer associated with the model.
        full_ids (torch.Tensor): Tensor of shape (1, seq_len) containing token IDs of the full sequence.
        prompt_len (int): Length of the prompt in tokens.
    """
    if len(item_id) != len(item_question) or len(item_id) != len(examinee_response):
        raise ValueError("Input lists must have the same length.")
    
    template = template_cls()

    output_list = []
    for id, ex_id, question, response in tqdm(
        zip(item_id, examinee, item_question, examinee_response), total=len(item_question)
    ):
        for dimension in template.DIMS:
            messages = template.format(dimension=dimension, question=question, response=response)
            tokenized_input = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
                return_dict=True,
            )

            # Generate model outputs
            with torch.no_grad():
                outputs = model.generate(
                    **tokenized_input.to(model.device),
                    max_new_tokens=max_new_tokens,  # Allow some generation length
                    do_sample=False,
                    return_dict_in_generate=True,
                )

            # Extract score and index
            prompt_len = tokenized_input['input_ids'].shape[1]
            full_ids = outputs["sequences"] # shape [1, N]

            # 2. VALIDATION STEP
            # We only pass prompt_len so it knows where to START searching
            generated_score, score_index = extract_generated_score_info(
                tokenizer, 
                full_ids, 
                prompt_len
            )

            if generated_score is None:
                print("❌ Model failed to generate a valid JSON score sequence.")
                continue
                # Handle fallback (e.g., retry, treat as failure, etc.)

                
            output_list.append({
                "item_id": id,
                "system_id": ex_id,
                "question": question,
                "response": response,
                "dimension": dimension,
                "score": generated_score,
                "score_index": score_index,
                "is_llm": True,
                "rater_id": model_id,
                "model_output": tokenizer.decode(full_ids[0]),
            })
    return pd.DataFrame(output_list)


def main(
        model_id:str, 
        template_class: PromptTemplate, 
        multimodal:bool=False
    ):
    # Load model and tokenizer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model, tokenizer = load_model_tokenizer(
        model_id=model_id, 
        device=device, 
        attn_implementation='eager',
        multimodal=multimodal
    )
    
    # Load human judgments data
    data_df = load_all_human_judgments()
    
    # Generate judge responses
    results_df = generate_judge_responses(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        template_cls=template_class,
        examinee=data_df['system_id'].tolist(),
        item_id=data_df['item_id'].tolist(),
        item_question=data_df['question'].tolist(),
        examinee_response=data_df['response'].tolist(),
    )
    
    # Save or process results as needed
    results_df.to_parquet(f"{model_id.replace('/', '_')}_judge_responses.parquet", index=False)
    print("Results saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate judge responses using a language model.")
    parser.add_argument("--model_id", type=str, required=True, help="Model identifier.")
    parser.add_argument("--template", type=str, required=True, help="Prompt template class name.")
    parser.add_argument("--multimodal", action='store_true', help="Whether to load a multimodal model.")
    args = parser.parse_args()

    # Convert template class name string to actual class
    template_class = globals()[args.template]

    main(
        model_id=args.model_id,
        template_class=template_class,
        multimodal=args.multimodal
    )