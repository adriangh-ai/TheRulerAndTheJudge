import argparse
import os
import re

import pandas as pd
from typing import List, Dict
from tqdm import tqdm
from openai import OpenAI

from ..prompt_templates import SummevallTemplate, HitzBasseTemplate, HannaTemplate
from ..data_preprocess import load_all_human_judgments, load_hanna, load_hitz_basse

from concurrent.futures import ThreadPoolExecutor


class APIJudge:
    def __init__(self, model_id: str, base_url: str = None, api_key: str = None):
        self.client = OpenAI(
            base_url=base_url or "https://openrouter.ai/api/v1",
            api_key=api_key# if api_key is None else api_key,
        )
        self.model_id = model_id

    def generate_and_judge(self, messages: List[Dict], max_tokens=10000, temperature=0.0):
        """
        Generates response and extracts score + uncertainty metrics 
        in a SINGLE pass using logprobs=True.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature, # Deterministic for scoring 
                extra_body={
                    "provider": {
                        "ignore": ["alibaba", "nextbit/int4"]
                    },
                }
            )
            
            choice = response.choices[0]
            full_text = choice.message.content
            # 1. Regex to find the Score value in the text
            # Matches: "score": 4  or "score":4
            pattern = re.compile(r'\b([1-5])\b')
            matches = list(pattern.finditer(full_text))
            if not matches:
                print("❌ Score pattern not found in generated text.")
                return None, full_text
            
            target_match = matches[0]
            score_val = int(target_match.group(1))
            
            return score_val, full_text

        except Exception as e:
            print(f"Error during generation: {e}")
            return None, str(e)


def run_api_judge_pipeline(
    df: pd.DataFrame, 
    model_id: str,
    template_cls, 
    api_key: str = None,
    variance = False,
):
    judge = APIJudge(model_id=model_id, api_key=api_key)
    template = template_cls()
    
    results = []

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        question = row['item']
        response = row['response']
        
        for dimension in template.DIMS:
            # Format Prompt
            messages = template.format(dimension=dimension, question=question, response=response)
           
            prompt_str = f"{messages[-1]['content']}" 
            
            # 2. Generate and Score
            score, raw_text = judge.generate_and_judge(messages)
            if score is None:
                continue
            
            if variance:
                score_list = [judge.generate_and_judge(messages, temperature=1.0)[0] for _ in range(10)]

            results.append({
                "item_id": row.get('item_id'),
                "system_id": row.get('system_id'),
                "dimension": dimension,
                "score": score,
                "model_output": raw_text,
                "is_llm": True,
                "rater_id": model_id,
                "question": question,
                "response": response,
                "score_list": score_list if variance else [],
                "prompt": prompt_str,
            })
            
    return pd.DataFrame(results)


def main():
    argparser = argparse.ArgumentParser(description="Run API-based LLM Judge on benchmarks.")
    argparser.add_argument(
        "--benchmark", 
        type=str,
         choices=["summeval", "hitz_basse", "hitz_basse_vas", "hanna"], 
        required=True, 
        help="Which benchmark to run on."
    )
    argparser.add_argument(
        "--judge_folder", 
        type=str, 
        default="llm_judge_outputs", 
        help="Folder to save judge outputs."
    )

    args = argparser.parse_args()
    if args.benchmark == "summeval":
        df = load_all_human_judgments(
        use_summeval_hf=True,
        )
        df.rename(columns={"answer": "item", "context": "response"}, inplace=True)
        template = SummevallTemplate
    elif args.benchmark == "hitz_basse":
        df = load_hitz_basse()
        df.rename(columns={"document": "item", "summary": "response"}, inplace=True)
        template = HitzBasseTemplate
    elif args.benchmark == "hanna":
        df = load_hanna()
        df.rename(columns={"Prompt": "item", "Story": "response"}, inplace=True)
        template = HannaTemplate
    elif args.benchmark == "hitz_basse_vas":
        df = load_hitz_basse(split="eu")
        df.rename(columns={"document": "item", "summary": "response"}, inplace=True)
        template = HitzBasseTemplate
    else:
        raise ValueError(f"Unsupported benchmark: {args.benchmark}")

    max_workers = 1

    items = df[["item_id", "system_id", "item", "response"]].drop_duplicates()
    partial_items = items.copy()
    partitions = len(items) // max_workers + 1
    partial_items = [items.iloc[i*partitions:(i+1)*partitions] for i in range(max_workers)]

    key = os.environ.get("OPEN_ROUTER")
    models = [
        "google/gemma-3-27b-it",
        "deepseek/deepseek-v3.2",
        "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        "meta-llama/llama-3.2-3b-instruct", 
        "meta-llama/llama-3.3-70b-instruct", 
        "qwen/qwen3-next-80b-a3b-instruct",
        "google/gemma-3-4b-it",
        "qwen/qwen3-14b"
        ]
    def run_partition(partial_i):
        return run_api_judge_pipeline(
            partial_i,
            model, 
            template, 
            api_key=key,
            variance=False,
        )
    
    for model in models:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                run_partition, 
                partial_items)
            output = pd.concat(results).reset_index(drop=True)
        
        folder = os.path.join(args.judge_folder, args.benchmark)
        os.makedirs(folder, exist_ok=True)
        output.to_parquet(
            os.path.join(folder, f"{args.benchmark}_{model.replace('/', '_')}.parquet"), 
            index=False
        )


if __name__ == "__main__":
    main()