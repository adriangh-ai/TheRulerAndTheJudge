from __future__ import annotations

import pandas as pd
import hashlib
import unicodedata

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional

# Optional but recommended for SummEval
try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


# ---------------------------------------------------------------------
# Common representation
# ---------------------------------------------------------------------
@dataclass
class Judgment:
    dataset: str          # "SummEval"
    item_id: str          # unique within dataset (e.g. question+model+claim)
    system_id: str        # model/system name ("gpt-3.5-turbo", "flan-t5-xxl", "gpt4", ...)
    dimension: str        # "correctness", "faithfulness", "coherence", ...
    rater_id: str         # anonymized annotator ID or synthetic ("expert_0", ...)
    score: Any            # raw label / Likert / int / float; you can map to logits later

    # optional useful fields (kept as columns in the final DF)
    context: Optional[str] = None
    reference: Optional[str] = None
    answer: Optional[str] = None
    claim: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def judgments_to_dataframe(judgments: Iterable[Judgment]) -> pd.DataFrame:
    rows = []
    for j in judgments:
        d = asdict(j)
        extra = d.pop("extra", None) or {}
        d.update(extra)
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2) SummEval, using HuggingFace "KnutJaegersberg/summeval_pairs"
#
#    HF schema for each row (from dataset card):
#      - id: article/summary id
#      - text: system summary
#      - model_id: system name
#      - references: list[str] (reference summaries)
#      - expert_annotations: list[ {coherence, consistency, fluency, relevance} ]
#      - turker_annotations: list[ {coherence, consistency, fluency, relevance} ]
#      - decoded: source article
#
#    This gives per-rater Likert scores for the four classic dimensions.
# ---------------------------------------------------------------------

SUMMEVAL_DIMS = ["coherence", "consistency", "fluency", "relevance"]


def load_summeval_hf_pairs() -> pd.DataFrame:
    """
    Load SummEval via HuggingFace dataset KnutJaegersberg/summeval_pairs
    and flatten expert + turker annotations.
    """
    if load_dataset is None:
        raise ImportError(
            "datasets is not installed. `pip install datasets` to use this loader."
        )

    ds = load_dataset("KnutJaegersberg/summeval_pairs", split="train")

    judgments: List[Judgment] = []

    for row_idx, row in enumerate(ds):
        item_id = row["id"]
        system_id = row.get("model_id", "unknown")
        sys_summary = row.get("text", "")
        article = row.get("decoded", "")
        references = row.get("references", [])

        for source, ann_list in [
            ("expert", row.get("expert_annotations") or []),
            #("turker", row.get("turker_annotations") or []),
        ]:
            for rater_idx, ann in enumerate(ann_list):
                for dim in SUMMEVAL_DIMS:
                    if dim not in ann:
                        continue
                    score = ann[dim]

                    j = Judgment(
                        dataset="SummEval",
                        item_id=f"summeval_{item_id}",
                        system_id=system_id,
                        dimension=dim,
                        rater_id=f"{source}_{rater_idx}",
                        score=score,
                        context=article,
                        reference=" ||| ".join(references) if references else None,
                        answer=sys_summary,
                        claim=None,
                    )
                    judgments.append(j)

    return judgments_to_dataframe(judgments)


# ---------------------------------------------------------------------
# Convenience: load everything and concat
# ---------------------------------------------------------------------
def load_all_human_judgments(

    use_summeval_hf: bool = True,
) -> pd.DataFrame:
    """
    Load:
      - InstructQA human_eval_annotations
      - ExpertQA r2_compiled_anon.jsonl
      - SummEval via HF (or skip if use_summeval_hf=False)
    """
    dfs = []

    if use_summeval_hf:
        print("Loading SummEval from HuggingFace (KnutJaegersberg/summeval_pairs)...")
        df_sum = load_summeval_hf_pairs()
        dfs.append(df_sum)

    df_all = pd.concat(dfs, ignore_index=True)
    return df_all


# ----------------------------------------------------------------------
# Loader for Hanna Benchmark ASG
# ----------------------------------------------------------------------
def load_hanna() -> pd.DataFrame:
    def stable_text_id(text: str, *, digest_size: int = 16) -> str:
        normalized = unicodedata.normalize("NFKC", str(text)).strip()
        return hashlib.blake2b(normalized.encode("utf-8"), digest_size=digest_size).hexdigest()

    url = "https://raw.githubusercontent.com/dig-team/hanna-benchmark-asg/main/hanna_stories_annotations.csv"
    df = pd.read_csv(url)
    id_vars = ['Story ID', 'Prompt', 'Story', 'Worker ID', 'Model']
    formatted_df = df.melt(
        id_vars=id_vars, 
        value_vars=['Relevance', 'Coherence', 'Empathy', 'Surprise', 'Engagement', 'Complexity'],
        var_name='dimension', 
        value_name='score'
    )

    formatted_df.rename(
        columns={'uid': 'item_id', 'Worker ID': 'rater_id', 'Model': 'system_id'},
        inplace=True
    )

    formatted_df['item_id'] = formatted_df['Prompt'].apply(stable_text_id)
    formatted_df['dataset'] = 'hanna'
    return formatted_df


def load_hitz_basse(split: str = "es") -> pd.DataFrame:
    df = load_dataset("HiTZ/BASSE", split)["test"].to_pandas()
    id_vars = ['idx', 'url', 'document', 'summary', 'model', 'prompt']
    df = df.rename(columns={'5W1H': '5w1h'})
    formatted_df = df.melt(
        id_vars=id_vars, 
        value_vars=['relevance', 'fluency', 'consistency', 'coherence', '5w1h'],
        var_name='dimension', 
        value_name='score'
    )

    formatted_df["prompt"] = formatted_df["prompt"].fillna("subhead")
    formatted_df["model_prompt"] = (
        formatted_df["model"].astype(str).str.strip()
        + "_"
        + formatted_df["prompt"].astype(str).str.strip()
    )

    formatted_df["annotator_id"] = formatted_df["score"].apply(
        lambda vals: [f"human_{i+1}" for i in range(len(vals))]
    )

    formatted_df = formatted_df.explode(["score", "annotator_id"], ignore_index=True)
    formatted_df["score"] = formatted_df["score"].astype(int)

    formatted_df.rename(
        columns={'url': 'item_id', 'annotator_id': 'rater_id', 'model_prompt': 'system_id'},
        inplace=True
    )

    return formatted_df