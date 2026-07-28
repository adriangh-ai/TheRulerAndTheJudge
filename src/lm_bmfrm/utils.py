import re
import numpy as np
import pickle


def extract_generated_score_info(tokenizer, full_ids, prompt_len):
    """
    Scans the GENERATED portion of full_ids to find the score.
    Returns the score value and its absolute index in full_ids.
    
    Args:
        full_ids (torch.Tensor): [1, Total_Len]
        prompt_len (int): The length of the prompt (to skip prompt tokens).
        anchor_str (str): The marker to look for.
        
    Returns:
        tuple: (score_value (int), score_token_index (int)) OR (None, None) if not found.
    """
    gen_ids = full_ids[0, prompt_len:]
    
    # 1. Define strict decoding kwargs to ensure consistency between
    #    the full decode and the partial decodes.
    decode_kwargs = {
        "skip_special_tokens": True,
        "clean_up_tokenization_spaces": False # CRITICAL: Disables invisible character merges
    }
    
    # 2. Decode fully to find the character target
    gen_text = tokenizer.decode(gen_ids, **decode_kwargs)
    
    # Regex search "score": <digit>
    pattern = re.compile(r'"score"\s*:\s*(\d)')
    matches = list(pattern.finditer(gen_text))
    if not matches: return None, None
        
    target_match = matches[-1]
    score_val = int(target_match.group(1))
    target_char_start = target_match.start(1)
    
    found_gen_idx = -1
    
    # 3. Backwards search to avoid possible reasoning matches
    for i in range(len(gen_ids) - 1, -1, -1):
        prefix_ids = gen_ids[:i]
        
        # We decode the prefix using the SAME kwargs. 
        # This guarantees the length metric is consistent.
        prefix_text = tokenizer.decode(prefix_ids, **decode_kwargs)
        
        if len(prefix_text) <= target_char_start:
            found_gen_idx = i
            break
            
    if found_gen_idx == -1: return None, None

    return score_val, prompt_len + found_gen_idx


def preprocess_signals(raw_ppl, raw_variance):
    """
    raw_ppl: vector of shape (N,)
    raw_variance: vector of shape (N,)
    """
    
    # --- 1. Handle Perplexity (Log-Normal -> Normal) ---
    # We log it first to squash outliers and linearize the "confusion" metric
    log_ppl = np.log(raw_ppl + 1e-6) # add epsilon just in case
    
    # Standardize (Z-Score)
    z_prompt = (log_ppl - np.mean(log_ppl)) / (np.std(log_ppl) + 1e-6)
    
    # --- 2. Handle Option Variance (Small Scale -> Normal) ---
    # Usually doesn't need log, just scaling
    z_option = (raw_variance - np.mean(raw_variance)) / (np.std(raw_variance) + 1e-6)
    
    return z_prompt, z_option


class CalibrationScaler:
    def __init__(self, metric_config):
        """
        metric_config: Dictionary defining the metrics and their preprocessing needs.
        
        Example:
        {
            'prompt_loss':      {'log': True},  # Log-Normal distribution
            'option_variance':  {'log': False}, # Already small/linear
            'semantic_entropy': {'log': False}  # Already linear
        }
        """
        self.config = metric_config
        
        # Structure: { 'model_name': { 'metric_name': {'mu': ..., 'std': ...} } }
        self.registry = {} 
        
        # Fallback for new/unknown models
        self.global_stats = {} 

    def _safe_log(self, data):
        """Helper to apply log with epsilon safety."""
        return np.log(np.array(data) + 1e-8)

    def fit(self, df, model_col='model_name'):
        """
        Computes training stats for all configured metrics, per model.
        df: Pandas DataFrame containing the columns defined in self.config
        """
        # 1. Compute Global Stats (Fallback)
        for metric, params in self.config.items():
            data = df[metric].values
            
            # Apply Log if configured
            if params.get('log', False):
                if metric == 'entropy':
                    data = np.log1p(data)  # log(1 + x) for entropy to avoid variance inflation
                    print("Applied log1p transformation for entropy metric.")
                else:
                    data = self._safe_log(data)
                
            self.global_stats[metric] = {
                'mu': np.mean(data),
                'std': np.std(data)
            }

        # 2. Compute Per-Model Stats
        unique_models = df[model_col].unique()
        
        for m in unique_models:
            subset = df[df[model_col] == m]
            self.registry[m] = {}
            
            for metric, params in self.config.items():
                data = subset[metric].values
                
                # Apply Log if configured
                if params.get('log', False):
                    data = self._safe_log(data)
                
                self.registry[m][metric] = {
                    'mu': np.mean(data),
                    'std': np.std(data)
                }
                
        print(f"Fitted scaler for {len(unique_models)} models and {len(self.config)} metrics.")

    def transform(self, df, model_col='model_name'):
        """
        Transforms data using the FROZEN training stats.
        Returns a Dictionary of {metric_name: z_score_array}
        """
        # Iterate row by row (or model-group by model-group for speed)
        # Here we do vectorization by model group which is much faster
        
        # We need to preserve the original index order, so we'll store results in a temporary series
        # or just iterate if the dataset isn't massive.
        # For simplicity and readability in research, let's iterate by unique models in the input.
        
        # Prepare empty arrays aligned with input dataframe length
        n_samples = len(df)
        final_z_scores = {k: np.zeros(n_samples) for k in self.config.keys()}
        
        # We process each model batch separately to apply the correct "ruler"
        unique_models = df[model_col].unique()
        
        for m in unique_models:
            # Indices for this model
            idx = df.index[df[model_col] == m]
            
            # Get the ruler (stats) for this model
            # If model unknown, use global fallback
            stats_bundle = self.registry.get(m, self.global_stats)
            
            for metric, params in self.config.items():
                raw_values = df.loc[idx, metric].values
                
                # 1. Log Transform?
                if params.get('log', False):
                    raw_values = self._safe_log(raw_values)
                
                # 2. Z-Score using SAVED stats
                mu = stats_bundle[metric]['mu']
                std = stats_bundle[metric]['std']
                
                z_values = (raw_values - mu) / (std + 1e-8)
                
                # Assign back to correct positions
                final_z_scores[metric][df.index.get_indexer(idx)] = z_values

        return final_z_scores

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.registry, self.global_stats, self.config), f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            self.registry, self.global_stats, self.config = pickle.load(f)