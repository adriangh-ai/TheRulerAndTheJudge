import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_polygraph.estimators import semantic_entropy, SAR, mahalanobis_distance
from lm_polygraph import WhiteboxModel
from lm_polygraph import estimate_uncertainty

def get_calibrated_score_tokens(tokenizer, json_prefix):
    """
    Returns a dictionary mapping int(1-5) to the correct token_id 
    based on the whitespace in the json_prefix.
    """
    # 1. Get IDs for both variants
    # Note: We use the generic encode to handle potential BOS/EOS wrapper logic, 
    # then take the last token.
    
    # Variant A: "1", "2"... (Use this if prefix has space)
    ids_no_space = {
        i: tokenizer.encode(str(i), add_special_tokens=False)[-1] 
        for i in range(1, 6)
    }
    
    # Variant B: " 1", " 2"... (Use this if prefix has NO space)
    ids_with_space = {
        i: tokenizer.encode(" " + str(i), add_special_tokens=False)[-1] 
        for i in range(1, 6)
    }
    
    # 2. Check overlap (Some tokenizers like Llama 2 might map "1" and " 1" to same ID)
    distinct_vocab = (ids_no_space[1] != ids_with_space[1])
    
    # 3. Detect intent from prefix
    if json_prefix.endswith(" "):
        # Prefix: '... "score": ' -> Next token should be "5"
        selected_map = ids_no_space
        variant_name = "No-Space Digits (e.g., '5')"
    else:
        # Prefix: '... "score":' -> Next token should be " 5"
        selected_map = ids_with_space
        variant_name = "Leading-Space Digits (e.g., ' 5')"

    # 4. Debug Output for Researcher
    print(f"--- Tokenizer Calibration ---")
    print(f"Prefix ends with space? {json_prefix.endswith(' ')}")
    print(f"Distinct whitespace tokens? {distinct_vocab}")
    print(f"Selected Token Variant: {variant_name}")
    print(f"Target IDs: {selected_map}")
    print("-----------------------------")
    
    return selected_map


def get_teacher_forced_confidence(model, tokenizer, prompt_text, generated_reasoning):
    """
    Calculates the normalized probability distribution over scores 1-5 
    given the prompt and reasoning.
    """
    model.eval()
    
    # ------------------------------------------------------------------
    # Step 1: Define the Prefix
    # ------------------------------------------------------------------
    # We construct the text exactly up to the point where the number is expected.
    # CRITICAL: We add a space after the colon if that's how the model usually sees JSON.
    # Most models predict " 5" (space+5) if the input ends in ":", 
    # or "5" if the input ends in ": ". 
    # Let's enforce the prefix ending with ": " to target pure number tokens.
    
    json_prefix = ' {"score": ' 
    score_token_map = get_calibrated_score_tokens(tokenizer, json_prefix)
    target_ids = list(score_token_map.values())

    full_context = prompt_text + generated_reasoning + json_prefix
    
    # ------------------------------------------------------------------
    # Step 2: Identify the Target Token IDs
    # ------------------------------------------------------------------
    # We need the IDs for ["1", "2", "3", "4", "5"]
    # We assume the tokenizer handles single digits as single tokens (true for Llama, GPT, etc.)
    # target_map = {}
    # for i in range(1, 6):
    #     # We encode strictly the number string. 
    #     # CAUTION: Some tokenizers add a BOS token or handle spaces weirdly.
    #     # We assume the context ends in a space, so we look for the token "1", not " 1".
        
    #     # This generic way usually finds the correct single ID:
    #     # We encode "1" and take the last token ID (ignoring potential BOS at start)
    #     token_id = tokenizer.encode(str(i), add_special_tokens=False)[-1]
    #     target_map[i] = token_id
        
    # # Create a tensor of these IDs for easy indexing
    # # keys: 1..5, values: token_ids
    # target_ids = list(target_map.values()) # e.g. [29896, 29906, ...]
    
    # ------------------------------------------------------------------
    # Step 3: Forward Pass
    # ------------------------------------------------------------------
    inputs = tokenizer(full_context, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
        # Get the logits of the very last token
        # Shape: [1, vocab_size]
        last_token_logits = outputs.logits[0, -1, :]
        
    # ------------------------------------------------------------------
    # Step 4: Extract and Normalize
    # ------------------------------------------------------------------
    # Extract logits only for our target tokens [1, 2, 3, 4, 5]
    score_logits = last_token_logits[target_ids]
    
    # Apply Softmax to get a probability distribution that sums to 1.0 over these 5 options
    probs = F.softmax(score_logits, dim=0)
    
    # Pack into a nice dictionary
    result = {
        score_val: probs.item() for score_val, prob in zip(score_token_map.keys(), probs)
    }
    
    # Optional: Calculate expected value (weighted average)
    expected_score = sum(s * p for s, p in result.items())
    
    return result, expected_score

# Usage Example
# prompt = "Rate this movie..."
# reasoning = "The acting was great but the plot was thin."
# dist, score = get_teacher_forced_confidence(model, tokenizer, prompt, reasoning)
# print(dist) 
# -> {1: 0.01, 2: 0.10, 3: 0.70, 4: 0.15, 5: 0.04}



def get_confidence_from_ids(model, tokenizer, full_ids, anchor_str='{"score":'):
    """
    Calculates the confidence distribution for a score (1-5) embedded in a full sequence 
    of token IDs, using the anchor_str to locate the score position.
    
    Args:
        full_ids (torch.Tensor): The complete sequence [Prompt + Reasoning + JSON]. Shape [1, N]
        anchor_str (str): The string immediately preceding the score (e.g., '{"score":')
        
    Returns:
        dict: Probability distribution {1: p1, ..., 5: p5}
        float: Expected Value
    """
    model.eval()
    
    # ------------------------------------------------------------------
    # 1. Prepare Target Tokens (1-5)
    # ------------------------------------------------------------------
    # We need to check if the score token in full_ids has a leading space or not.
    # This depends on how the tokenizer handled the merge between the anchor and the number.
    
    # Let's find the anchor first to see what the NEXT token looks like.
    anchor_ids = tokenizer.encode(anchor_str, add_special_tokens=False)
    
    # Convert to lists for easy substring search
    full_list = full_ids[0].tolist()
    anchor_len = len(anchor_ids)
    
    # Find the start index of the anchor in the full sequence
    # We search backwards because the score is usually at the end
    start_idx = -1
    for i in range(len(full_list) - anchor_len, -1, -1):
        if full_list[i : i + anchor_len] == anchor_ids:
            start_idx = i
            break
            
    if start_idx == -1:
        raise ValueError(f"Anchor '{anchor_str}' not found in the sequence IDs.")
        
    # The score token is immediately after the anchor
    score_pos = start_idx + anchor_len
    
    if score_pos >= len(full_list):
        raise ValueError("Anchor found at the very end; no score token follows it.")
        
    actual_score_token_id = full_list[score_pos]
    
    # ------------------------------------------------------------------
    # 2. Determine which set of number tokens to check (Space vs No-Space)
    # ------------------------------------------------------------------
    # We check if the actual generated token matches "5" or " 5"
    # This automatically handles the "space trap" without guessing.
    
    digits = [str(i) for i in range(1, 6)]
    
    # Get IDs for "1"..."5" (No space)
    ids_no_space = [tokenizer.encode(d, add_special_tokens=False)[-1] for d in digits]
    # Get IDs for " 1"..." 5" (Leading space)
    ids_with_space = [tokenizer.encode(" " + d, add_special_tokens=False)[-1] for d in digits]
    
    target_ids = []
    
    if actual_score_token_id in ids_no_space:
        target_ids = ids_no_space
        # print("Detected format: No-Space (e.g., '5')")
    elif actual_score_token_id in ids_with_space:
        target_ids = ids_with_space
        # print("Detected format: Leading-Space (e.g., ' 5')")
    else:
        # Fallback: The model might have generated something else, or the tokenization is complex.
        # We default to the set that 'looks' most like the token we found, 
        # or simply check both if we want to be safe (but usually one set is correct contextually).
        # Heuristic: Check which set is closer in vector space? No, too slow.
        # Simple Heuristic: Assume Leading Space if anchor ends in ":"
        if anchor_str.endswith(" "):
            target_ids = ids_no_space
        else:
            target_ids = ids_with_space
        # print(f"Warning: Actual token {actual_score_token_id} not in 1-5 list. Using heuristic.")

    # ------------------------------------------------------------------
    # 3. Forward Pass (Teacher Forcing)
    # ------------------------------------------------------------------
    # We input everything UP TO the score token.
    # We want the model to predict the token at `score_pos`.
    # So input is full_ids[0, :score_pos]
    
    input_ids = full_ids[:, :score_pos] # Shape [1, score_pos]
    
    with torch.no_grad():
        outputs = model(input_ids)
        # Logits for the next token (which would be the score)
        # Shape: [1, vocab_size]
        next_token_logits = outputs.logits[0, -1, :]
        
    # ------------------------------------------------------------------
    # 4. Extract & Normalize
    # ------------------------------------------------------------------
    # Select only the 5 target logits
    score_logits = next_token_logits[target_ids]
    
    # Softmax
    probs = F.softmax(score_logits, dim=0)
    
    # Create result dict
    dist = {i+1: probs[i].item() for i in range(5)}
    
    # Calculate Expected Score
    expected_score = sum(score * prob for score, prob in dist.items())
    
    return dist, expected_score

# Usage Example:
# full_ids = model.generate(**inputs) # Returns [Prompt + Response]
# dist, score = get_confidence_from_ids(model, tokenizer, full_ids)


def calculate_uncertainty_at_index(model, tokenizer, full_ids, target_index):
    """
    Teacher forces the model up to `target_index` and calculates prob dist over 1-5.
    """
    model.eval()
    
    # 1. Prepare Input (Up to the token BEFORE the score)
    # The model predicts t from 0..t-1. 
    # To predict the token at target_index, we feed full_ids[0, :target_index]
    input_ids = full_ids[:, :target_index]
    
    # 2. Determine Calibration (Space vs No-Space)
    # Check the ACTUAL token the model generated to know which ID set to check
    actual_token_id = full_ids[0, target_index].item()
    
    digits = [str(i) for i in range(1, 6)]
    ids_no_space = [tokenizer.encode(d, add_special_tokens=False)[-1] for d in digits]
    ids_with_space = [tokenizer.encode(" " + d, add_special_tokens=False)[-1] for d in digits]
    
    if actual_token_id in ids_no_space:
        target_ids = ids_no_space
    elif actual_token_id in ids_with_space:
        target_ids = ids_with_space
    else:
        # Fallback: Model output valid digit but tokenizer is weird. 
        # Default to no-space as it's cleaner for JSON.
        target_ids = ids_no_space

    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(input_ids)
        next_token_logits = outputs.logits[0, -1, :]
        
    # 4. Extract Distribution
    score_logits = next_token_logits[target_ids]
    probs = F.softmax(score_logits, dim=0)
    
    dist = {i+1: probs[i].item() for i in range(5)}
    expected_score = sum(s * p for s, p in dist.items())
    
    return dist, expected_score

def calculate_uncertainty_metrics(model, tokenizer, full_ids, target_index):
    """
    Teacher forces the model up to `target_index` and calculates multiple uncertainty 
    signals on the score (1-5), including Margin and Entropy.
    """
    model.eval()
    
    # 1. Prepare Input (Up to the token BEFORE the score)
    input_ids = full_ids[:, :target_index]
    
    # 2. Determine Calibration (Space vs No-Space)
    # Check the ACTUAL token the model generated
    actual_token_id = full_ids[0, target_index].item()
    
    digits = [str(i) for i in range(1, 6)]
    ids_no_space = [tokenizer.encode(d, add_special_tokens=False)[-1] for d in digits]
    ids_with_space = [tokenizer.encode(" " + d, add_special_tokens=False)[-1] for d in digits]
    
    if actual_token_id in ids_no_space:
        target_ids = ids_no_space
    elif actual_token_id in ids_with_space:
        target_ids = ids_with_space
    else:
        # Fallback
        target_ids = ids_no_space

    # 3. Forward Pass
    with torch.no_grad():
        outputs = model(input_ids)
        # Get raw logits (unnormalized scores)
        next_token_logits = outputs.logits[0, -1, :].float()
        
    # Extract logits for scores 1-5
    score_logits = next_token_logits[target_ids]
    
    # ---------------------------------------------------------
    # Metric A: Standard Probability Distribution & Expected Score
    # ---------------------------------------------------------
    probs = F.softmax(score_logits.float(), dim=0)
    dist = {i+1: probs[i].item() for i in range(5)}
    
    expected_score = sum(s * p for s, p in dist.items())

    # ---------------------------------------------------------
    # Metric B: Logit Margin (The "Gap")
    # ---------------------------------------------------------
    # We sort the raw logits to find the winner and the runner-up.
    # A small gap implies the model struggled to decide between two scores
    # (e.g., 3 vs 4) even if the final probability looks like 1.0.
    sorted_logits, _ = torch.sort(score_logits.float(), descending=True)
    
    top_1_logit = sorted_logits[0].item()
    top_2_logit = sorted_logits[1].item()
    
    # Margin = Distance between Winner and Runner-Up
    # High Margin = Absolute certainty. Low Margin = "It was a close call."
    logit_margin = top_1_logit - top_2_logit

    # ---------------------------------------------------------
    # Metric C: Entropy
    # ---------------------------------------------------------
    # Measures the "spread" of the distribution.
    # 0.0 = All mass on one score. High value = Mass spread across many scores.
    # We use a clamp to avoid log(0) errors.
    
    # safe_probs: [5]
    p = probs + 1e-12 # numerical stability
    entropy = -torch.sum(p * torch.log(p)).item()

    variance = sum([p * ((v - expected_score)**2) for p, v in dist.items()])

    dist = {str(k): v for k, v in dist.items()}

    return {
        "distribution": dist,
        "expected_score": expected_score,
        "logit_margin": logit_margin,
        "entropy": entropy,
        "variance": variance,
        "score_logits": score_logits.float().cpu().tolist()
    }

def prompt_ppl(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    # Input can now be a string OR a torch.Tensor/List[int]
    input_data, 
):
    model.eval()

    # 1. Handle Tensors (The "Golden Path" for your workflow)
    if isinstance(input_data, (torch.Tensor, list)):
        # If it's a list of IDs, convert to tensor
        if isinstance(input_data, list):
            input_ids = torch.tensor([input_data])
        else:
            input_ids = input_data
            
        # Ensure it is 2D [batch, seq_len]
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # Create attention mask (assume all 1s if passing raw IDs unless padding is explicit)
        # Note: If passing raw outputs from generate, padding might exist.
        if tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids)
            
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    # 2. Handle Strings (Your old logic)
    elif isinstance(input_data, str):

        inputs = tokenizer(input_data, return_tensors='pt', padding=True, truncation=True)
    else:
        raise ValueError("Input must be a string or input_ids tensor")

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 3. Label Masking (Standard Logic)
    labels = inputs['input_ids'].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    # 4. Calculation
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        token_loss = token_loss.view(labels.size(0), -1)
        non_pad_tokens = (shift_labels != -100).sum(dim=1)
        sequence_loss = token_loss.sum(dim=1) / non_pad_tokens.clamp(min=1)
        perplexity = torch.exp(sequence_loss)

    return perplexity.item() if perplexity.numel() == 1 else perplexity


def prompt_ppl_bpc_score(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    input_data
):
    """
    Calculates Perplexity and Bits Per Character (BPC) for OOD detection.
    Accepts strings or pre-tokenized tensors (to avoid detokenization traps).
    """
    model.eval()
    
    # ------------------------------------------------------------------
    # 1. Input Handling (Matches your prompt_ppl structure)
    # ------------------------------------------------------------------
    num_chars = 0
    
    # Case A: Tensors (The "Golden Path")
    if isinstance(input_data, (torch.Tensor, list)):
        if isinstance(input_data, list):
            input_ids = torch.tensor([input_data])
        else:
            input_ids = input_data
            
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        # Attention Mask
        if tokenizer.pad_token_id is not None:
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        else:
            attention_mask = torch.ones_like(input_ids)
            
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # for BPC, we MUST know character count. 
        # We decode strictly to count chars, but use original IDs for loss.
        # skip_special_tokens=True ensures we don't count <s> or padding as "content"
        decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        num_chars = len(decoded_text)

    # Case B: String
    elif isinstance(input_data, str):
        # Add special tokens for correct causal masking
        inputs = tokenizer(input_data, return_tensors='pt', add_special_tokens=True)
        num_chars = len(input_data)
        
    else:
        raise ValueError("Input must be a string or input_ids tensor")

    # Move to device
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ------------------------------------------------------------------
    # 2. Label Masking
    # ------------------------------------------------------------------
    labels = inputs['input_ids'].clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    # ------------------------------------------------------------------
    # 3. Calculation
    # ------------------------------------------------------------------
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Shift for Causal LM
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Element-wise loss
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Reshape to [batch, seq_len]
        token_loss = token_loss.view(labels.size(0), -1)
        
        # --------------------------------------------------------------
        # Metric 1: Perplexity (Token-Normalized)
        # --------------------------------------------------------------
        non_pad_tokens = (shift_labels != -100).sum(dim=1)
        # Avoid division by zero
        mean_token_loss = token_loss.sum(dim=1) / non_pad_tokens.clamp(min=1)
        perplexity = torch.exp(mean_token_loss)

        # --------------------------------------------------------------
        # Metric 2: Bits Per Character (BPC) - Length Normalized
        # --------------------------------------------------------------
        # Total Negative Log Likelihood (Sum of all token losses)
        total_nll = token_loss.sum(dim=1).item() # scalar for batch size 1
        
        if num_chars > 0:
            # Convert Nats (ln) to Bits (log2): value * log2(e) ≈ value * 1.44
            bpc = (total_nll / num_chars) * 1.442695
        else:
            bpc = 0.0

    # Return dict for easy inspection
    return {
        "ppl": perplexity.item() if perplexity.numel() == 1 else perplexity,
        "bpc": bpc
    }


def semantic_entropy_estimator(model_id, model, tokenizer, input_text):
    """Estimate sequence-level semantic entropy with LM-Polygraph."""
    ue_method = semantic_entropy.SemanticEntropy()
    model_wb = WhiteboxModel(model, tokenizer, model_path=model_id)
    ue = estimate_uncertainty(model_wb, ue_method, input_text=input_text)
    return ue.uncertainty


def sar_estimator(model_id, model, tokenizer, input_text):
    """Estimate sequence uncertainty using the SAR estimator."""

    model_wb = WhiteboxModel(model, tokenizer, model_path=model_id)
    ue_method = SAR()
    
    ue = estimate_uncertainty(model_wb, ue_method, input_text=input_text)
    return ue.uncertainty

def mahalanobis_estimator(model_id, model, tokenizer, input_text):
    """Estimate sequence uncertainty using Mahalanobis distance."""

    model_wb = WhiteboxModel(model, tokenizer, model_path=model_id)
    ue_method = mahalanobis_distance.MahalanobisDistanceSeq()
    
    ue = estimate_uncertainty(model_wb, ue_method, input_text=input_text)
    return ue.uncertainty