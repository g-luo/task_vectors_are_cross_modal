import inspect
import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    BitsAndBytesConfig
)

# ===========================
#        VLM Loaders
# ===========================
class Idefics2Wrapper(torch.nn.Module):
    def __init__(self, text_model):
        super().__init__()
        self.model = text_model
        self.config = text_model.config
        
class LLMProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, text=None, images=None, **kwargs):
        return self.tokenizer(text, **kwargs)
    
def remove_cache(module):
    def new_forward(self, **kwargs):
        outputs = self.old_forward(**kwargs)
        outputs["past_key_values"] = None
        return outputs
    if not hasattr(module, "old_forward"):
        module.old_forward = module.forward
    # Fix hf method signature complaints
    new_forward.__signature__ = inspect.signature(module.old_forward)
    module.forward = new_forward.__get__(module, type(module))
    
def get_quantization_config(model_kwargs):
    torch_dtype = torch.float16
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_storage=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )
    return model_kwargs

def get_model_config(model):
    model_config = {
        "name_or_path": model.config._name_or_path,
        "n_heads": model.config.num_attention_heads,
        "n_layers": model.config.num_hidden_layers,
        "resid_dim": model.config.hidden_size
    }
    return model_config
    
def load_text_model(model_id, model):
    text_model_loaders = {
        "idefics2": lambda model: Idefics2Wrapper(model.model.text_model),
        "llava": lambda model: model.language_model,
        "Mantis": lambda model: model.language_model,
    }
    text_model_loader = None
    for k, v in text_model_loaders.items():
        if k in model_id:
            text_model_loader = v
            break
    if text_model_loader is None:
        return model
    else:
        text_model = text_model_loader(model)
        return text_model

def load_hf_model(model_id, device="cuda", load_in_4bit=True):
    model_kwargs = {}
    if load_in_4bit:
        model_kwargs = get_quantization_config(model_kwargs)
    if "Mantis" in model_id:
        from mantis.models.mfuyu import MFuyuForCausalLM, MFuyuProcessor
        processor = MFuyuProcessor.from_pretrained(model_id)
        model = MFuyuForCausalLM.from_pretrained(model_id, device_map=device, **model_kwargs)
    elif "mistral" in model_id or "vicuna" in model_id:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map=device, **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        processor = LLMProcessor(tokenizer)
    else:
        model = AutoModelForVision2Seq.from_pretrained(model_id, device_map=device, **model_kwargs)
        processor = AutoProcessor.from_pretrained(model_id)
        processor.image_processor.do_image_splitting = False
    return model, processor

# ===========================
#         Generation
# ===========================
def prepare_batch(inputs, device):
    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to(device)
        if type(v) is list:
            inputs[k] = [v_.to(device) if torch.is_tensor(v_) else v_ for v_ in v]
    return inputs

def generate(model, processor, inputs, remove_input=True, **generate_kwargs):
    with torch.no_grad():
        inputs = prepare_batch(inputs, model.device)
        output = model.generate(**inputs, **generate_kwargs)
        if remove_input:
            input_len = inputs["input_ids"].shape[1]
            output = output[:, input_len:]
        output = processor.tokenizer.batch_decode(output, skip_special_tokens=True)
    return output

def find_token(processor, input_ids, token_ids=None, token=None, token_offset=0, match="last", pad_id=0):
    if token is None:
        idxs = (input_ids != pad_id).sum() - (token_offset + 1)
        idxs = idxs[None, ...]
        return idxs
    else:
        # Replace the last instance in src and first in tgt
        if token_ids is None:
            token_ids = processor.tokenizer(token)["input_ids"][1:]
        token_ids = torch.tensor(token_ids, device=input_ids.device)
        windows = input_ids.unfold(0, len(token_ids), 1)
        matches = torch.all(windows == token_ids, dim=1)
        matches = torch.where(matches)[0]
        if match == "last":
            matches = matches[-1]
        elif match == "first":
            matches = matches[0]
        else:
            matches = matches[match]
        matches = matches.item()
        idxs = torch.arange(matches, matches + len(token_ids))
        return idxs

# ===========================
#          Patching
# ===========================
"""
Patching adapted from Function Vectors
(Todd et. al., ICLR 2024)
https://github.com/ericwtodd/function_vectors
"""
def patch_output_pair(src_idx=0, tgt_idx=0, src_token=-1, tgt_token=-1, L=None, cache_L=None, feats=None):
    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split(".")[2])
        return_cache = type(output) is tuple
        if return_cache:
            act = output[0]
            cache = output[1]
        else:
            act = output
        if act.shape[1] == 1:
            print("WARNING: cache seems to be in use")
        if L == current_layer:
            act[tgt_idx, tgt_token] = feats[cache_L][src_idx, src_token].detach()
        if return_cache:
            output = (act, cache)
        else:
            output = act
        return output
    return rep_act

# ===========================
#          Metrics
# ===========================
"""
Metrics adapted from In-Context Learning 
Creates Task Vectors (Hendel et. al., EMNLP Findings 2023)
https://github.com/roeehendel/icl_task_vectors
"""

def preprocess_text(lst, allow_lower_case=False):
    lst = [x.strip() for x in lst]
    if  allow_lower_case:
        lst = [x.lower() for x in lst]
    return lst

def compare_outputs(output1, output2):
    output1, output2 = output1.strip(), output2.strip()
    nonempy = len(output1) > 0 and len(output2) > 0
    return nonempy and (output1.startswith(output2) or output2.startswith(output1))

def compute_text_acc(pred_text, labels_text, allow_lower_case=False):
    pred_text = preprocess_text(pred_text, allow_lower_case=allow_lower_case)
    labels_text = preprocess_text(labels_text, allow_lower_case=allow_lower_case)
    vectorized_compare = np.vectorize(compare_outputs)
    correct = vectorized_compare(pred_text, labels_text)
    acc = correct.mean()
    return acc

# ===========================
#         Logging
# ===========================
def get_max(x, verbose=False):
    max_value = np.max(x)
    max_indices = np.argwhere(x == max_value)
    max_index = max_indices[np.argmin(max_indices[:, 0])]
    if verbose:
        print("All indices with the maximum value:", max_indices)
    return max_index

def avg_seed_accuracy(file_prefix, patch_L=None, verbose=True, allow_lower_case=False):
    baseline = []
    regular = []
    hypothesis = {}
    files = glob.glob(f"{file_prefix}*")
    if len(files) == 0:
        raise Exception("No files found!")
    for file in files:
        results = json.load(open(file))
        baseline.append(compute_text_acc(results["baseline_text"], results["labels_text"], allow_lower_case=allow_lower_case))
        regular.append(compute_text_acc(results["regular_text"], results["labels_text"], allow_lower_case=allow_lower_case))
        for L in range(len(results["hypothesis_text"])):
            if patch_L is not None and L != patch_L:
                acc = 0
            else:
                results["hypothesis_text"][L] = [x if x is not None else "" for x in results["hypothesis_text"][L]]
                acc = compute_text_acc(results["hypothesis_text"][L], results["labels_text"], allow_lower_case=allow_lower_case)
            hypothesis[L] = hypothesis.get(L, []) + [acc]
    baseline = np.mean(baseline)
    regular = np.mean(regular)
    hypothesis = [np.mean(v) for v in hypothesis.values()]
    max_idx = get_max(hypothesis).item()
    if verbose:
        print(f"===== Accuracy ({len(files)} Seeds) =====")
        print(f"Baseline: {baseline:.2f}")
        print(f"Regular: {regular:.2f}")
        print(f"Hypothesis (L={max_idx}): {hypothesis[max_idx]:.2f}")
    return baseline, regular, hypothesis, max_idx

def avg_task_accuracy(file_prefixes, patch_L=None, verbose=True, allow_lower_case=False):
    all_baseline, all_regular, all_hypothesis = [], [], []
    for file_prefix in file_prefixes:
        baseline, regular, hypothesis, _ = avg_seed_accuracy(file_prefix, patch_L=patch_L, verbose=False, allow_lower_case=allow_lower_case)
        all_baseline.append(baseline)
        all_regular.append(regular)
        all_hypothesis.append(hypothesis)
    all_baseline_avg = np.mean(all_baseline)
    all_regular_avg = np.mean(all_regular)
    # For the same index in each list, compute the mean
    all_hypothesis_avg = [np.mean([hypothesis[L] for hypothesis in all_hypothesis]) for L in range(len(all_hypothesis[0]))]
    max_idx = get_max(all_hypothesis_avg).item()
    if verbose:
        print(f"===== Accuracy ({len(file_prefixes)} Tasks) =====")
        print(f"Baseline: {all_baseline_avg:.2f}")
        print(f"Regular: {all_regular_avg:.2f}")
        print(f"Hypothesis (L={max_idx}): {all_hypothesis_avg[max_idx]:.2f}")
    acc_info = {
        "raw": [all_baseline, all_regular, all_hypothesis],
        "accuracy": [all_baseline_avg, all_regular_avg, all_hypothesis_avg[max_idx]],
        "max_idx": max_idx
    }
    return acc_info

def plot_icl(text, images):
    n = len(text)
    width = n * 4
    if images:
        height = 4
    else:
        height = 1
    fig, axes = plt.subplots(1, n, figsize=(width, height))
    if n == 1:
        axes = [axes]
    for i in range(len(text)):
        if images:
            axes[i].imshow(images[i])
        axes[i].axis('off')
        axes[i].set_title(text[i], wrap=True, size=24)
    plt.tight_layout()
    plt.show()