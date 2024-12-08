from baukit import TraceDict
import copy
import json
import numpy as np
from omegaconf import OmegaConf
import os
import pandas as pd
import sys
import torch
from tqdm import tqdm

import xpatch_helpers, xpatch_dataset

SEED_START = 1

# ===========================
#       Feature Loading
# ===========================
def get_file(folder, task=None, seed=None, filetype=None):
    file = []
    if task is not None:
        file += [f"task-{task}"]
    if seed is not None:
        file += [f"seed-{seed}"]
    file = "_".join(file)
    if filetype:
        file += filetype
    return f"{folder}/{file}"
    
def open_feats(feats_file):
    if not os.path.exists(feats_file):
        raise Exception(f"File {feats_file} does not exist!")
    else:
        print(f"Loading feats, {feats_file}")
    feats = torch.load(feats_file)
    feats = feats.detach().cpu()
    return feats

def postprocess_feats(feats, config, seed):
    if config.get("save_feats_all", False):
        save_feats_subset = config.get("save_feats_subset")
        if save_feats_subset:
            np.random.seed(seed)
            random_idxs = np.random.permutation(range(feats.shape[0]))[:save_feats_subset]
            feats = feats[random_idxs]
        feats = feats.mean(dim=0)
    return feats

def get_feats(config, task, seed):
    feats_file = get_file(config['feats_folder'], task, seed, ".pt")
    feats = open_feats(feats_file)
    feats = postprocess_feats(feats, config, seed)
    # Ensemble instruction feats (optional)
    if config.get("ensemble"):
        ensemble_feats_folder = config['feats_folder'].replace("icl", "instruction")
        ensemble_feats_file = get_file(ensemble_feats_folder, task, SEED_START, ".pt")
        ensemble_feats = open_feats(ensemble_feats_file)
        feats = (feats + ensemble_feats) / 2
    return feats

# ===========================
#       Feature Saving
# ===========================
def embed_prompt(model, processor, text_model, model_config, config, text, images=None):
    # Assumes the special token is the last one
    src_token = -1
    src_batch = processor(text=text, images=images, return_tensors="pt")
    with TraceDict(text_model, layers=model_config["layer_hook_names"], retain_output=True) as ret:
        model.forward(**src_batch)
        feats = [ret[k].output[0] for k in model_config["layer_hook_names"]]
        feats = [feat[:, src_token, :][:, None, :] for feat in feats]
        feats = [feat.detach().cpu() for feat in feats]
        feats = torch.stack(feats)
    instruction_feats = [feats]
    return instruction_feats

def embed_dataset(model, processor, text_model, model_config, config, icl_pair_xpatch_dataset):
    dataset_feats = []
    for i in tqdm(range(len(icl_pair_xpatch_dataset))):
        src_batch, src_meta = icl_pair_xpatch_dataset.__getitem__(i, "src")
        with TraceDict(text_model, layers=model_config["layer_hook_names"], retain_output=True) as ret:
            xpatch_helpers.generate(model, processor, src_batch, **config["generate_kwargs"])
        feats = [ret[k].output[0] for k in model_config["layer_hook_names"]]
        feats = [feat[:, src_meta["src_token"], :] for feat in feats]
        feats = [feat.detach().cpu() for feat in feats]
        feats = torch.stack(feats)
        dataset_feats.append(feats)
    return dataset_feats

def save_feats(model, processor, text_model, model_config, config, task, seed, icl_pair_xpatch_dataset, instruction):
    # Create the feats
    if config["spec"] == "icl":
        feats = embed_dataset(model, processor, text_model, model_config, config, icl_pair_xpatch_dataset)
    elif config["spec"] == "instruction":
        feats = embed_prompt(model, processor, text_model, model_config, config, instruction)
    else:
        raise NotImplementedError("Config does not have a valid spec!")
    # Average across all feats
    feats = torch.stack(feats)
    if not config.get("save_feats_all", False):
        feats = feats.mean(dim=0)
    # Write the feats to file
    feats_file = get_file(config['feats_folder'], task, seed, ".pt")
    save_feats_folder = os.path.dirname(feats_file)
    if not os.path.exists(save_feats_folder):
        os.makedirs(save_feats_folder, exist_ok=True)
    torch.save(feats, feats_file)

# ===========================
#          Patching
# ===========================
def patch_layer(model, processor, text_model, model_config, generate_kwargs, feats, L, cache_L, tgt_batch, tgt_token, patch_L=None, return_feats=False):
    if patch_L and L != patch_L:
        return None
    intervention_fn = xpatch_helpers.patch_output_pair(
        tgt_token=tgt_token,
        L=L,
        cache_L=cache_L,
        feats=feats
    )
    with TraceDict(text_model, layers=model_config['layer_hook_names'], edit_output=intervention_fn, retain_output=return_feats) as ret:
        # Convert to text and back to input_ids
        # since tokenization is different based on position
        output_text = xpatch_helpers.generate(model, processor, tgt_batch, **generate_kwargs)
        output_text = output_text[0]
    if return_feats:
        feats = [ret[k].output[0] for k in model_config['layer_hook_names']]
        return output_text, feats 
    else:
        return output_text
    
def evaluate(model, processor, text_model, model_config, config, task, seed, icl_pair_xpatch_dataset):
    use_patching = config.get("use_patching", True)
    patch_L = config.get("patch_L")

    # Create save folder
    save_file = get_file(config['save_folder'], task, seed, ".json")
    save_folder = os.path.dirname(save_file)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=True)
    save_keys = ["baseline_text", "regular_text", "labels_text", "train_idxs"]
    results = {k: [] for k in save_keys}
    results["hypothesis_text"] = [[] for _ in range(model_config["n_layers"])]
    results["config"] = OmegaConf.to_container(config, resolve=True)
    json.dump(results, open(save_file, "w"))
    
    # Load cached feats
    feats = get_feats(config, task, seed)
    feats = [feat.to(model.device).to(model.dtype) for feat in feats]

    for i in tqdm(range(len(icl_pair_xpatch_dataset))):
        src_batch, src_meta = icl_pair_xpatch_dataset.__getitem__(i, "src")
        tgt_batch, tgt_meta = icl_pair_xpatch_dataset.__getitem__(i, "tgt")
        # Run Regular
        output_text = xpatch_helpers.generate(model, processor, src_batch, **config["generate_kwargs"])
        results["regular_text"].append(output_text[0])
        # Run Baseline
        output_text = xpatch_helpers.generate(model, processor, tgt_batch, **config["generate_kwargs"])
        results["baseline_text"].append(output_text[0])
        # Patch Task Vector
        if use_patching:
            patch_layer_kwargs = {
                "model": model,
                "processor": processor,
                "text_model": text_model,
                "model_config": model_config,
                "generate_kwargs": config["generate_kwargs"],
                "feats": feats,
                "tgt_batch": tgt_batch,
                "tgt_token": tgt_meta["tgt_token"]
            }
            layer_results = [patch_layer(L=L, patch_L=patch_L, cache_L=L, **patch_layer_kwargs) for L in tqdm(range(model_config["n_layers"]))]
            for L, layer_result in enumerate(layer_results):
                results["hypothesis_text"][L].append(layer_result)
        # Save metadata
        labels = tgt_meta["labels"]
        train_idxs = src_meta["train_idxs"]
        results["labels_text"].append(processor.batch_decode(labels, skip_special_tokens=True)[0])
        results["train_idxs"].append(train_idxs)
        json.dump(results, open(save_file, "w"))

# ===========================
#           Setup
# ===========================
def load_model(model_id, model_revision=None, device="cuda"):
    model, processor = xpatch_helpers.load_hf_model(model_id, model_revision=model_revision, device=device)
    xpatch_helpers.remove_cache(model)
    text_model = xpatch_helpers.load_text_model(model_id, model)
    model_config = xpatch_helpers.get_model_config(text_model)
    model_config["layer_hook_names"] = [f"model.layers.{layer}" for layer in range(text_model.config.num_hidden_layers)]
    return model, processor, text_model, model_config

def get_best_layer(tasks, save_folder):
    save_files = []
    for task in tasks:
        save_file = get_file(save_folder, task, None)
        save_files.append(save_file)
    acc_info = xpatch_helpers.avg_task_accuracy(save_files, verbose=False)
    return acc_info["max_idx"]

def get_overall_results(tasks, save_folder, allow_lower_case):
    patch_L = get_best_layer(tasks, save_folder)
    results = []
    for task in tasks:
        save_file = get_file(save_folder, task, None)
        acc_info = xpatch_helpers.avg_task_accuracy([save_file], patch_L=patch_L, verbose=False, allow_lower_case=allow_lower_case)
        results.append(acc_info["accuracy"])
    df = pd.DataFrame(np.array(results).T)
    df.index = ["no_context", "xbase", f"xpatch (L={patch_L})"]
    df.index.name = "method"
    df.columns = list(tasks)
    df['avg'] = df.mean(axis=1)
    return df

def main(config):
    # Patch all layers by default
    # If patch_L == -1, find best layer by best val score
    # according to results files
    if config.get("patch_L") == -1:
        config["patch_L"] = get_best_layer(config["tasks"], config["save_folder"].replace(config["mode"], "val"))

    # Load model
    model_name = config["feats_model"] if config.get("save_feats", False) else config["patch_model"]
    model_info = OmegaConf.load(f"configs/models/{model_name}.yaml")
    model, processor, text_model, model_config = load_model(model_info["model_id"], model_revision=model_info.get("model_revision"))
    config = OmegaConf.merge(config, model_info)

    # Loop through tasks
    for t, task in enumerate(config["tasks"]):
        dataset_kwargs = copy.deepcopy(config["dataset_kwargs"])
        dataset_kwargs["annotation_file"] = f"{config['data_folder']}/{task}_{config['mode']}.json"
        # Setup instruction
        if config["spec"] == "instruction" and config.get("instructions"):
            instruction = config["instructions"][t]
            dataset_kwargs["num_examples"] = 0
        else:
            instruction = ""
        # Loop through seeds
        for seed in range(SEED_START, SEED_START + config["num_seeds"]):
            icl_pair_xpatch_dataset = xpatch_dataset.xPatchDataset(
                processor=processor, 
                seed=seed,
                **dataset_kwargs
            )
            if config.get("save_feats", False):
                save_feats(model, processor, text_model, model_config, config, task, seed, icl_pair_xpatch_dataset, instruction)
            else:
                evaluate(model, processor, text_model, model_config, config, task, seed, icl_pair_xpatch_dataset)
    if not config.get("save_feats", False):
        results = get_overall_results(config["tasks"], config["save_folder"], config.get("allow_lower_case", False))
        results.to_csv(f"{config['save_folder']}.csv")

if __name__ == "__main__":
    config = OmegaConf.load(sys.argv[1])
    if len(sys.argv) > 2:
        cli_overrides = OmegaConf.from_cli(sys.argv[2:])
        config = OmegaConf.merge(config, cli_overrides)
    OmegaConf.resolve(config)
    main(config)