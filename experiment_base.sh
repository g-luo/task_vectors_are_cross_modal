#!/bin/bash
# =========== CLI Args ========
feats_model=$1
patch_model=$2
spec=icl
src_modality=text
tgt_modality=image
patch_L=-1
config_name=configs/base.yaml
# =============================

cli_args="feats_model=${feats_model} patch_model=${patch_model} spec=${spec} src_modality=${src_modality} tgt_modality=${tgt_modality}"

# Feature Caching
python3 src/xpatch_evaluate.py ${config_name} save_feats=true mode=val ${cli_args}

# Validation
python3 src/xpatch_evaluate.py ${config_name} save_feats=false mode=val ${cli_args}

# Test
python3 src/xpatch_evaluate.py ${config_name} patch_L=${patch_L} save_feats=false mode=test ${cli_args}