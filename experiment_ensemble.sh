#!/bin/bash
# ======== CLI Args ===========
ensemble=$1
feats_model=idefics2
patch_model=${feats_model}
src_modality=text
tgt_modality=image
config_name=configs/base.yaml
# =============================

cli_args="feats_model=${feats_model} patch_model=${patch_model} src_modality=${src_modality} tgt_modality=${tgt_modality}"

# Instruction Feature Caching
instruction_args="num_seeds=1 exp=instruction_${src_modality}-${tgt_modality}_scaling"
python3 src/xpatch_evaluate.py ${config_name} spec=instruction save_feats=true mode=val ${cli_args} ${instruction_args}

# n=0
python3 src/xpatch_evaluate.py ${config_name} spec=instruction save_feats=false mode=test allow_lower_case=true ${cli_args} ${instruction_args}

# ICL Feature Caching
feats_all_args="save_feats_all=true exp=icl_${src_modality}-${tgt_modality}_scaling" 
python3 src/xpatch_evaluate.py ${config_name} spec=icl save_feats=true mode=val ${cli_args} ${feats_all_args}

# n=5 to 30
for r in {1..6}
do
    # Each row r in feats is composed of 5 ICL examples
    n=$((r * 5))
    save_folder=runs/experiments/icl_${src_modality}-${tgt_modality}_scaling/feats-${feats_model}_patch-${patch_model}/test/n-${n}_ensemble-${ensemble}
    scaling_args="save_feats_subset=${r} save_folder=${save_folder}"
    python3 src/xpatch_evaluate.py ${config_name} spec=icl ensemble=${ensemble} save_feats=false mode=test allow_lower_case=true ${cli_args} ${feats_all_args} ${scaling_args}
done