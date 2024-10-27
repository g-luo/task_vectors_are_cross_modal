# Task Vectors are Cross-Modal
**Grace Luo, Trevor Darrell, Amir Bar**

This repository contains the data and code for the paper Task Vectors are Cross-Modal.

[[`Project Page`](https://task-vectors-are-cross-modal.github.io)][[`arXiv`](https://arxiv.org/abs/2410.22330)]

## Releases
- 🚀 2024/10/29: Initial codebase release

## Setup
This code was tested with Python 3.8. Please run the following to install the necessary packages and authenticate with HuggingFace, which is necessary to download some models.
```
conda env create -f environment.yml
conda activate xpatch
huggingface-cli login
```

## Data
All data can be found on our [HuggingFace page](https://huggingface.co/datasets/g-luo/task_vectors_are_cross_modal/tree/main). To download and set up the data, run the following script:
```
./download_data.sh
```

## Demo
In `demo.ipynb`, we walk through a few qualitative examples that illustrate the cross-modal nature of task vectors. Specifically, it demonstrates:

- **Cross-Modal Transfer.** Task vectors can be derived from text ICL examples, instructions, and image ICL examples and transferred to queries in another modality.
- **LLM to VLM Transfer.** Task vectors can also be patched from the base LLM to its corresponding fine-tuned VLM.
- **Vector Ensembling.** Instructions can improve the sample effiency of text ICL when averaging their corresponding task vectors.
- **Task Conflict.** When one task is provided in the prompt and a conflicting one is patched as a task vector, the model completes one of those two tasks.

## Experiments
To evaluate cross-modal patching on our six tasks, run the following scripts. Once the scripts are finished running, you can find per-task and average task accuracy statistics saved as csvs under `runs/experiments/<your_exp_folder>`.

- **Cross-Modal Transfer.** Run the script `./experiment_base.sh`. By default, the script is set to cross-modal patching from text ICL to image queries. The arguments to the script correspond to `feats_model` and `patch_model`, or the model to extract features from and the model to patch those features to.

```
# Default VLM Transfer
./experiment_base.sh idefics2 idefics2
./experiment_base.sh llava llava
./experiment_base.sh mantis mantis

# LLM to VLM Transfer
./experiment_base.sh mistral idefics2
./experiment_base.sh vicuna llava
```

- **Vector Ensembling.** Run the script `./experiment_ensemble.sh`. The script evaluates the scaling properties of task vectors derived from text ICL examples, ranging from 5 to 30 examples. It also demonstrates the performance when averaging with instruction-based task vectors, where the instructions are defined in `configs/base.yaml`. The argument to the script corresponds to `ensemble`, i.e., whether to run only exemplar-based vectors or ensemble them with instruction-based vectors.

```
./experiment_ensemble.sh false
./experiment_ensemble.sh true
```

## Citing
```
@article{luo2024tvacm,
  title={Task Vectors are Cross-Modal}, 
  author={Grace Luo and Trevor Darrell and Amir Bar},
  journal={arXiv preprint arXiv:2410.22330}
  year={2024}
}
```