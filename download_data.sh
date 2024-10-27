#!/bin/bash
data_root=data
repo=https://huggingface.co/datasets/g-luo/task_vectors_are_cross_modal

zip_name=annotations
wget -P ${data_root} -O ${data_root}/${zip_name}.zip ${repo}/resolve/main/data/${zip_name}.zip?download=true
unzip ${data_root}/${zip_name}.zip -d ${data_root}
rm -rf ${data_root}/${zip_name}.zip

zip_name=images
wget -P ${data_root} -O ${data_root}/${zip_name}.zip ${repo}/resolve/main/data/${zip_name}.zip?download=true
unzip ${data_root}/${zip_name}.zip -d ${data_root}
rm -rf ${data_root}/${zip_name}.zip