# HPML_Project
Shuai Zhang (sz3714)
Melody Qu (mq2088)

## A description of the project
The goal of the project is to conduct a comparative analysis of optimization techniques applied to image segmentation models(such as FCN, Unet, Segnet, etc.). Through this analysis, find accurate and efficient image segmentation models that can be applied in medical image dataset.

## Code structure
```
├── FCN                     // fcn model
├── Segnet                  // segnet model
├── Unet                    // unet model
├── lgg-mri-segmentation    // dataset
├── result                  // result
```

## Commands to execute the code        
run this script in NYU HPC
you need modified the dataset dir in the model code
```
#!/bin/bash

#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:4
#SBATCH --job-name=torch
#SBATCH --output="%A\_%x.txt"
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1c24m128-v100-4

echo "GPU: $(nvidia-smi -q | grep 'Product Name')"

singularity exec --nv \
	    --overlay /scratch/sz3714/pytorch-example/my_pytorch.ext3:ro \
	    /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python fcn.py"
```

## Results
