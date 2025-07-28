<div align="center">
  
# Implicit Neural Representations for Brain Aging Trajectory Classification

</div>

<p align="center">
<img src="fig/workflow.png?raw=true" width="800">
</p>

Implementation of our paper "_Semi-Disentangled Spatiotemporal Implicit Neural Representations of Longitudinal Neuroimaging Data for Trajectory Classification_," accepted at the [LMID workshop](https://ldtm-miccai.github.io/lmid2025/) at MICCAI 2025.

[[Paper - TBA]()] [[Web Page - TBA]()] [[Corresponding Author Email](mailto:agampreet.aulakh@ucalgary.ca?subject=INR%20trajectory%20classification)]

---
### Abstract
The human brain undergoes dynamic, potentially pathology-driven, structural changes throughout a lifespan. Longitudinal Magnetic Resonance Imaging (MRI) and other neuroimaging data are valuable for characterizing trajectories of change associated with typical and atypical aging. However, the analysis of such data is highly challenging given their discrete nature with different spatial and temporal image sampling patterns within individuals and across populations. This leads to computational problems for most traditional deep learning methods that cannot represent the underlying continuous biological process. To address these limitations, we present a new, fully data-driven method for representing aging trajectories across the entire brain by modelling subject-specific longitudinal T1-weighted MRI data as continuous functions using implicit neural representations (INRs).

The main contributions of this work are as follows:
1. We propose an INR architecture with semi-disentangled spatial and temporal parameters to model subject-level aging trajectories across the entire brain from longitudinal MRI data. We show that these INRs can be flexibly adapted to irregular temporal sampling.

2. We propose a method for directly classifying the brain aging trajectories encoded by INRs.

3. We develop a brain aging simulation framework to generate realistic, longitudinal 3D imaging data of healthy and Alzheimer's-like trajectories. We use this simulation to validate our methods.

---
### Setup
This code was developed and tested using Python 3.12.7 and CUDA 12.4. To replicate the conda environment used in this work, run the following after cloning our repo:
```
conda env create -f environment.yml
```
### Data
Our brain aging trajectory simulation was tested on longitudinal data produced by our [3D Conditional Diffusion Model](https://github.com/wilmsm/lightweightbraindiff).
To reproduce the data used in all experiments:
```
TBA
```

To create data splits for subjects with regular and irregular sampling:
```
preprocess.py --data-dir <path/to/data-dir>
```
### Experiments
We provide shell scripts with all necessary hyperparameters used for training INRs and classifying their brain aging trajectories:
```bash
├── scripts
│   ├── inr_init.sh
│   ├── inr_adapt.sh
│   ├── classify_inr.sh
│   └── classify_baseline.sh
```
---
### Citation
Please cite our paper if you find this work useful:
```
{TBA}
```

### Acknowledgements
We gratefully acknowledge this code is based on and incorporates other publically available projects, including:

- [Neonatal Development INR](https://github.com/FlorentinBieder/Neonatal-Development-INR) from Bieder et al., 2024.
- [inr2vec](https://github.com/CVLAB-Unibo/inr2vec) from De Luigi, Cardace, et al., 2023.
- [WIRE](https://github.com/vishwa91/wire) from Saragadam et al., 2023.
- [Implicit Segmentation](https://github.com/NILOIDE/Implicit_segmentation) from Stolt-Ansó et al., 2023.