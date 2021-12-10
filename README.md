# Plenoxels: Radiance Fields without Neural Networks
Alex Yu\*, Sara Fridovich-Keil\*, Matthew Tancik, Qinhong Chen, Benjamin Recht, Angjoo Kanazawa

UC Berkeley

Website and video: <https://alexyu.net/plenoxels>

arXiv: <https://arxiv.org/abs/2112.05131>

**Note:** This JAX implementation is intended to be high-level and user-serviceable, but is much slower than the CUDA implementation <https://github.com/sxyu/svox2>, and there is not perfect feature alignment between the two versions (this JAX version can likely be sped up significantly, and we may push performance improvements and extra features in the future). Currently, this version only supports bounded scenes and trains using SGD without regularization.



Citation:
```
@misc{yu2021plenoxels,
      title={Plenoxels: Radiance Fields without Neural Networks}, 
      author={{Alex Yu and Sara Fridovich-Keil} and Matthew Tancik and Qinhong Chen and Benjamin Recht and Angjoo Kanazawa},
      year={2021},
      eprint={2112.05131},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
## Setup

We recommend setup with a conda environment, using the packages provided in `requirements.txt`.

## Downloading data

Currently, this implementation only supports NeRF-Blender, which is available at:

<https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1>

## Voxel Optimization (aka Training)

The training file is `plenoptimize.py`; its flags specify many options to control the optimization (scene, resolution, training duration, when to prune and subdivide voxels, where the training data is, where to save rendered images and model checkpoints, etc.). You can also set the frequency of evaluation, which will compute the validation PSNR and render validation images (comparing the reconstruction to the ground truth).
