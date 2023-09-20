# StyleCLIP as a Data Sanitization Tool for Deep Learning Adversarial Attacks on Image Data

### Developers
* Carter Gilbert (cartergilbert@vt.edu)
* Daniel Hassler (hassledw@vt.edu)


## Introduction
Deep learning adversarial attacks such as FGSM (fast gradient sign method), a white-box approach aimed to misclassify images on image classifiers, can do serious damage to the trustworthiness of deep learning classification models. Our goal is to defend against such attacks using StyleCLIP as an image denoising tool. 

![image](https://github.com/hassledw/StyleCLIP_Defense/assets/72363518/dbee92af-5e26-4ab0-8a87-e52c8892a6fd)\
**FGSM Attack on Macaw**

## Setup

### Environment Setup
We are doing all of our development on a linux platform with access to Tesla T4 GPUs.
```
# Clone our git repo
git clone git@github.com:hassledw/StyleCLIP_Defense.git
```

### StyleCLIP
```
git clone https://github.com/orpatashnik/StyleCLIP.git
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

pip install tensorboard
cd StyleCLIP
mkdir pretrained_models && cd pretrained_models
```
