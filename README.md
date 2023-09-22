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
# Reference: https://github.com/orpatashnik/StyleCLIP

git clone https://github.com/orpatashnik/StyleCLIP.git
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=<CUDA_VERSION>
pip install ftfy regex tqdm gdown
pip install git+https://github.com/openai/CLIP.git

pip install tensorboard
cd StyleCLIP
mkdir pretrained_models && cd pretrained_models
wget https://www.dropbox.com/s/kzo52d9neybjxsb/model_ir_se50.pth?dl=0 -O model_ir_se50.pth
wget https://huggingface.co/akhaliq/jojogan-stylegan2-ffhq-config-f/resolve/main/stylegan2-ffhq-config-f.pt
(move the train_faces.pt and test_faces.pt from downloads into VSCode ./mappers folder... seems to be the easiest way)

cd ../mapper
python ./scripts/train.py --exp_dir ../results/mohawk_hairstyle --no_fine_mapper --description "mohawk hairstyle"
```
