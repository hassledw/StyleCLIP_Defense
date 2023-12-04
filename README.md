# StyleCLIP as a Defense Tool Against Deep Learning Facial Recognition Attacks

### Developers
* Carter Gilbert (cartergilbert@vt.edu)
* Daniel Hassler (hassledw@vt.edu)


## Introduction
Deep learning adversarial attacks such as FGSM (fast gradient sign method), a white-box approach aimed to misclassify images on image classifiers, can do serious damage to the trustworthiness of deep learning classification models. Our goal is to defend against such attacks using StyleCLIP as an image denoising/detection tool. 

![image](https://github.com/hassledw/StyleCLIP_Defense/assets/72363518/dbee92af-5e26-4ab0-8a87-e52c8892a6fd)\
**FGSM Attack on Macaw**

## Dataset
Since StyleCLIP is limited to face data, we are running all of our evaluations on the [CelebA_HQ](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) dataset. This dataset is specifically used for facial recognition; classifier details are in the CelebA_HQ github link.

## Setup

### Environment Setup
Here are some notes about our environment. We did all of our development on a linux platform with access to Tesla T4 GPUs, and our directory structure looks like this:
```
.
├── bin
├── encoder4editing
├── ninja-linux.zip
├── StyleCLIP
    ├── cog_predict.py
    ├── cog.yaml
    ├── criteria
    ├── defense.py
    ├── global_directions
    ├── global_torch
    ├── img
    ├── LICENSE
    ├── licenses
    ├── mapper
    ├── models
    ├── notebooks
    ├── optimization
    ├── pretrained_models
    ├── __pycache__
    ├── README.md
    ├── results
    └── utils.py
├── StyleCLIP_Defense
    ├── attackstorch.py
    ├── CelebA_HQ_facial_identity_dataset
    ├── CelebA_HQ-Labeled
    ├── celeb_classifier.py
    ├── celeb_metrics.ipynb
    ├── celeb_train.py
    ├── defense.py
    ├── FaceRecog_Adv.py
    ├── facial_identity_classification_transfer_learning_with_ResNet18.pth
    ├── FFHQ-Sentiment-Files
    ├── figures
    ├── README.md
    └── resources
```
### Clone Repo
```
# Clone our git repo
git clone git@github.com:hassledw/StyleCLIP_Defense.git
```

### Source Code Environment Download
```
https://drive.google.com/file/d/1PlnKXhi--SjtLdXxR8gWlaI5tF5qpDhK/view?usp=sharing
```
The ZIP folder in the source code environment download, titled "Source-Env-StyleCLIP_Defense.zip" contains our specific instance of StyleCLIP and encoder4editing, the specific envrionment instances we tweaked for our defense implementation. The only changes we made to StyleCLIP were path modifications. For our defense implementation in `defense.py`, please change the absolute path to the correct path on your machine.

## Instructions
Our main driver file used to classify, run attacks, and generate defense images is `FaceRecog_Adv.py`. To run our full-stack framework, run this on a CUDA enabled GPU:
```
python3 FaceRecog_Adv.py
```
If you would like to modify this result, navigate to the `main()` method. Here is an example of running four different FGSM attacks and running the defense sequentially:
```
    model = load_model()
    model.to(device)
    defense = Defense()

    labels_test = classify("test", model)

    # runs attack
    attacknames = ["FGSM05", "FGSM10", "FGSM25", "FGSM50"]
    attacks = [FGSM(model, eps=0.05), FGSM(model, eps=0.10), FGSM(model, eps=0.25), FGSM(model, eps=0.50)]
    for attackname, attack in zip(attacknames, attacks):
        attack_celeb(attack, labels_test, attackname)
        _ = classify(attackname, model)
        defend_celeb(attackname, defense)
        _ = classify(f"StyleCLIP-{attackname}", model)
```
The output of the attacked and generated defended images should be placed in a folder called `CelebA_HQ_facial_identity_dataset` with the respective name. An attacked foldername should be of naming convention `AttackXX` and the defense folder name should be of naming convention `StyleCLIP-AttackXX`.
