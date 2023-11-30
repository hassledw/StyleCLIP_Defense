import sys
sys.path.insert(1, '/home/grads/hassledw/StyleCLIP')
from defense import Defense

import celeb_classifier as cc
import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.models as models
from torchattacks.attack import Attack
import torchvision.transforms as transforms
from torchattacks import JSMA, PGD, FGSM, SPSA, RFGSM, Jitter, OnePixel, FAB, AutoAttack
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import art.attacks.evasion as evasion
from art.estimators.classification import PyTorchClassifier

import attackstorch
from transformers import AutoImageProcessor, ViTForImageClassification
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import pandas as pd
import numpy as np
import os
import time

path = "/home/grads/hassledw"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_model():
    '''
    Loads the model.
    '''
    save_path = f'{path}/StyleCLIP_Defense/facial_identity_classification_transfer_learning_with_ResNet18.pth'
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 50) 
    model.load_state_dict(torch.load(save_path))
    model.to(device)
    model.eval()
    return model

def attack_celeb(attack, labels_arr, attackfolder, rootsubdir="test"):
    '''
    Attacking all the images in the /CelebA_HQ.../{rootsubdir} folder.
    Stores attacked images into /CelebA_HQ.../attackfolder
    ''' 
    attackstorch.generate_attack(attack, labels_arr, attackfolder, rootsubdir=rootsubdir)

def defend_celeb(attackname, defense):
    '''
    Generate defended images from the /CelebA_HQ.../attack folder.
    Stores defended images into /CelebA_HQ.../defend
    '''
    df = pd.DataFrame(columns=['image', 'image_path'])
    curr_dir = "/home/grads/hassledw"
    rootdir = f'{curr_dir}/StyleCLIP_Defense/CelebA_HQ_facial_identity_dataset/{attackname}'
    savedir = f'{curr_dir}/StyleCLIP_Defense/CelebA_HQ_facial_identity_dataset/StyleCLIP-{attackname}'
    
    if os.path.exists(savedir):
        print("\nDefense has already been run...")
        return 0
    
    os.mkdir(savedir)
    count = 0
    for subdir, _, files in os.walk(rootdir):
        if len(files) == 0:
            continue
        subdir_arr = subdir.split("/")[-1]
        os.mkdir(f"{savedir}/{subdir_arr}")
        for i, file in enumerate(files):
            path = os.path.join(subdir, file)
            neutral = 'a face with noise'
            target = 'a face without noise' 
            # beta = 0.1 
            # alpha = 4.0
            # Accuracy on StyleCLIP-test 63.96%
            # beta = 0.13
            # alpha = 4.0
            # Accuracy on StyleCLIP-test 66.50%
            beta = 0.13
            alpha = 2.0
            # Accuracy on StyleCLIP-test 68.53%
            try:
                generated = defense.styleCLIP_def(neutral, target, alpha, beta, path)
                generated.save(f'{savedir}/{subdir_arr}/{file}')
                print(f"Saving... {count}")
            except:
                detected_image = Image.open(f"{path}")
                detected_image.save(f'{savedir}/{subdir_arr}/{file}')
                entry = [path.split("/")[-1], f'{savedir}/{subdir_arr}/{file}']
                df_entry = pd.DataFrame(entry, index=["image", "image_path"]).T
                df = pd.concat((df, df_entry))
                print(f"Adversarially Attacked Image Detected: {file}, saving incident...")
            count += 1
            df.to_csv(f"/home/grads/hassledw/StyleCLIP_Defense/CelebA_HQ-Labeled/StyleCLIP-{attackname}-detected.csv")

def classify(subdir, model, count=200):
    '''
    Passes a subdirectory of images into the model, outputs
    an array of labels.
    
    EXAMPLE:
    path = /CelebA_HQ.../attack/class
    path = /CelebA_HQ.../defend/class

    output = [array of classifications]
    '''
    data_dir = f'{path}/StyleCLIP_Defense/CelebA_HQ_facial_identity_dataset'
    # num_files=len(os.listdir(f'{data_dir}/{subdir}'))
    return cc.celebClassifier(data_dir, subdir, count, model)


def main():
    model = load_model()
    model.to(device)
    # defense = Defense()

    labels_test = classify("test", model)

    # runs attack
    # attacknames = ["FGSM90", "PGD1010", "PGD2010", "PGD2020", "PGD5050"]
    # attacks = [FGSM(model, eps=0.90), PGD(model, eps=0.10, alpha=0.10),
    #            PGD(model, eps=0.20, alpha=0.10), PGD(model, eps=0.20, alpha=0.20), 
    #            PGD(model, eps=0.50, alpha=0.50)]
    attacknames = ["test"]
    attacks = [FGSM(model, eps=0.75)]
    
    for attackname, attack in zip(attacknames, attacks):
        # _ = classify(attackname, model)
        # defend_celeb(attackname, defense)
        _ = classify(f"StyleCLIP-{attackname}", model)

if __name__ == "__main__":
    main()