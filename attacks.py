import time
import numpy as np 
import torch.nn as nn 
from datetime import datetime

from PIL import Image
from torchvision import transforms
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt 

from art.estimators.classification import PyTorchClassifier
from skimage.metrics import structural_similarity as ssim 

import warnings
warnings.filterwarnings('ignore') 

from art.attacks.evasion import ProjectedGradientDescent
from art.attacks.evasion import FastGradientMethod
import requests 

def save_input(name, input):
    norm = plt.Normalize(vmin=input.min(), vmax=input.max())
    image = norm(input)
    plt.imsave(name + ".jpg", image)
    
def softmax_activation(inputs): 
    inputs = inputs.tolist()
    exp_values = np.exp(inputs - np.max(inputs)) 
    
    # Normalize 
    probabilities = exp_values / np.sum(exp_values)
    return probabilities 

def compare_images(imageA, imageB):
    return 1 - ssim(imageA, imageB, multichannel=True) 

preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()
]) 

# load the model
model_resnet18 = models.resnet18(pretrained=True)  

criterion = nn.CrossEntropyLoss()

# Create the ART classifier

classifier = PyTorchClassifier(
    model=model_resnet18,
    loss=criterion,
    input_shape=(3, 224, 224),
    nb_classes=1000,
    device_type='gpu'
)

def FGSM(file_path):
    input_image = Image.open(file_path) 
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).numpy().astype(np.float32)
    input = input_batch[0].transpose((1,2,0))
    preds = classifier.predict(input_batch)
    accuracy = np.max(softmax_activation(preds), axis=1)
    accuracy = round(accuracy[0], 2)

    fgsm_attack = FastGradientMethod(estimator = classifier, eps=0.05) 
    
    x_test_adv = fgsm_attack.generate(x=input_batch)
    predictions = classifier.predict(x_test_adv)
    # print(np.argmax(predictions, axis=1))
    accuracy = round(np.max(softmax_activation(predictions), axis=1)[0]*100,2)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy))
    
    input = x_test_adv[0].transpose((1,2,0))
    
    save_input("FGSM", input)
    
def PGD():
    input_image = Image.open(r"./FFHQ512/00002.png") 
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0).numpy().astype(np.float32)

    input = input_batch[0].transpose((1,2,0))

    preds = classifier.predict(input_batch)

    accuracy = np.max(softmax_activation(preds), axis=1)
    accuracy = round(accuracy[0], 2)