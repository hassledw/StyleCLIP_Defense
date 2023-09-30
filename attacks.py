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
'''
Attack class.

Authors: Carter Gilbert, Daniel Hassler
Version: 09/29/2023

'''
class Attack():
    '''
    Attack class is meant to initialize an attacker instance as 
    the heart of the adversarial framework.
    '''
    def __init__(self, imdim=224, default_classifier=True, classifier=None, criterion=None):
        '''
        Constructor function

        imdim (int): image dimensions (default 224)
        default_classifier (bool): a basic classifier for running the image. (default resnet18)
        classifier (Classifier): an instance of your own classifier non-default (default None)
        criterion (*Loss): specify your own loss function non-default (default None)  
        '''
        self.imdim = imdim

        if default_classifier:
            self.model_resnet18 = models.resnet18(pretrained=True)  
            
            self.criterion = nn.CrossEntropyLoss()

            self.classifier = PyTorchClassifier(
                model=self.model_resnet18,
                loss=self.criterion,
                input_shape=(3, self.imdim, self.imdim),
                nb_classes=1000,
                device_type='gpu'
            
            )
        else:
            self.criterion = criterion
            self.classifier = classifier

        # Create the ART classifier
        self.preprocess = transforms.Compose([
                transforms.Resize(self.imdim),
                transforms.CenterCrop(self.imdim),
                transforms.ToTensor()
            ])


    def save_image(self, name, input):
        '''
        Saves the image given its input

        input: image
        '''
        norm = plt.Normalize(vmin=input.min(), vmax=input.max())
        image = norm(input)
        plt.imsave(name + ".jpg", image)
        
    def softmax_activation(self, inputs):
        '''
        Preform softmax activation on input layer.
        Used in accuracy calculation to normalize data.

        inputs: an array of input.
        ''' 
        inputs = inputs.tolist()
        exp_values = np.exp(inputs - np.max(inputs)) 
        
        # Normalize 
        probabilities = exp_values / np.sum(exp_values)
        return probabilities 

    def compare_images(imageA, imageB):
        '''
        Comparing two images' structual similarity.
        '''
        return 1 - ssim(imageA, imageB, multichannel=True)

    def initial(self, file_path):
        '''
        Creates an input image from file_path, 
        '''
        input_image = Image.open(file_path) 
        input_tensor = self.preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).numpy().astype(np.float32)
        input = input_batch[0].transpose((1,2,0))
        preds = self.classifier.predict(input_batch)
        accuracy = np.max(self.softmax_activation(preds), axis=1)
        accuracy = round(accuracy[0], 2)

        return input_batch

    def FGSM(self, file_path):
        '''
        Runs an FGSM (Fast Gradient Sign Method) attack on image.

        file_path: path to image file.
        '''
        input_batch = self.initial(file_path)

        fgsm_attack = FastGradientMethod(estimator = self.classifier, eps=0.05) 
        
        x_test_adv = fgsm_attack.generate(x=input_batch)
        predictions = self.classifier.predict(x_test_adv)
        # print(np.argmax(predictions, axis=1))
        accuracy = round(np.max(self.softmax_activation(predictions), axis=1)[0]*100,2)
        # print("Accuracy on adversarial test examples: {}%".format(accuracy))
        
        input_fgsm = x_test_adv[0].transpose((1,2,0))
        
        self.save_image("FGSM", input_fgsm)
        
    def PGD(self, file_path):
        '''
        Runs a PGD (Projected Gradient Descent) attack on image.

        file_path: path to image file.
        '''
        input_batch = self.initial(file_path)
        pass

    def SPSA(self, file_path):
        '''
        Runs an SPSA attack on image.

        file_path: path to image file.
        '''
        pass

    def JSMA(self, file_path):
        '''
        Runs a JSMA (Jacobian Saliency Map Attack) attack on image.
        '''
        pass


if __name__ == "__main__":
    attack = Attack()
    attack.FGSM("./FFHQ512/00000.png")
