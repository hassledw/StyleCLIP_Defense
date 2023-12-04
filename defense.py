import os
path = '/home/grads/hassledw'
os.chdir(path)
CODE_DIR = 'encoder4editing'
os.chdir(f'./{CODE_DIR}')
from argparse import Namespace
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
sys.path.append(".")
sys.path.append("..")
from encoder4editing.utils.common import tensor2im
from encoder4editing.models.psp import pSp
os.chdir(f'{path}/StyleCLIP/global_torch/')
import clip
from global_torch.manipulate import Manipulator
from StyleCLIP.global_torch.StyleCLIP import GetDt, GetBoundary
import dlib 
from encoder4editing.utils.alignment import align_face
'''
StyleCLIP Defense framework. Utilizes OOB structure to call functions more efficiently.
'''
# ! pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html 

# !git clone https://github.com/omertov/encoder4editing.git $CODE_DIR
# !wget https://github.com/ninja-build/ninja/releases/download/v1.8.2/ninja-linux.zip
# !sudo unzip ninja-linux.zip -d /usr/local/bin/
# !sudo update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force
# ! pip install ftfy regex tqdm 
# !pip install git+https://github.com/openai/CLIP.git 
# ! git clone https://github.com/orpatashnik/StyleCLIP
# helper methods

class Defense():
    def __init__(self):
        '''
        Initialization code for our defense.
        '''
        self.experiment_type = 'ffhq_encode'
        self.M, self.EXPERIMENT_ARGS, self.model, self.fs3, self.resize_dims, self.net = self.preprocess()

    def display_alongside_source_image(self, result_image, source_image):
        '''
        Helper function.
        '''
        res = np.concatenate([np.array(source_image.resize(self.resize_dims)),
                            np.array(result_image.resize(self.resize_dims))], axis=1)
        return Image.fromarray(res)

    def run_on_batch(self, inputs):
        '''
        Helper function.
        '''
        images, latents = self.net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
        if self.experiment_type == 'cars_encode':
            images = images[:, :, 32:224, :]
        return images, latents

    def run_alignment(self, image_path):
        '''
        Helper function.
        '''
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        aligned_image = align_face(filepath=image_path, predictor=predictor) 
        print("Aligned image has shape: {}".format(aligned_image.size))
        return aligned_image 

    def preprocess(self):
        '''
        Model preprocessing.
        '''
        dataset_name='ffhq' 

        if not os.path.isfile('./model/'+dataset_name+'.pkl'):
                url='https://nvlabs-fi-cdn.nvidia.com/stylegan2/networks/'
                name='stylegan2-'+dataset_name+'-config-f.pkl'
                os.system('wget ' +url+name + '  -P  ./model/')
                os.system('mv ./model/'+name+' ./model/'+dataset_name+'.pkl')


        # input prepare data
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device,jit=False)

        network_pkl='./model/'+dataset_name+'.pkl'
        device = torch.device('cuda')
        M=Manipulator()
        M.device=device
        G=M.LoadModel(network_pkl,device)
        M.G=G
        M.SetGParameters()
        num_img=100_000
        M.GenerateS(num_img=num_img)
        M.GetCodeMS()
        np.set_printoptions(suppress=True)

        file_path='./npy/'+dataset_name+'/'
        fs3=np.load(file_path+'fs3.npy')
        #@title e4e setup
        #@ e4e setup 
        if not os.path.isfile(f"{path}/encoder4editing/e4e_ffhq_encode.pt"):
            from gdown import download as drive_download
            drive_download("https://drive.google.com/uc?id=1O8OLrVNOItOJoNGMyQ8G8YRTeTYEfs0P", f"{path}/encoder4editing/e4e_ffhq_encode.pt", quiet=False)

        os.chdir(f'{path}/encoder4editing')

        EXPERIMENT_ARGS = {
                "model_path": "e4e_ffhq_encode.pt"
            }
        EXPERIMENT_ARGS['transform'] = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        resize_dims = (256, 256)

        model_path = EXPERIMENT_ARGS['model_path']
        ckpt = torch.load(model_path, map_location='cuda')
        opts = ckpt['opts']
        # pprint.pprint(opts)  # Display full options used
        # update the training options
        opts['checkpoint_path'] = model_path
        opts= Namespace(**opts)
        net = pSp(opts)
        net.eval()
        net.cuda()
        print('Model successfully loaded!')
        return (M, EXPERIMENT_ARGS, model, fs3, resize_dims, net)

    def styleCLIP_def(self, neutral, target, alpha, beta, image_path):
        '''
        Main StyleCLIP implementation code.
        '''
        original_image = Image.open(image_path)
        original_image = original_image.convert("RGB")
        if self.experiment_type == "ffhq_encode" and 'shape_predictor_68_face_landmarks.dat' not in os.listdir():
            os.system("wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            os.system("bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2")

        if self.experiment_type == "ffhq_encode":
            input_image = self.run_alignment(image_path)
        else:
            input_image = original_image

        input_image.resize(self.resize_dims)
        img_transforms = self.EXPERIMENT_ARGS['transform']
        transformed_image = img_transforms(input_image)

        with torch.no_grad():
            images, latents = self.run_on_batch(transformed_image.unsqueeze(0))
            result_image, latent = images[0], latents[0]
        torch.save(latents, 'latents.pt')

        self.display_alongside_source_image(tensor2im(result_image), input_image)

        img_index =  1

        mode='real image' 

        if mode == 'real image':
            img_index = 0
            latents=torch.load(f'{path}/encoder4editing/latents.pt')
            dlatents_loaded=self.M.G.synthesis.W2S(latents)
            img_indexs=[img_index]
            dlatents_loaded=self.M.S2List(dlatents_loaded)
            dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
        elif mode == 'generated image':
            img_indexs=[img_index]
            dlatents_loaded=self.M.S2List(dlatents_loaded)
            dlatent_tmp=[tmp[img_indexs] for tmp in self.M.dlatents]
            
        self.M.num_images=len(img_indexs)
        self.M.alpha=[0]
        self.M.manipulate_layers=[0]
        codes,out=self.M.EditOneC(0,dlatent_tmp) 
        original=Image.fromarray(out[0,0]).resize((512,512))
        self.M.manipulate_layers=None
        original


        classnames=[target,neutral]
        dt=GetDt(classnames, self.model)

        self.M.alpha=[alpha]
        boundary_tmp2,c=GetBoundary(self.fs3,dt,self.M,threshold=beta)
        codes=self.M.MSCode(dlatent_tmp,boundary_tmp2)
        out=self.M.GenerateImg(codes)
        generated=Image.fromarray(out[0,0])#.resize((512,512))
        # print("Generated image.")
        # generated.save(f"{path}/StyleCLIP_Defense/images/generated.png")
        return generated
        

# if __name__ == '__main__':

#     defense = Defense()
#     neutral = 'a face with skin'
#     target = 'a face with clear skin' 
#     beta = 0.1 
#     alpha = 4.0

#     image_path = f"{path}/StyleCLIP_Defense/images/FGSM.png"
#     for i in range(5):
#         defense.styleCLIP_def(neutral, target, alpha, beta, image_path)
