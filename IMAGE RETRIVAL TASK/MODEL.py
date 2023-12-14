import os 
import torch
from torchvision import models,transforms
from PIL import Image
from torch import nn
from annoy import AnnoyIndex



image_folder=r'DATASET_FOR_IMAGE_RETRIVAL'
images=os.listdir(image_folder)
weights=models.ResNet101_Weights.IMAGENET1K_V1
model=models.resnet101(weights=weights)
model.fc=nn.Identity()
model.eval()


transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()

])


annoy_index=AnnoyIndex(2048,'manhattan')


for i in range(len(images)):
   image=Image.open(os.path.join(image_folder,images[i]))
   input_tensor=transform(image).unsqueeze(0)
   if input_tensor.size()[1]==3:
    output_tensor=model(input_tensor)
    annoy_index.add_item(i,output_tensor[0])
   if i % 100==0:
     print(f'processed {i} images.')



annoy_index.build(12)
annoy_index.save('output.sample')

