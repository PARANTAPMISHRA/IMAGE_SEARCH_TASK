import os
from PIL import Image,ImageDraw
import torch
from torchvision import models, transforms
import torch.nn as nn
from annoy import AnnoyIndex

images_folder=r'DATASET_FOR_IMAGE_RETRIVAL'
images=os.listdir(images_folder)
weights=models.ResNet101_Weights.IMAGENET1K_V1
model=models.resnet101(weights=weights)
model.fc=nn.Identity()
model.eval()


transforms=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()

])


annoy_index=AnnoyIndex(2048,'manhattan')


annoy_index.load(r'C:\Users\pc\OneDrive\Desktop\IMAGE RETRIVAL TASK\output.sample')

image_grid=Image.new('RGB',(1000,1000))

print(f'{"-" *40}')


for i in range (len(images)):
  image=Image.open(os.path.join(images_folder,images[i]))
  input_tensor=transforms(image).unsqueeze(0)

  if input_tensor.size()[1]==3:
    output_tensor=model(input_tensor)
    nns=annoy_index.get_nns_by_vector(output_tensor[0],24)
    image=image.resize((200,200))
    image_draw=ImageDraw.Draw(image)
    image_draw.rectangle([(0,0),(199,199)],outline='green',width=8)
    image_grid.paste(image,((0,0)))


    for j in range(24):
      search_image=Image.open(os.path.join(images_folder,images[nns[j]]))
      search_image=search_image.resize((200,200))
      image_grid.paste(search_image,(200*((j+1)%5),200*((j+1)//5)))
      image_grid.save(f'SIMILAR_IMAGES\SIMILAR_IMAGES_of{i}IMAGE.png')
  
  
  if i % 100==0:
    
    print(f'processed {i} th image')


print(f'{"-"*10}  EXECUTION END {"-"*10}')
  