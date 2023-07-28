from PIL import Image
from torchvision import transforms

img_path="dataset/1/train/ants/0013035.jpg"
img=Image.open(img_path)

tensor1=transforms.ToTensor()
tensor2=tensor1(img)
print(tensor2)

