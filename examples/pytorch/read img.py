from torchvision import transforms
from PIL import Image

img = Image.open(r'E:\desktop\pytorch\data\red.jpg')


tensor_tool=transforms.ToTensor()
img=tensor_result=tensor_tool(img)
print(img.shape)

