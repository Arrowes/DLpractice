import torchvision
from torch.utils.data import DataLoader
from nn_CIFAR import *

train_data = torchvision.datasets.CIFAR10("data", True, torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10("data", False, torchvision.transforms.ToTensor(), download=True)
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集长度：{}".format(train_data_size))
print("测试集长度：{}".format(test_data_size))

train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

tudui=Tudui()

loss_fn=nn.CrossEntropyLoss()

optim=torch.optim.SGD(tudui.parameters(),0.001)

trainstep=0
teststep=0
epoch=10

for i in range(epoch):
    print("——————————————————for {} epoches ——————————————————".format(i+1))
    for data in train_dataloader:
        imgs,targets=data
        outputs=tudui(imgs)
        loss=loss_fn(outputs,targets)
        optim.zero_grad()
        loss.backward()
        optim.step()

        trainstep=trainstep+1
        print("——————————————————train times:{},loss:{}——————————————————".format(trainstep,loss.item()))