举派尤特人notebook

class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.lable_dir=lable_dir


        self.path=os.path.join(self.root_dir,self.lable_dir)
        self.img_path=os.listdir(self.path)

    def __getitem__(self, idx): #取每一个图片
        img_name=self.img_path[idx] #引用图片名
        img_item_path=os.path.join(self.root_dir,self.lable_dir,img_name) #相对路径
        img=Image.open(img_item_path)
        label=self.lable_dir
        return img,label

    def __len__(self.img_path):
    return len(self.img_path)

root_dir="dataset/1/train"
ants_label_dir="ants"
bees_label_dir="bees"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)

train_dataset=ants_dataset+bees_dataset