import urllib.request
import torch
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

url = url = "https://upload.wikimedia.org/wikipedia/commons/4/45/A_small_cup_of_coffee.JPG"
fpath = "coffee.jpg"
urllib.request.urlretrieve(url, fpath)


img = Image("coffee.jpg")
plt.imshow(img)

transform  = transforms.Compose([
    transforms.resize(254),
    transforms.CenterCorp(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

img_tensor = transform(img)
batch = img_tensor.unsqueeze(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = models.alexnet(pretrained=True)
model.to(device)
model.eval()

y = model()
ymax, index = torch.max(y, 1)

url = "https://pytorch.tips/imagenet-labels"
fpath = "classes.txt"
urllib.request.urlretrieve(url, fpath)

with open('classes.txt') as f : 
    classes = [line.strip() for line in f.readlines()]

prob  = torch.nn.functional.softmax(y, 1)[0]

_, indices  = torch.sort(y,descending=True)

for idx in indices[0][:5]:
    print(classes[idx], prob[idx].item())
