import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image

label_encoder = {'mask': 0, 'mask_incorrect': 1, 'no_mask': 2}
reverse  = {v:k for k,v in label_encoder.items()}

test_transforms = T.Compose([
    T.Resize(size=(128,128)),
    T.ToTensor()
])

class CNN(nn.Module):
    def __init__(self,input=3,output=len(label_encoder)):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,4,1),
            nn.ReLU(),
            nn.Conv2d(64,32,4,1),
            nn.ReLU(),
            nn.MaxPool2d(2,1)
        )
        self.fc = nn.Sequential(
            nn.Linear(468512,32),
            nn.ReLU(),
            nn.Linear(32,output)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = torch.flatten(x,start_dim=1)
        x = self.fc(x)
        return x
    
model = CNN()

model.load_state_dict(torch.load('pro3.pth',map_location=torch.device('cpu')))


def prediction(text):
    pil_image = Image.open(text)
    transformed = test_transforms(pil_image)
    model.eval()
    with torch.inference_mode():
        logits = model(transformed.unsqueeze(0))
    prob = torch.softmax(logits,1)
    pred = torch.argmax(prob,1)
    return reverse[pred.item()]

if __name__ == '__main__':
    text = input('enter the location of the image')
    output = prediction(text)
    print(output)