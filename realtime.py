import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
import cv2

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

model.eval()

def transform_image(frame):
    # Convert BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(frame_rgb)
    # Apply the required transformations
    transformed = test_transforms(pil_image)
    return transformed.unsqueeze(0)



def predict_frame(frame, model):
    with torch.no_grad():
        logits = model(frame)
    prob = torch.softmax(logits, 1)
    pred = torch.argmax(prob, 1)
    return reverse[pred.item()]




def main():
    cap = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it if you have multiple cameras)
    cv2.namedWindow('Real-time Mask Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Real-time Mask Detection', 800, 600)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame to match the input size expected by the model
        frame = cv2.resize(frame, (128, 128))

        # Transform the frame
        transformed_frame = transform_image(frame)

        # Make prediction
        prediction_result = predict_frame(transformed_frame, model)

        # Display the result on the frame
        cv2.putText(frame, f'Prediction: {prediction_result}', (10, 60), cv2.FONT_HERSHEY_DUPLEX, 0.3, (0, 255, 0), 1)




        # Display the frame
        cv2.imshow('Real-time Mask Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()