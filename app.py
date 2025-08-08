from flask import Flask, redirect, render_template, request
import numpy as np
import os

import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

os.makedirs('static/tests', exist_ok=True)

verbose_name = {
    0: "Mild Demented",
    1: "Moderate Demented",
    2: "Non Demented",
    3: "Very Mild Demented"
}

class AlzheimerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerCNN, self).__init__()
        self.model = models.resnet18(pretrained=False)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

model = None

def load_pytorch_model():
    global model
    if model is None:
        model = AlzheimerCNN(num_classes=4)
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        model.eval()
    return model

def predict_label(img_path):
    # Load the model
    model = load_pytorch_model()
    
    transform = transforms.Compose([
        transforms.Resize((176, 176)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    
    return verbose_name[predicted.item()]


@app.route("/")
@app.route("/first")
def first():
    return render_template('first.html')
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('uname')
        password = request.form.get('pwd')
        
        # Add your authentication logic here
        if username and password:  # Add proper authentication
            # Successful login
            return redirect('index')
        else:
            # Failed login
            return render_template('login.html', error="Invalid credentials")
    
    return render_template('login.html')
 
    
@app.route("/index", methods=['GET', 'POST'])
def main():
    return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
    if request.method == 'POST':
        img = request.files['my_image']

        img_path = "static/tests/" + img.filename    
        img.save(img_path)

        predict_result = predict_label(img_path)

    return render_template("prediction.html", prediction = predict_result, img_path = img_path)

@app.route("/performance")
def performance():
    return render_template('performance.html')
    
@app.route("/chart")
def chart():
    return render_template('chart.html') 

if __name__ =='__main__':
    load_pytorch_model()
    print("PyTorch model loaded successfully!")
    app.run()