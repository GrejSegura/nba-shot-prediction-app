from torch import nn, optim
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim

# LOAD DNN (PYTORCH) MODEL ARCHITECTURE
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(len(column_list), 200)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.3)
                
        self.fc2 = nn.Linear(200, 200)
        self.prelu = nn.PReLU(1)
        self.dout = nn.Dropout(0.3)
        
        self.fc6 = nn.Linear(200, 100)
        self.prelu = nn.PReLU(1)
        
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()
        
    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        dout = self.dout(h2)
        
        a6 = self.fc6(dout)
        h6 = self.prelu(a6)
        
        a7 = self.out(h6)
        
        y = self.out_act(a7)
        return y

# PREDICT THE OUTCOME
def predict_pytorch(data):
    dtype = torch.FloatTensor

    data_1 = torch.tensor(data.values).type(dtype)

    with torch.no_grad():
        y_pred = model.forward(data_1).cpu().numpy()
    
    return y_pred
