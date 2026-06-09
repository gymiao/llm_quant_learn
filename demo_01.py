import torch
import torch.nn as nn

import numpy as np

t = np.arange(0, 100, 0.1)
prices = np.sin(t) + 0.1*np.random.randn(len(t))

seq_len = 20
X_train, y_train = [], []

for i in range(len(prices) - seq_len):
    X_train.append(prices[i:i+seq_len])
    y_train.append(prices[i+seq_len])
    
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, seq_len=20, d_model=32, nhead=4, num_layers=2):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_fc = nn.Linear(1, d_model)
        
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=128
            )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers)
        
        self.output_fc = nn.Linear(d_model, 1)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x_embed = self.input_fc(x) + self.pos_embedding
        
        x_embed = x_embed.permute(1, 0, 2)
        
        encoded = self.transformer_encoder(x_embed)
        
        last_feat = encoded[-1]
        
        out = self.output_fc(last_feat)
        
        return out.squeeze(-1)
    
model = TimeSeriesTransformer(
    seq_len=seq_len,
    d_model=32,
    nhead=4,
    num_layers=2)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    
    preds = model(X_train)
    loss = loss_fn(preds, y_train.squeeze(-1))
    
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        
model.eval()

last_seq = torch.tensor(
    prices[-seq_len:], dtype=torch.float32).unsqueeze(0)

pred_next = model(last_seq).item()
print(f'Predicted next price: {pred_next:.4f}')