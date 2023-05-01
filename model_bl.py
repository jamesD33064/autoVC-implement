import torch
import torch.nn as nn

class D_VECTOR(nn.Module):
    """d vector speaker embedding."""
    def __init__(self, num_layers=3, dim_input=40, dim_cell=256, dim_emb=64):
        super(D_VECTOR, self).__init__()
        self.lstm = nn.LSTM(input_size=dim_input, hidden_size=dim_cell, 
                            num_layers=num_layers, batch_first=True)  
        self.embedding = nn.Linear(dim_cell, dim_emb)


    def forward(self, x):
        self.lstm.flatten_parameters()            
        lstm_out, _ = self.lstm(x)
        embeds = self.embedding(lstm_out[:,-1,:])
        norm = embeds.norm(p=2, dim=-1, keepdim=True) 
        embeds_normalized = embeds.div(norm)
        return embeds_normalized
    

# root - 
#     - /spmel 
#     - /wavs
#     - data_loader.py
#     - make_metadata.py
#     - model_vc.py 
#     - model_bl.py 
#     - preprocess.py
#     - solver_encoder.py

#     - train.ipynb
#     - MelGan 的 model

#     - interface.py (For MelGan)
#     - modules.py (For MelGan)