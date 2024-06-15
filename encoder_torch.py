import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.utils.data import dataset
import numpy as np
import matplotlib.pyplot as plt

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_k,d_model,n_heads):
        super().__init__()

        self.d_k= d_k
        self.n_heads = n_heads

        # d_v = d_k
        self.keyCalc=nn.Linear(d_model,d_k * n_heads)  #dmodel inputs & d_k*n_heads output. Matrix of size "d_model, d_k * n_heads" (is transposed later)
        self.queryCalc = nn.Linear(d_model, d_k * n_heads)
        self.valueCalc = nn.Linear(d_model,d_k*n_heads)

        #Final Linear layer (To reverse dimensionality of output back to orignal dimension)
        self.fc = nn.Linear(d_k * n_heads , d_model)


    def forward(self, q, k, v, mask=None):
    

        q = self.queryCalc(q) #q,k,v in attention equation
        k = self.keyCalc(k)
        v = self.valueCalc(v)

        N = q.shape[0] #Batch size
        T= q.shape[1]  #Sequence Length


        #Change shape:
        # (N, T, n_head*dk) --> (N, T, n_head, d_k) --> (N, n_head, T, d_k)
        q = q.view(N,T,self.n_heads,self.d_k).transpose(1,2)
        k = k.view(N,T,self.n_heads,self.d_k).transpose(1,2)
        v = v.view(N,T,self.n_heads,self.d_k).transpose(1,2)

        
        #Computing Attention score 1st Half (Q*K^T) / sqrt(d_k)
        attn_score = q @ k.transpose(-2,-1) / math.sqrt(self.d_k)
        
        #Correcting Mask Matrix to 4D and adding -inf in place of 0:
        if mask is not None:  
            attn_score = attn_score.masked_fill(mask[:,None,None,:] == 0, float('-inf'))
        
        attn_weights = f.softmax(attn_score, dim=-1)



        #Computing Attention score 2nd attn_weight * Value-Vector
        A = attn_weights @ v
    
        #Reshape
        A = A.transpose(1,2)  # (N, n_heads, T, d_k) -> (N,T,n_heads,d_k)
        A = A.contiguous().view(N, T, self.d_k * self.n_heads)   #(N,T,n_heads,d_k) -> (N,T,n_heads*d_k)
        
        #Projection
        return self.fc(A)  #fc = forward linear layer.

class TransformerBlock(nn.Module):
    
    #Architecture for adding Normalization & non-linearity to the Attention-Matrix (from MHA Architectural Network)

    def __init__(self, d_k, d_model, n_heads, dropout_prob=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model) #Normalization
        self.ln2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttentionLayer(d_k, d_model, n_heads)
        self.ann = nn.Sequential(
            nn.Linear(d_model,d_model*4),   #Linear Layer has d_model inputs & d_model*4 outputs. Mathematically, matrix of size [d_model, d_model*4] (transposed later) !
            nn.GELU(),  #Activation function; like RELU
            nn.Linear(d_model*4, d_model),
            nn.Dropout(dropout_prob)
        )
        self.dropout = nn.Dropout(p=dropout_prob)
    

    def forward(self, x, mask=None):
        x = self.ln1(x + self.mha(x,x,x, mask))
        #q=x, k=x, v=x; because q,k,v calculated vectors are derived from same input!
        x = self.ln2(x + self.ann(x)) #Adding Non Linearity
        x = self.dropout(x)
        return x
    
class PositionalEncoding(nn.Module):
    
    #Architecture for adding positional information to the embnedding vector & Dropout to force Generalization and elevation of consoiusness of Model

    def __init__(self, d_model, max_len=2048, dropout_prob=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout_prob)
        position = torch.arange(max_len).unsqueeze(1) #A list of numbers from 0 -> Max Length (represent 'pos' variable in formula). unsqueeze(1) = adds another dimension. Now size = (max-Length x 1)
        exp_term = torch.arange(0, d_model, 2) # 1D array 0 -> d_model; jumping by 2 [0,2,4,...,dmodel]. Represents '2i' term in exponent of 1000
        div_term = torch.exp(exp_term * (-math.log(10000.0) / d_model))

        pe = torch.zeros(1,max_len,d_model) # A 3d Matrix of 0's
        pe[0,:,0::2] = torch.sin(position * div_term)
        pe[0,:,1::2] = torch.cos(position * div_term)
        self.register_buffer("pe",pe) #save the variable

    
    def forward(self, x):
        #x shape = (N,T,Dmodel)
        x = x + self.pe[:, :x.size(1), :]   
    
        return self.dropout(x) 

class Encoder(nn.Module):

    def __init__(self, vocab_size, max_len, d_k, d_model, n_heads, n_layers, n_classes, dropout_prob):
        
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model) #dictionary vector -> embedding matrix (each integer token to a vector)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout_prob)
        
        transformer_blocks = [
            TransformerBlock(d_k,d_model,n_heads, dropout_prob) for _ in range(n_layers)
        ]
        
        #Defining Actual Layers
        self.transformer_blocks = nn.Sequential(*transformer_blocks) # '*' to unpack. Each item passed seperately. Each Transformer Block in a Sequence (Not complex)
        self.ln = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, n_classes) #Input is d_model; output is n_classes. Means, it will be used for a classification task.
        


    def forward(self, x, mask=None): #x = input
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for block in self.transformer_blocks:
            x = block(x,mask) #Passing individually because of 'mask' variable
        
        #OPTIONAL : x has shape NxTxD. We change to NxD
        x = x[:,0,:] #(for Text classification we only keep 1st Vector)
        
        x = self.ln(x)
        x = self.fc(x)

        return x
