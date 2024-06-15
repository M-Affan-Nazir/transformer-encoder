from encoder_torch import Encoder
import torch
import numpy as np
from datetime import datetime

#Tokenize
from data_tokenizer import getData
max_len_size = 512


# Populate Input & Target
input = []
target = []
train_loader, vocabSize = getData(input, target)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
model = Encoder(
    vocab_size=max_len_size,
    max_len=max_len_size,
    d_k=6,
    d_model=8,
    n_heads=5,
    n_layers=3,
    n_classes=3,
    dropout_prob=0.1
)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, train_loader, epochs):
    train_losses = np.zeros(epochs) #array of zeros to store loss data
    model.train()
    for it in range(epochs):
        t0 = datetime.now()
        train_loss = 0
        n_train = 0
        i = 1
        for batch in train_loader:
            #zero the parameter gradients
            optimizer.zero_grad()

            #Forward Pass
            inputs, masks, labels = batch['input_ids'].to(device), batch['attention_mask'].to(device), batch['labels'].to(device)
            labels = torch.argmax(labels, dim=1)
            outputs = model(inputs, masks)
            loss = criterion(outputs, labels)


            if (torch.isnan(loss)):
                print("--------NAN FOUND------")
            print(f"i = {i}" + " loss: " + str(loss.item()))
            i+=1
            

            #Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*batch['input_ids'].size(0)
            n_train += inputs.size(0)

        train_loss = train_loss/n_train
        train_losses[it] = train_loss
        dt = datetime.now() - t0
        print(f"Epoch {it+1}/{epochs}, Train Loss: {train_loss}, | Duration: {dt}")   #:.4f = 4 figures



train(model, criterion, optimizer, train_loader, epochs=4)
