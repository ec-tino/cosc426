import torch
from torch.utils.data import DataLoader
from LM import LM_Evaluator

class Masked_LM(torch.nn.Module):
    def __init__(self, vocabSize, nEmbed, nHidden, nLayers):
        super(Masked_LM, self).__init__()
        self.vocabSize = vocabSize
        self.nHidden = nHidden
        self.nLayers = nLayers
        self.directions = 2 #for bidirectional 
        self.embed = torch.nn.Embedding(vocabSize, nEmbed)
        self.lstm = torch.nn.LSTM(nEmbed, nHidden, nLayers, batch_first=True, bidirectional=True)
        self.decoder = torch.nn.Linear(nHidden * self.directions, vocabSize)  # the input will take 2 different directions of output from the hidden layer 

    def forward(self, X, hidden, cell):
        embedded = self.embed(X.long())
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        y_pred = self.decoder(output)
        return y_pred, hidden, cell

    def init_hidden(self, batchSize):
        hidden = cell =  torch.zeros((self.nLayers * self.directions, batchSize, self.nHidden), dtype=torch.float)

        return hidden, cell

    def loss(self, y_pred, y_target):
        loss_fn = torch.nn.CrossEntropyLoss() ## takes logits not probs
        return loss_fn(y_pred, y_target)


class MaskedLM_Trainer():
    def __init__(self, num_epochs, lr, batch_size, device):
        self.num_epochs = num_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device

    def train(self, model, train_data, val_data):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)

        train_loader = DataLoader(train_data, batch_size = self.batch_size, shuffle=True) 

        evaluator = LM_Evaluator(val_data, self.device)

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            num_batches = 0

            for batch in train_loader:
                X,y_target = batch
 
                num_batches+=1

                hidden, cell = model.init_hidden(X.size(0))
                hidden, cell = hidden.to(self.device), cell.to(self.device) #move tensors for hidden and cell to device

                X,y_target = X.to(self.device), y_target.to(self.device)
                y_pred, hidden, cell = model(X, hidden, cell)

                y_pred_reshaped = y_pred.reshape(-1, model.vocabSize) 

                loss = model.loss(y_pred_reshaped, y_target.flatten().long())


                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25) 
                optimizer.step()
                epoch_loss += loss.item()


            if epoch%10 == 0:
                val_loss = round(evaluator.compute_loss(model), 5)

                print(f"Epoch {epoch}:\t Avg Train Loss: {round(epoch_loss/num_batches,5)}\t Avg Val Loss: {val_loss}")


