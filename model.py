import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

## parameters.pth is the network parameters after training for 500 epochs
## sentence_embeddings_test.pt and sentence_embeddings_train is Bert's encoding of the train.csv set
## Done so encodding doesn't have to be computed every epoch
## delete or move paramaters.pth from directory to train model from scratch

def read_data(filename : str) -> []:
    data = pd.read_csv(filename)
    data = data.dropna()
    x = data.iloc[:,1]
    y = data.iloc[:,2]
    y = y-1
    return [x,y]

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

class lstm(torch.nn.Module):
    def __init__(self):
        super(lstm, self).__init__()
        self.lstm = torch.nn.LSTM(input_size=768,hidden_size=48, num_layers=1).to(device)
        self.l3 = torch.nn.Sigmoid()
        self.l4 = torch.nn.Linear(48, 6).to(device)

    def forward(self, x):
        x = x.squeeze(1)
        x, _ = self.lstm(x)
        x = self.l3(x[:,-1,:])
        return self.l4(x)

class Data(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = list(sentences)
        self.labels = list(labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

def getEmbeddings(input : [], name : str) -> torch.Tensor:
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()

    newdata = []

    for x in tqdm(input):
        with torch.no_grad():
            x = tokenizer(x, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = bert(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                            token_type_ids=x['token_type_ids'])
            sentence_embeddings = outputs.last_hidden_state[:, 0, :]
            newdata.append(sentence_embeddings.unsqueeze(1))

    newdata_tensor = torch.stack(newdata)

    torch.save(newdata_tensor, name + ".pt")

    return newdata_tensor.to(device)

def Eval(model, x, y):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    data = Data(x, y)
    data_loader = DataLoader(data)

    total_loss = 0
    correct = 0

    for x, y in data_loader:
        predicted = model.forward(x)

        y = y.to(dtype=torch.float32)
        y = y.to(device)
        predicted = predicted.to(torch.float32)
        y = y.to(torch.int64)
        loss = criterion(predicted, y)

        predicted = torch.argmax(predicted, dim=1)
        if y == predicted:
            correct += 1

        total_loss += loss.item()
    model.train()
    return total_loss / len(data_loader), correct / len(data_loader)




if __name__ == '__main__':
    x, y = read_data('train.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=42)

    model = lstm().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    cwd = os.getcwd()
    embed = cwd + "/sentence_embeddings_train.pt"
    if os.path.exists(embed):
        x_train = torch.load(embed, weights_only = True).to(device)
    else:
        x_train = getEmbeddings(x_train, "sentence_embeddings_train")

    embed = cwd + "/sentence_embeddings_test.pt"
    if os.path.exists(embed):
        x_test = torch.load(embed, weights_only=True).to(device)
    else:
        x_test = getEmbeddings(x_test, "sentence_embeddings_test")

    embed = cwd + "/sentence_embeddings_val.pt"
    if os.path.exists(embed):
        x_val = torch.load(embed, weights_only=True).to(device)
    else:
        x_val = getEmbeddings(x_val, "sentence_embeddings_val")


    data = Data(x_train, y_train)
    data_loader = DataLoader(data,batch_size=5000, shuffle=True, num_workers = 0)

    x_vals = []
    training = []
    testing = []

    parameters = cwd + "/parameters.pth"

    if os.path.exists(parameters):
        model.load_state_dict(torch.load(parameters))
    else:
        for i in tqdm(range(500)):
            for x, y in data_loader:
                modelOutput = model.forward(x)

                y = y.to(dtype=torch.int64)
                y = y.to(device)
                loss = criterion(modelOutput, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                ## used for generating plot
                ## comment out for much faster training

                # x_vals.append(i)
                # training.append(loss.item())
                # testing.append(Eval(model,x_test,y_test))


    torch.save(model.state_dict(), "parameters.pth")

    plt.plot(x_vals, training, label="Training")
    plt.xlabel('Epochs')
    plt.ylabel('Entropy Loss')
    plt.title('Loss throughout training')
    plt.plot(x_vals, testing, label="Testing")
    plt.legend(loc='upper right')
    plt.show()

    training_loss = Eval(model, x_train, y_train)
    testing_loss = Eval(model, x_test, y_test)
    val_loss = Eval(model, x_val, y_val)

    print(f"Training: Cross entropy loss: {training_loss[0]}")
    print(f"Training: Correct accuracy (%): {training_loss[1] * 100}")
    print(f"Testing: Cross entropy loss: {testing_loss[0]}")
    print(f"Testing: Correct accuracy (%): {testing_loss[1] * 100}")
    print(f"Validatino: Cross entropy loss: {val_loss[0]}")
    print(f"Validation: Correct accuracy (%): {val_loss[1] * 100}")


