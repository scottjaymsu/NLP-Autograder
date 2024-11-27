import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns
import os

# If the loss function should add weight to less frequent classes when training
WEIGHT = True

# If a graph measuring performance should be generated while training
GRAPH = False

BATCH_SIZE = 5000

EPOCHS = 500

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

# The model used to make predictions
class BertNN(torch.nn.Module):
    def __init__(self):
        super(BertNN, self).__init__()
        self.l1 = torch.nn.Linear(768, 6).to(device)

    def forward(self, x):
        return self.l1(x.squeeze(1)).squeeze(1)

# Used to represent the data to make batching/looping easier
class Data(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = list(sentences)
        self.labels = list(labels)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], torch.tensor(self.labels[idx], device=device, dtype=torch.int64)

# Gets berts encodings of the raw input and stores them in a file
def getEmbeddings(input : [], name : str) -> torch.Tensor:
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    bert = BertModel.from_pretrained("bert-base-cased").to(device)
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

# Calculates the Cross Entropy loss and Absolute error for model on x, y
def CELoss(model, x, y):
  model.eval()
  criterion = torch.nn.CrossEntropyLoss()
  distance = 0
  samples = 0

  data = Data(x, y)
  data_loader = DataLoader(data, batch_size = BATCH_SIZE)

  total_loss = 0

  for x, y in data_loader:
    predicted = model(x)
    loss = criterion(predicted, y)

    total_loss += loss.item() * x.size(0)
    distance += abs(torch.argmax(predicted, dim=1) - y).sum().item()
    samples += x.size(0)
  return total_loss / samples, distance / samples

def Eval(model, x, y):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()

    y_true = []
    y_pred = []

    data = Data(x, y)
    data_loader = DataLoader(data)

    total_loss = 0
    correct = 0
    distance = 0

    for x, y in data_loader:
        predicted = model.forward(x)
        loss = criterion(predicted, y)

        predicted = torch.argmax(predicted, dim=1)
        if y == predicted:
            correct += 1

        distance += abs(predicted - y)

        y_true.extend(y.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        total_loss += loss.item()
    model.train()

    matrix = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    return total_loss / len(data_loader), correct / len(data_loader), matrix, report, distance / len(data_loader)


# performs and prints testing on model for data x, y
# Cross Entropy Loss, Accuracy, Confusian matrix
def Test(model, x, y, name):
  stats = Eval(model, x, y)

  print(f"Statistics for {name}")
  print(f"Cross entropy loss: {stats[0]}")
  print(f"Correct accuracy (%): {stats[1] * 100}")
  print(f"Average absolute error:  {stats[4].item()}")
  print(f"Classification report:\n{stats[3]}")

  plt.figure(figsize=(8, 6))
  sns.heatmap(stats[2], annot=True, fmt='d', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.show()
  print()


if __name__ == '__main__':
    x, y = read_data('train.csv')
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, random_state=42)


    if WEIGHT:
      # Gives weights to the loss function,
      # So that the model is punished for classifing infrequent
      # data points more
      y = torch.tensor(y, dtype=torch.int64, device=device)
      counts = torch.bincount(y)
      numSamples = len(y)
      weights = numSamples / (len(counts) * counts.float())
      criterion = torch.nn.CrossEntropyLoss(weight=weights)

    else:
      criterion = torch.nn.CrossEntropyLoss()

    model = BertNN().to(device)
    optimizer = torch.optim.Adam(model.parameters())


    # Following code loads the embeddings from a file if they exist, else it creates them
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
    data_loader = DataLoader(data,batch_size=BATCH_SIZE, shuffle=True, num_workers = 0)

    x_vals = []
    training = []
    testing = []

    # If parameters already exists, it loads the weights and tests the model
    # Else it trains a new model from scratch
    parameters = cwd + "/parameters.pth"
    if os.path.exists(parameters):
        model.load_state_dict(torch.load(parameters))
    else:
        for i in tqdm(range(EPOCHS)):
            for x, y in data_loader:
                modelOutput = model(x)
                loss = criterion(modelOutput, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if GRAPH:
                  x_vals.append(i)
                  training.append(loss.item())
                  testing.append(CELoss(model,x_test,y_test))


    torch.save(model.state_dict(), "parameters.pth")

    if GRAPH:
      plt.plot(x_vals, training, label="Training")
      plt.xlabel('Epochs')
      plt.ylabel('Entropy Loss')
      plt.title('Loss throughout training')
      plt.plot(x_vals, testing, label="Testing")
      plt.legend(loc='upper right')
      plt.show()

    Test(model, x_train, y_train, "Training")
    Test(model, x_test, y_test, "Testing")
    Test(model, x_val, y_val, "Validation")