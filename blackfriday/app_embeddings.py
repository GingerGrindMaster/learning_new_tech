import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

label_encoder = LabelEncoder()

# Načtení dat
tr_data = pd.read_csv('train.csv')  # nahraďte názvem vašeho souboru
train_data = tr_data.copy()

tst_data = pd.read_csv('test_Vges7qu.csv')  # nahraďte názvem vašeho souboru
test_data = tst_data.copy()

scaler = MinMaxScaler()

def count_avg_puchase_per_age_group(data):
    age_group_sums = data.groupby('Age')['Purchase'].mean().to_dict()
    return age_group_sums

avg_age_purchase = count_avg_puchase_per_age_group(train_data)

train_data['Avg_Purchase_Age_Group'] = train_data['Age'].map(avg_age_purchase)


#drop age col
#train_data.drop(columns=['Age'], inplace=True)

# Příprava dat
def preprocess_data(data, train_columns=None):

    data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4})
    #data['Age'] = data['Age'].map({'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6})

    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['Gender', 'City_Category', 'Age'], drop_first=False)

    # labels pro kategorie ve sloupcích
    data['Occupation'] = label_encoder.fit_transform(data['Occupation'])
    data['Product_Category_1'] = label_encoder.fit_transform(data['Product_Category_1'].fillna(0).astype(str))
    data['Product_Category_2'] = label_encoder.fit_transform(data['Product_Category_2'].fillna(0).astype(str))
    data['Product_Category_3'] = label_encoder.fit_transform(data['Product_Category_3'].fillna(0).astype(str))

    # Převod pouze sloupců obsahujících Boolean hodnoty na 0 a 1
    bool_columns = data.select_dtypes(include=['bool']).columns
    data[bool_columns] = data[bool_columns].astype('int64')

    # Zajištění, že testovací data mají stejné sloupce jako trénovací data
    if train_columns is not None:
        data = data.reindex(columns=train_columns, fill_value=0)

    return data

# Preprocess both train and test data
preprocess_train_data = preprocess_data(train_data)
preprocess_test_data = preprocess_data(test_data, train_columns=preprocess_train_data.columns)
preprocess_test_data.drop(columns=['Purchase'], inplace=True)

# Rozdělení na X (vstupy) a y (cílovou proměnnou)
y_train_processed = preprocess_train_data['Purchase']
X_train_processed = preprocess_train_data.drop(columns=['Purchase', 'User_ID', 'Product_ID'])  # odstranění sloupců 'Purchase', 'User_ID', 'Product_ID'
X_test = preprocess_test_data.drop(columns=['User_ID', 'Product_ID'])




"""vytvoreni emb tensoru """
# Convert -1 to a special index (this index will be used to represent missing values)
cat_1_train = torch.tensor(X_train_processed['Product_Category_1'].replace(-1, 0).values, dtype=torch.long)
cat_2_train = torch.tensor(X_train_processed['Product_Category_2'].replace(-1, 0).values, dtype=torch.long)
cat_3_train = torch.tensor(X_train_processed['Product_Category_3'].replace(-1, 0).values, dtype=torch.long)
occupation_train = torch.tensor(X_train_processed['Occupation'].values, dtype=torch.long)

#embeding sloupce odstraneny od normalnich hodnot
X_train_processed.drop(columns=['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Occupation'], inplace=True)



# Normalizace
X_train_scaled = scaler.fit_transform(X_train_processed)

# Převod na tensor formát
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_processed.values, dtype=torch.float32)


# Vytvoření TensorDataset pro trénovací a testovací data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor, cat_1_train, cat_2_train, cat_3_train, occupation_train)
# Vytvoření DataLoaderu pro trénování a testování
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

p1unique = preprocess_train_data['Product_Category_1'].nunique()
p2unique = preprocess_train_data['Product_Category_2'].nunique()
p3unique = preprocess_train_data['Product_Category_3'].nunique()
occ_unique = preprocess_train_data['Occupation'].nunique()



# Zpracování X_test
cat_1_test = torch.tensor(X_test['Product_Category_1'].replace(-1, 0).values, dtype=torch.long)
cat_2_test = torch.tensor(X_test['Product_Category_2'].replace(-1, 0).values, dtype=torch.long)
cat_3_test = torch.tensor(X_test['Product_Category_3'].replace(-1, 0).values, dtype=torch.long)
occupation_test = torch.tensor(X_test['Occupation'].values, dtype=torch.long)

# Odstranění sloupců kategorií z X_test
X_test.drop(columns=['Product_Category_1', 'Product_Category_2', 'Product_Category_3', 'Occupation'], inplace=True)

# Normalizace X_test
X_test_scaled = scaler.transform(X_test)  # Použijeme scaler, který byl fitován na trénovacích datech

# Převod na tensor formát
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Vytvoření DataLoaderu pro testovací data
test_dataset = TensorDataset(X_test_tensor, cat_1_test, cat_2_test, cat_3_test, occupation_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)





# Neural Network class with embeddings
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, embedding_size):
        super(NeuralNetwork, self).__init__()

        # Embedding layers for categorical variables (Product Category 1, 2, 3 and Occupation)
        self.product_category_1_emb = nn.Embedding(p1unique, embedding_size)  # Adjust size based on data
        self.product_category_2_emb = nn.Embedding(p2unique, embedding_size)
        self.product_category_3_emb = nn.Embedding(p3unique, embedding_size)
        self.occupation_emb = nn.Embedding(occ_unique, embedding_size)

        # Linear layers
        self.layer1 = nn.Linear(input_size + embedding_size * 4, 128)  # První vrstva bere v úvahu numerické i kategorie
        self.dropout1 = nn.Dropout(p=0.01)
        self.layer2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(p=0.01)
        self.layer3 = nn.Linear(64, 1)

    def forward(self, x, cat_1, cat_2, cat_3, occupation):
        # Extract embeddings for categorical variables
        cat_1_emb = self.product_category_1_emb(cat_1)
        cat_2_emb = self.product_category_2_emb(cat_2)
        cat_3_emb = self.product_category_3_emb(cat_3)
        occupation_emb = self.occupation_emb(occupation)

        # Concatenate embeddings with numerical features
        x = torch.cat((x, cat_1_emb, cat_2_emb, cat_3_emb, occupation_emb), dim=1)

        # Pass through fully connected layers
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        return self.layer3(x)



# Inicializace modelu
model = NeuralNetwork(input_size=X_train_scaled.shape[1], embedding_size=5)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.006, weight_decay=0.003)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Trénovací smyčka
def train_model(model, train_loader, num_epochs):
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0

        for X_batch, y_batch, cat_1_batch, cat_2_batch, cat_3_batch, occupation_batch in train_loader:
            optimizer.zero_grad()

            # Forward pass
            pred = model(X_batch, cat_1_batch, cat_2_batch, cat_3_batch, occupation_batch)

            loss = loss_fn(pred.squeeze(), y_batch)

            running_loss += torch.abs(pred - y_batch.view(-1, 1)).mean().item()

            loss.backward()
            optimizer.step()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f'Epoch {epoch+1} ')
    return train_losses


def test_model(model, test_loader):
    model.eval()
    predictions = []

    with torch.no_grad():
        for X_batch, cat_1_batch, cat_2_batch, cat_3_batch, occupation_batch in test_loader:
            # Forward pass
            pred = model(X_batch, cat_1_batch, cat_2_batch, cat_3_batch, occupation_batch)
            predictions.extend(pred.squeeze().cpu().numpy())

    return predictions






if __name__ == "__main__":
    num_epochs = 10
    train_losses = train_model(model, train_loader, num_epochs)

    print("Train Losses: ", train_losses)
    # Vykreslení ztrát
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Avg Deviation from Actual purchase on TRAIN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Over Epochs\nlr={optimizer.param_groups[0]["lr"]}, weight_decay={optimizer.param_groups[0]["weight_decay"]}, dropout1={model.dropout1.p}, dropout2={model.dropout2.p}')
    plt.legend()

    # Uložení grafu jako PNG soubor
    plt.savefig('emb6_scheduler05_5')
    plt.close()

    torch.save(model.state_dict(), "model_embedding.pth")  # Uložení modelu

    inputsize = X_test.shape[1]
    # load model
    testingMODEL = NeuralNetwork(inputsize, embedding_size=5)  # Create a new instance of the model
    testingMODEL.load_state_dict(torch.load("model_embedding.pth", weights_only=True))

    # Testování modelu
    predictions = test_model(testingMODEL, test_loader)
    print("done")


    data = {'Purchase': predictions, 'User_ID': test_data['User_ID'],
            'Product_ID': test_data['Product_ID']}
    df = pd.DataFrame(data)
    df.to_csv('emd_test.csv', index=False)