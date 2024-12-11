import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


#https://www.analyticsvidhya.com/datahack/contest/black-friday/#

# 1. Načtení dat
pd.set_option('display.max_columns', None) #pro vypisovani vsech sloupcu kdyz budu chtit

tr_data = pd.read_csv('train.csv')  # nahraďte názvem vašeho souboru
train_data = tr_data.copy()

tst_data = pd.read_csv('test_Vges7qu.csv')  # nahraďte názvem vašeho souboru
test_data = tst_data.copy()




def count_avg_puchase_per_age_group(data):
    age_group_sums = data.groupby('Age')['Purchase'].mean().to_dict()
    return age_group_sums

avg_age_purchase = count_avg_puchase_per_age_group(train_data)

#train_data['Avg_Purchase_Age_Group'] = train_data['Age'].map(avg_age_purchase)


#drop age col
#train_data.drop(columns=['Age'], inplace=True)




# 2. Příprava dat
def preprocess_data(data,  train_columns=None):
    # Vyplnění chybějících hodnot v produktových kategoriích nulou

    data['Product_Category_2'] = data['Product_Category_2'].apply(lambda x: 0 if pd.isna(x) else x).astype(int)
    data['Product_Category_3'] = data['Product_Category_3'].apply(lambda x: 0 if pd.isna(x) else x).astype(int)

    #data['Age'] = data['Age'].map({'0-17': 0, '18-25': 1, '26-35': 2, '36-45': 3, '46-50': 4, '51-55': 5, '55+': 6})
    #data['Stay_In_Current_City_Years'] = data['Stay_In_Current_City_Years'].map({'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4})

    # One-Hot Encoding
    data = pd.get_dummies(data,
                          columns=['Gender', 'Age', 'Stay_In_Current_City_Years', 'Marital_Status', 'City_Category',   'Occupation', 'Product_Category_1', 'Product_Category_2', 'Product_Category_3'],
                          drop_first=False)

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



""" 3. Rozdělení na X (vstupy) a y (cílovou proměnnou)"""
#trenovaci sada
y_train_processed = preprocess_train_data['Purchase']
X_train_processed = preprocess_train_data.drop(columns=['Purchase', 'User_ID', 'Product_ID']) # odstranění sloupců 'Purchase', 'User_ID', 'Product_ID'

# testovaci sada
X_test = preprocess_test_data.drop(columns=['User_ID', 'Product_ID'])
# --- y predikuju

# Kontrola, zda mají trénovací a testovací data stejné sloupce
print("maji trenovaci a testovaci data stejne sloupce?")
print((X_train_processed.columns == X_test.columns).all())

"""3.1 rozdeleni trenovaci sady na train a validation ve zvolenem pomeru """
X_train, X_train_validation, y_train, y_train_validation = train_test_split(X_train_processed, y_train_processed, test_size=0.3,
                                                                            random_state=42, shuffle=False)

""" 4. Normalizace (StandardScaler) """
scaler = MinMaxScaler()   # zmena z standart scaler
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data
X_test_scaled = scaler.transform(X_test)
X_train_validation_scaled = scaler.transform(X_train_validation)  # Transform validation data

print("X_train_scaled min, max:", X_train_scaled.min(), X_train_scaled.max())
print("X_test_scaled min, max:", X_test_scaled.min(), X_test_scaled.max())


# """ 5. Převod na tensor formát """
# Převod trénovacích, validačních a testovacích dat na tensor formát
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_train_validation_tensor = torch.tensor(X_train_validation_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Převod cílové proměnné (y) na tensor
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_train_validation_tensor = torch.tensor(y_train_validation.values, dtype=torch.float32)


""" 6. Vytvoření DataLoaderu """
# Vytvoření TensorDataset pro trénovací, validační a testovací data
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
validation_dataset = TensorDataset(X_train_validation_tensor, y_train_validation_tensor)
test_dataset = TensorDataset(X_test_tensor)

# Vytvoření DataLoaderu pro trénování, validaci a testování
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),  # První lineární vrstva
            nn.LeakyReLU(),  # Aktivace
            nn.Dropout(p=0.4),

            nn.Linear(128, 64),  # Druhá lineární vrstva
            nn.LeakyReLU(),  # Aktivace
            nn.Dropout(p=0.5),

            nn.Linear(64, 1)  # Výstupní vrstva (1 neuron)
        )

    def forward(self, x):
        return self.model(x)  # Forward pass je zajištěn sekvencí


def train_model(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs):
    validation_losses, train_losses= [], []

    for epoch in range(num_epochs):
        model.train()  # Nastavení modelu do trénovacího módu
        running_loss = 0
        total_loss = 0
        total_samples = 0
        # training
        for X, y in train_loader:
            optimizer.zero_grad()

            # Forward pass
            pred_tr = model(X)

            # Výpočet ztráty
            loss = loss_fn(pred_tr, y.view(-1, 1))
            # running_loss += torch.abs(pred_tr - y.view(-1, 1)).mean().item()  # Mean absolute error

            total_loss += torch.abs(pred_tr - y.view(-1, 1)).sum().item()  # Sum of absolute errors (pro všechny vzorky v batchi)
            total_samples += y.size(0)  # Počet vzorků v tomto batchi

            # Backward pass
            loss.backward()

            # Aktualizace vah
            optimizer.step()

        # prumerna odchylka od realne ceny za epochu
        train_losses.append(total_loss / total_samples)

        # Validační smyčka
        model.eval()
        validation_loss = 0
        t_samples = 0
        with torch.no_grad():
            for Xbatch, ybatch in validation_loader:
                pred = model(Xbatch)
                validation_loss += torch.abs(pred - ybatch.view(-1, 1)).sum().item()  # Sum of absolute errors (pro všechny vzorky v batchi)
                t_samples += ybatch.size(0)  # Počet vzorků v tomto batchi
                #validation_loss += torch.abs(pred - y.view(-1, 1)).mean().item()  # Mean absolute error

        # Uložení průměrné validační ztráty pro epochu
        validation_losses.append(validation_loss / t_samples)
        print("epoch: ", epoch," done\n")



    return train_losses, validation_losses



def test_model(model, test_loader):
    model.eval() # Nastavení modelu do evalučního módu
    predictions = []
    with torch.no_grad(): # Vypnutí gradientu
        for X_batch in test_loader:
            pred = model(X_batch[0])  # Odstraňte nadbytečný rozměr
            predictions.append(pred)

    return torch.cat(predictions, dim=0) # Spojení všech predikcí do jednoho tensoru


""" Tady jsem sehnal nejaky styl hodnoceni kvality odhadovani"""
# Funkce pro výpočet RMSE
def calculate_rmse(predictions, actuals):
    return torch.sqrt(((predictions - actuals) ** 2).mean())
def calculate_r2(predictions, actuals):
    # Celková variabilita (variance of actuals)
    total_variance = ((actuals - actuals.mean()) ** 2).sum()
    # Variabilita chyb (variance of residuals)
    residual_variance = ((actuals - predictions) ** 2).sum()
    # R2 = 1 - (residual variance / total variance)
    return 1 - (residual_variance / total_variance)


model = NeuralNetwork(input_size=X_train.shape[1]) # Vytvoření instance modelu
loss_fn = nn.MSELoss()         # Mean Squared Error jako ztrátová funkce
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=0.002) # Adam jako optimalizační algoritmus

#
if __name__ == "__main__":
    num_epochs = 80
    train_losses, validation_losses = train_model(model, train_loader, validation_loader, loss_fn, optimizer, num_epochs)

    print("Train Losses: ", train_losses,"Valid losses: ", validation_losses)
    # Vykreslení ztrát
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, num_epochs + 1), train_losses, label='Avg Deviation from Actual purchase on TRAIN')
    plt.plot(range(1, num_epochs + 1), validation_losses, label='Avg deviation from Actual purchase on VALIDATION')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()

    # Uložení grafu jako PNG soubor
    plt.savefig('AA3 - allonehot40e')
    plt.close()

    torch.save(model.state_dict(), "model.pth")  # Uložení modelu

    inputsize = X_test.shape[1]
    #load model
    testingMODEL = NeuralNetwork(inputsize) # Create a new instance of the model
    testingMODEL.load_state_dict(torch.load("model.pth",  weights_only=True))

    # Testování modelu
    predictions = test_model(testingMODEL, test_loader)

    data = {'Purchase': predictions.numpy().flatten(), 'User_ID': test_data['User_ID'], 'Product_ID': test_data['Product_ID']}
    df = pd.DataFrame(data)
    df.to_csv('TESTpredictions.csv', index=False)




























