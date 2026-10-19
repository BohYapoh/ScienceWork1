import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
import SchrodingerEquation as Sch

input1 = np.load("input.npy")
output = np.load("output.npy")
output_mean = output.mean(axis=0)
output_std = output.std(axis=0)
output = (output - output_mean) / output_std
input1 = torch.from_numpy(input1).to(torch.float64)
output = torch.from_numpy(output).to(torch.float64)



t_ev = np.linspace(0., 1, 1000)

# Архитектура нейросети
class QuantumNet(nn.Module):
    def __init__(self):
        super(QuantumNet, self).__init__()
        self.layer1 = nn.Linear(4, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 3)

    def forward (self, x):
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        return self.layer4(x)


class MyDataset(Dataset):
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        x = self.input[idx]
        target = self.output[idx]
        return x, target
dataset = MyDataset(input1, output)
data = DataLoader(
    dataset,
    batch_size=512,
    shuffle=True
)

model = QuantumNet()
model = model.double()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
model.train()
print(input1)
scheduler = ExponentialLR(optimizer, gamma=0.99999816)
num = int(input())

if num == 1:      #Загружаем из файла
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
else:               # Или обучаем
    for i in range(100):
        for x, target in data:
           optimizer.zero_grad()
           output = model (x)
           loss = criterion(output, target)
           loss.backward()
           optimizer.step()
           scheduler.step()
           print(loss)
    model.eval()
delta =0

for i in range(100):      # Проверка
    New_X, norm = Sch.Make_random_Unitar()
    New_X_formated = Sch.Pauli_from_Unitar(New_X)
    New_X_formated = torch.tensor(New_X_formated, dtype=torch.float64)
    Y_pred = model(New_X_formated)
    Y_pred_new = Y_pred.detach().numpy()
    Y_pred_new = Y_pred_new * output_std + output_mean
    print("Придуманный импульс", Y_pred_new)
    phase, Freq_E, Freq_Rabi, norm = Sch.Calculate_impulse(New_X)
    print("Посчитанный импульс", phase, Freq_E, Freq_Rabi)
    sv_init, sv_final, bloch_sv_final = Sch.Generate_random_state(New_X, 1, 0)
    bloch = Sch.Simulate_impulse(sv_init, t_ev, Y_pred_new[0], Y_pred_new[1], Y_pred_new[2])
    print("Посчитанный блох", bloch_sv_final)
    print("Придуманный блох",bloch[999])
    delta += np.sqrt((bloch[999][0]- bloch_sv_final[0]) ** 2 + (bloch[999][1]- bloch_sv_final[1]) ** 2
                     + (bloch[999][2]- bloch_sv_final[2]) ** 2)
    print("Средняя длина вектора ошибки",delta / (1+i))
    Sch.Visualize(bloch, norm, bloch_sv_final)
print(delta/100)
torch.save(model.state_dict(), "model.pth")
