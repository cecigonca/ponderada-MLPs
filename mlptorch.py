import torch
import torch.nn as nn
import torch.optim as optim

class MLPTorch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPTorch, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size) # Camada que conecta os ocultos com o input
        self.layer2 = nn.Linear(hidden_size, output_size) # Camada que conecta os ocultos com o output
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x)) # Ativação da primeira camada
        x = self.activation(self.layer2(x)) # Ativação da segunda camada
        return x

if __name__ == "__main__":
    # Dados de treinamento para XOR
    x_train = torch.Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = torch.Tensor([[0], [1], [1], [0]])

    model = MLPTorch(2, 4, 1)
    loss_fn = nn.MSELoss() # Função de perda (MSE)
    optimizer = optim.Adam(model.parameters(), lr=0.01) # Adam para considerar a atualizações de gradiente

    for epoch in range(10000): # Treinamento loop de 10000
        optimizer.zero_grad() 
        predictions = model(x_train)
        loss = loss_fn(predictions, y_train)
        loss.backward()
        optimizer.step()

    with torch.no_grad(): # Desabilita cálculo depois de treinado
        for input_tensor in x_train:
            predicted = model(input_tensor)
            # Resultado da predição
            print(f"Input: {input_tensor.numpy()} -> Output: {predicted.numpy()}")
