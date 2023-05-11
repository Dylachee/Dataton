import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim


class MorphemeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sequence = self.data[index]
        input_tensor = torch.tensor(sequence[:-1], dtype=torch.long)
        target_tensor = torch.tensor(sequence[1:], dtype=torch.long)
        return input_tensor, target_tensor

def split_word(word):
    # Разделение слова на морфемы по разделительному символу "-"
    morphemes = word.split('-')
    return morphemes

def prepare_data(data, batch_size, train_ratio=0.8):
    morpheme_sequences = []
    for line in data:
        word = line.strip()
        morphemes = split_word(word)
        morpheme_sequences.append(morphemes)
    
    dataset = MorphemeDataset(morpheme_sequences)
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size
    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# Загрузка данных из файла
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data

# Путь к файлу с данными
data_file = 'data.txt'

# Загрузка данных
data = load_data(data_file)

# Подготовка данных
batch_size = 32
train_loader, test_loader = prepare_data(data, batch_size)

# Определение переменной vocab и установка значения
vocab = {}  # Здесь вам нужно создать словарь морфем и присвоить его переменной vocab

# Определение архитектуры модели
class MorphologyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MorphologyModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        embedded = self.embedding(input)
        output, _ = self.rnn(embedded)
        output = self.fc(output)
        return output

# Определение параметров модели
input_size = len(vocab)  # Размер словаря морфем
hidden_size = 128  # Размер скрытого состояния модели
output_size = len(vocab)  # Размер выхода модели (количество морфем)

# Создание экземпляра модели
model = MorphologyModel(input_size, hidden_size, output_size)

# Определение функции потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Обучение модели
num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        
        # Передача входных данных через модель
        outputs = model(inputs)
        
        # Вычисление функции потерь
        loss = criterion(outputs.view(-1, output_size), targets.view(-1))
        
        # Обратное распространение ошибки и оптимизация параметров
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Вывод средней потери на каждой эпохе
    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

# Оценка производительности модели
def evaluate_model(model, data_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, dim=2)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.numel()
    
    accuracy = total_correct / total_samples
    return accuracy

# Оценка производительности модели на обучающей и тестовой выборках
train_accuracy = evaluate_model(model, train_loader)
test_accuracy = evaluate_model(model, test_loader)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

