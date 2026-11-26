import torch  # главная библиотека
import torch.nn as nn  # для слоев сети
from torch.utils.data import DataLoader, random_split  # для батчей и деления данных
from dataset import AnimalDataset, CLASS_NAMES  # импортируем наши классы

# собираем сеть по слоям
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),  # свертка: ищем 16 признаков
    nn.ReLU(),  # активация (убирает минусы)
    nn.MaxPool2d(2),  # сжимаем картинку в 2 раза
    nn.Conv2d(16, 32, 3, padding=1),  # вторая свертка: ищем 32 признака
    nn.ReLU(),  # активация
    nn.MaxPool2d(2),  # снова сжимаем в 2 раза
    nn.Flatten(),  # вытягиваем все числа в один ряд
    nn.Linear(32 * 16 * 16, 128),  # полносвязный слой (128 нейронов)
    nn.ReLU(),  # активация
    nn.Linear(128, len(CLASS_NAMES)),  # выходной слой (5 классов)
)

# путь к папке с картинками
folder = r"Animals_recognize-main\animal-dataset\versions\1\animal_dataset_intermediate\train"
dataset = AnimalDataset(folder)  # загружаем картинки

if len(dataset) == 0:  # если пусто
    exit()  # выходим

n_val = int(len(dataset) * 0.2)  # 20% картинок на проверку
n_train = len(dataset) - n_val  # остальные 80% на обучение
train_set, val_set = random_split(dataset, [n_train, n_val])  # делим
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)  # батчи для обучения (перемешанные)
val_loader = DataLoader(val_set, batch_size=32, shuffle=False)  # батчи для проверки

loss_fn = nn.CrossEntropyLoss()  # функция ошибки (для классификации)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # оптимизатор (учит сеть)

for epoch in range(10):  # 10 эпох обучения
    model.train()  # переводим сеть в режим обучения
    for images, labels in train_loader:  # идем по батчам
        optimizer.zero_grad()  # обнуляем старые ошибки
        outputs = model(images)  # сеть делает предсказание
        loss = loss_fn(outputs, labels)  # считаем ошибку
        loss.backward()  # обратный проход (ищем где ошиблись)
        optimizer.step()  # обновляем веса

    model.eval()  # переводим сеть в режим проверки
    correct, total = 0, 0  # счетчики правильных ответов
    with torch.no_grad():  # отключаем вычисление градиентов (для скорости)
        for images, labels in val_loader:  # идем по проверочным картинкам
            outputs = model(images)  # предсказываем
            pred = outputs.argmax(dim=1)  # берем самый вероятный класс
            correct += (pred == labels).sum().item()  # плюс если угадали
            total += labels.size(0)  # плюс общее количество
            
    print(f"эпоха {epoch+1}/10, точность: {correct/total*100:.1f}%")  # печатаем точность

torch.save(model.state_dict(), "model.pth")  # сохраняем веса в файл