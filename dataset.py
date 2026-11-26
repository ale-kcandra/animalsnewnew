import os  # для работы с папками
from PIL import Image  # для открытия картинок
from torch.utils.data import Dataset  # базовый класс для датасета
from torchvision import transforms  # для превращения картинки в тензор

# список наших животных
CLASS_NAMES = ["elefante", "farfalla", "mucca", "pecora", "scoiattolo"]

# тут говорим что сделать с картинкой перед подачей в сеть
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # сжимаем до 64 на 64 пикселя
    transforms.ToTensor(),  # превращаем в тензор (числа от 0 до 1)
])

class AnimalDataset(Dataset):  # наш класс для загрузки данных
    def __init__(self, folder):  # при создании указываем путь к папке
        self.files = []  # сюда будем складывать пути к картинкам
        self.labels = []  # сюда будем складывать ответы (0, 1, 2...)
        for i, animal in enumerate(CLASS_NAMES):  # идем по каждому животному
            # ищем папку (например, elefante_train)
            animal_folder = os.path.join(folder, animal + "_train")
            if not os.path.exists(animal_folder):  # если такой нет
                animal_folder = os.path.join(folder, animal)  # пробуем без _train
            if not os.path.exists(animal_folder):  # если и такой нет
                continue  # пропускаем
            for file in os.listdir(animal_folder):  # идем по файлам в папке
                # если это картинка
                if file.lower().endswith((".jpg", ".png", ".jpeg")):
                    self.files.append(os.path.join(animal_folder, file))  # запоминаем путь
                    self.labels.append(i)  # запоминаем номер класса
        print(f"найдено {len(self.files)} картинок")  # пишем сколько нашли

    def __len__(self):  # сколько всего картинок
        return len(self.files)

    def __getitem__(self, idx):  # берем картинку по номеру
        img = Image.open(self.files[idx]).convert("RGB")  # открываем и переводим в rgb
        return transform(img), self.labels[idx]  # возвращаем картинку и ответ