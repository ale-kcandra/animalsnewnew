import torch  # библиотека
import torch.nn as nn  # для слоев
from PIL import Image  # для картинок
from torchvision import transforms  # для трансформаций
from dataset import CLASS_NAMES  # список животных

# собираем такую же сеть, как при обучении
model = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(32 * 16 * 16, 128), nn.ReLU(),
    nn.Linear(128, len(CLASS_NAMES)),
)

model.load_state_dict(torch.load("model.pth"))  # грузим обученные веса
model.eval()  # режим проверки (ничего не учим)

def predict(image_path):  # функция для предсказания
    transform = transforms.Compose([  # те же трансформации
        transforms.Resize((64, 64)),  # сжимаем до 64
        transforms.ToTensor(),  # в тензор
    ])
    img = Image.open(image_path).convert("RGB")  # открываем картинку
    x = transform(img).unsqueeze(0)  # применяем трансформации (добавляем размер батча)

    with torch.no_grad():  # не считаем градиенты
        out = model(x)  # сеть предсказывает
        probs = torch.softmax(out, dim=1)[0]  # превращаем в проценты

    best_idx = probs.argmax().item()  # самый вероятный класс
    conf = probs[best_idx].item() * 100  # переводим в проценты

    print(f"это {CLASS_NAMES[best_idx]}, уверенность: {conf:.2f}%")  # печатаем ответ

if __name__ == "__main__":  # если запускаем напрямую
    predict(r"C:\Users\user\Desktop\нейронки\laba2\test\e831b4082cf1003ed1584d05fb1d4e9fe777ead218ac104497f5c978a4eebdbd_640.jpg")