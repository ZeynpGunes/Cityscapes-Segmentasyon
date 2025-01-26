import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from scripts.model import UNet

# Veri seti sınıfı
class CityscapesDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Alt klasörleri ve dosyaları listele
        self.images = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    self.images.append(os.path.join(root, file))
        
        print(f"Found {len(self.images)} images.")
        for image in self.images:
            print(f"Image: {image}")
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(img_path).replace('_leftImg8bit', '_gtFine_labelIds'))
        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256))
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

# Veri yükleme
transform = transforms.Compose([transforms.ToTensor()])  # Resimleri tensor'a dönüştür
dataset = CityscapesDataset("data/leftImg8bit/train", "data/filtered_masks", transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model ve optimizasyon
model = UNet()
criterion = torch.nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Eğitim döngüsü
for epoch in range(10):
    model.train()  # Eğitim moduna al
    epoch_loss = 0.0  # Epoch başına toplam kayıp
    for images, masks in dataloader:
        # Modelin çıktısını al
        outputs = model(images)
        
        # Kaybı hesapla
        loss = criterion(outputs, masks)
        
        # Geri yayılım ve optimizasyon
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Epoch kaybını güncelle
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(dataloader)}")
