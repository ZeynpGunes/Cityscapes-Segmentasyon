import os
import numpy as np
from PIL import Image

# Veri ve çıktı dizinlerini belirle
data_dir = "data/gtFine/train"
output_dir = "data/filtered_masks"
os.makedirs(output_dir, exist_ok=True)

# Trafik işareti sınıf ID'si
TRAFFIC_SIGN_CLASS = 220

# Şehirler üzerinde döngü başlat
for city in os.listdir(data_dir):
    city_path = os.path.join(data_dir, city)
    
    # Şehirdeki dosyalar üzerinde döngü başlat
    for file in os.listdir(city_path):
        # .DS_Store dosyasını atla
        if file == '.DS_Store':
            continue
        
        # Eğer dosya _labelIds.png ile bitiyorsa işle
        if file.endswith("_labelIds.png"):
            try:
                # Maskeyi oku
                mask_path = os.path.join(city_path, file)
                mask = np.array(Image.open(mask_path))
                
                # Trafik işaretlerini filtrele (sınıf ID'sine göre)
                traffic_sign_mask = (mask == TRAFFIC_SIGN_CLASS).astype(np.uint8) * 255
                
                # Çıktı dosyasını kaydet
                output_file = os.path.join(output_dir, file)
                Image.fromarray(traffic_sign_mask).save(output_file)
                
            except Exception as e:
                print(f"Dosya işlenirken hata oluştu: {file}. Hata: {e}")

print("Maskeler filtrelendi ve kaydedildi.")
