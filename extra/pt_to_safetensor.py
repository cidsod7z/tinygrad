import os
import yaml
import glob
from ultralytics import YOLO
from safetensors.torch import save_file, load_file

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/NucleaDrone-6Class")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "../models/ground-truth")

# Paso 1: Conversión .pt → .safetensors
model_path = os.path.join(MODEL_PATH, "v01-yolo8n-box.pt")
model = YOLO(model_path)  # Modelo preentrenado

# Guardar el yaml en un archivo
yaml_path = os.path.join(MODEL_PATH, "yolov8n.yaml")
with open(yaml_path, "w") as f:
    yaml.dump(model.model.yaml, f)
print(f"YAML guardado en: {yaml_path}")

save_file(model.model.state_dict(), os.path.join(MODEL_PATH, "yolov8n.safetensors"))

# Paso 2: Cargar desde .safetensors
modelo_nuevo = YOLO(os.path.join(MODEL_PATH, "yolov8n.yaml"))  # Arquitectura base
modelo_nuevo.model.load_state_dict(load_file(os.path.join(MODEL_PATH, "yolov8n.safetensors")))

# Verificar inferencia en todas las imágenes de la carpeta "ground-truth"
imagenes_path = os.path.join(GROUND_TRUTH_PATH, "*.jpg")
imagenes = glob.glob(imagenes_path)

print(f"Imágenes encontradas: {len(imagenes)}")

for img_path in imagenes[:]:
    resultados = modelo_nuevo.predict(img_path)
    # print(resultados[0].speed)
    # resultados[0].show()