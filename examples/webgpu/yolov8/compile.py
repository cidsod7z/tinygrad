from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import safe_save
from extra.export_model import export_model
from tinygrad.device import Device
from tinygrad.nn.state import safe_load, load_state_dict

from tinygrad.nn import Conv2d  
from examples.yolov8 import C2f, SPPF, Upsample, DetectionHead

class MyYOLOv8:
    def __init__(self, nc=6, width_mult=0.25, depth_mult=0.33):
        # Backbone
        self.conv1 = Conv2d(3, int(64*width_mult), 3, stride=2)
        self.conv2 = Conv2d(int(64*width_mult), int(128*width_mult), 3, stride=2)
        self.c2f1 = C2f(int(128*width_mult), int(128*width_mult), n=round(3*depth_mult), shortcut=True)
        self.conv3 = Conv2d(int(128*width_mult), int(256*width_mult), 3, stride=2)
        self.c2f2 = C2f(int(256*width_mult), int(256*width_mult), n=round(6*depth_mult), shortcut=True)
        self.conv4 = Conv2d(int(256*width_mult), int(512*width_mult), 3, stride=2)
        self.c2f3 = C2f(int(512*width_mult), int(512*width_mult), n=round(6*depth_mult), shortcut=True)
        self.conv5 = Conv2d(int(512*width_mult), int(1024*width_mult), 3, stride=2)
        self.c2f4 = C2f(int(1024*width_mult), int(1024*width_mult), n=round(3*depth_mult), shortcut=True)
        self.sppf = SPPF(int(1024*width_mult), int(1024*width_mult), k=5)

        # Head (ejemplo, debes adaptar las conexiones y concatenaciones según el grafo)
        self.upsample1 = Upsample(scale_factor=2, mode="nearest")
        self.c2f5 = C2f(int(512*width_mult)+int(1024*width_mult), int(512*width_mult), n=round(3*depth_mult))
        self.upsample2 = Upsample(scale_factor=2, mode="nearest")
        self.c2f6 = C2f(int(256*width_mult)+int(512*width_mult), int(256*width_mult), n=round(3*depth_mult))
        self.conv6 = Conv2d(int(256*width_mult), int(256*width_mult), 3, stride=2)
        self.c2f7 = C2f(int(256*width_mult)+int(512*width_mult), int(512*width_mult), n=round(3*depth_mult))
        self.conv7 = Conv2d(int(512*width_mult), int(512*width_mult), 3, stride=2)
        self.c2f8 = C2f(int(512*width_mult)+int(1024*width_mult), int(1024*width_mult), n=round(3*depth_mult))
        self.detect = DetectionHead(nc, [int(256*width_mult), int(512*width_mult), int(1024*width_mult)])

    def __call__(self, x):
        # Backbone
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.c2f1(x2)
        x4 = self.conv3(x3)
        x5 = self.c2f2(x4)
        x6 = self.conv4(x5)
        x7 = self.c2f3(x6)
        x8 = self.conv5(x7)
        x9 = self.c2f4(x8)
        x10 = self.sppf(x9)

        # Head (concatenaciones corregidas)
        y1 = self.upsample1(x10)
        y1 = y1.cat(x7, dim=1)
        y1 = self.c2f5(y1)
        y2 = self.upsample2(y1)
        y2 = y2.cat(x5, dim=1)
        y2 = self.c2f6(y2)
        y3 = self.conv6(y2)
        y3 = y3.cat(y1, dim=1)
        y3 = self.c2f7(y3)
        y4 = self.conv7(y3)
        y4 = y4.cat(x10, dim=1)
        y4 = self.c2f8(y4)

        out = self.detect([y2, y1, y4])
        return out

def map_weights(state_dict):
    # Mapea los nombres de tus atributos a los nombres de los pesos
    rename_map = {
        'conv1': 'model.0.conv',
        'conv2': 'model.1.conv',
        'c2f1': 'model.2',
        'conv3': 'model.3.conv',
        'c2f2': 'model.4',
        'conv4': 'model.5.conv',
        'c2f3': 'model.6',
        'conv5': 'model.7.conv',
        'c2f4': 'model.8',
        'sppf': 'model.9',
        # Agrega aquí los bloques del head si es necesario
    }
    mapped = {}
    for attr, prefix in rename_map.items():
        for k in state_dict:
            if k.startswith(prefix):
                mapped[k.replace(prefix, attr)] = state_dict[k]
    # Añade los pesos que no necesitan mapeo
    for k in state_dict:
        if not any(k.startswith(v) for v in rename_map.values()):
            mapped[k] = state_dict[k]
    return mapped

def load_state_dict_ignore_missing(model, state_dict):
    for k, v in model.__dict__.items():
        if hasattr(v, 'parameters'):
            params = v.parameters()
            for pk, pv in params.items():
                full_key = f"{k}.{pk}"
                if full_key in state_dict:
                    pv.replace(state_dict[full_key].to(pv.device)).realize()
                else:
                    # Si no existe, simplemente ignóralo
                    pass

if __name__ == "__main__":
    Device.DEFAULT = "WEBGPU"
    model = MyYOLOv8()
    state_dict = safe_load("models/NucleaDrone-6Class/yolov8n.safetensors")
    # Mapea los nombres antes de cargar
    state_dict = map_weights(state_dict)
    load_state_dict_ignore_missing(model, state_dict)
    
    prg, inp_sizes, out_sizes, state = export_model(
        model, Device.DEFAULT.lower(), Tensor.randn(1,3,640,640), model_name="yolov8"
    )
    dirname = Path(__file__).parent
    safe_save(state, (dirname / "net.safetensors").as_posix())
    with open(dirname / f"net.js", "w") as text_file:
       text_file.write(prg)
