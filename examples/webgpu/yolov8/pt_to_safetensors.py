import torch
#from safetensors.torch import save_file
import ultralytics
#
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])
#state_dict = torch.load(f'./best(2).pt')
#weights = state_dict['model'].state_dict()
#safe_weights = {key.replace('model.model', 'model'): value for key, value in weights.items()}
#save_file(safe_weights, filename=f'./yolov8n.safetensors')
#
from safetensors.torch import load_file
from safetensors.torch import save_file

def rename(name: str):
    name = name.replace("model.0.", "net.b1.0.")
    name = name.replace("model.1.", "net.b1.1.")
    name = name.replace("model.2.m.", "net.b2.0.bottleneck.")
    name = name.replace("model.2.", "net.b2.0.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.3.", "net.b2.1.")
    name = name.replace("model.4.m.", "net.b2.2.bottleneck.")
    name = name.replace("model.4.", "net.b2.2.")
    name = name.replace("model.5.", "net.b3.0.")
    name = name.replace("model.6.m.", "net.b3.1.bottleneck.")
    name = name.replace("model.6.", "net.b3.1.")
    name = name.replace("model.7.", "net.b4.0.")
    name = name.replace("model.8.m.", "net.b4.1.bottleneck.")
    name = name.replace("model.8.", "net.b4.1.")
    name = name.replace("model.9.", "net.b5.0.")
    name = name.replace("model.12.m.", "fpn.n1.bottleneck.")
    name = name.replace("model.12.", "fpn.n1.")
    name = name.replace("model.15.m.", "fpn.n2.bottleneck.")
    name = name.replace("model.15.", "fpn.n2.")
    name = name.replace("model.16.", "fpn.n3.")
    name = name.replace("model.18.m.", "fpn.n4.bottleneck.")
    name = name.replace("model.18.", "fpn.n4.")
    name = name.replace("model.19.", "fpn.n5.")
    name = name.replace("model.21.m.", "fpn.n6.bottleneck.")
    name = name.replace("model.21.", "fpn.n6.")
    name = name.replace("model.22.", "head.")
    return name

data = torch.load("./best(2).pt")
#data = torch.load("/tmp/yolov8s.pt")
print(data)
tensors = data['model'].state_dict().items()
tensors = dict(tensors)
tensors = {rename(k): t for k, t in tensors.items()}

for k in tensors:
    if 'num_batches_tracked' in k:
        print(k)
        tensors[k] = torch.Tensor([1])

#print(data["model"])
save_file(tensors, "./yolov8n.safetensors")
#for k, v in tensors.items():
#    print(str(k), v.shape)
