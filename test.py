import torch
print(torch.__version__)  # Version de PyTorch
print(torch.version.cuda)  # Version de CUDA reconnue par PyTorch
print(torch.cuda.is_available())  # Devrait retourner True
print(torch.cuda.device_count())  # Nombre de GPU disponibles
print(torch.cuda.get_device_name(0))  # Nom du GPU