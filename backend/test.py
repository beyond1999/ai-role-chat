import torch
print(torch.version.cuda)   # PyTorch 内置的 CUDA 版本
print(torch.backends.cudnn.version())  # cuDNN 版本（如果报错，说明没装好）

print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('cudnn:', torch.backends.cudnn.version())
print('available:', torch.cuda.is_available())
