import ctypes
try:
    libcudnn = ctypes.CDLL("libcudnn.so")
    print("✅ cuDNN loaded:", libcudnn)
except OSError as e:
    print("❌ cuDNN not found:", e)
