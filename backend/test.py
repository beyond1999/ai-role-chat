import ctypes

try:
    libcudnn = ctypes.cdll.LoadLibrary("libcudnn.so")
    handle = ctypes.c_void_p()
    status = libcudnn.cudnnCreate(ctypes.byref(handle))
    print("cudnnCreate status:", status)
except OSError as e:
    print("cuDNN not found:", e)
