import os, ctypes, sys

# —— 可选：明确只用第 0 块卡
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# —— 如果你的 libcudnn/ libcudart 不在默认搜索路径，手动补一手：
DEFAULT_LIBPATHS = [
    "/usr/local/cuda/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/lib/x86_64-linux-gnu",
]
ld_paths = os.environ.get("LD_LIBRARY_PATH", "")
for p in DEFAULT_LIBPATHS:
    if p not in ld_paths:
        ld_paths = f"{p}:{ld_paths}" if ld_paths else p
os.environ["LD_LIBRARY_PATH"] = ld_paths

def load_lib(name_candidates):
    err = None
    for name in name_candidates:
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            err = e
    raise OSError(f"Failed to load any of {name_candidates}: {err}")

# 加载 CUDA runtime 与 cuDNN 9
libcudart = load_lib(["libcudart.so", "libcudart.so.13", "libcudart.so.12"])
libcudnn  = load_lib(["libcudnn.so", "libcudnn.so.9"])

# 绑定需要的符号
# cudaError_t cudaSetDevice(int device);
libcudart.cudaSetDevice.argtypes = [ctypes.c_int]
libcudart.cudaSetDevice.restype  = ctypes.c_int

# cudaError_t cudaFree(void* devPtr);
libcudart.cudaFree.argtypes = [ctypes.c_void_p]
libcudart.cudaFree.restype  = ctypes.c_int

# cudnnStatus_t cudnnCreate(cudnnHandle_t* handle)
cudnnHandle_t = ctypes.c_void_p
libcudnn.cudnnCreate.argtypes = [ctypes.POINTER(cudnnHandle_t)]
libcudnn.cudnnCreate.restype  = ctypes.c_int

# cudnnStatus_t cudnnDestroy(cudnnHandle_t handle)
libcudnn.cudnnDestroy.argtypes = [cudnnHandle_t]
libcudnn.cudnnDestroy.restype  = ctypes.c_int

# descriptors（随便测一个 4D tensor 描述符）
cudnnTensorDescriptor_t = ctypes.c_void_p
libcudnn.cudnnCreateTensorDescriptor.argtypes = [ctypes.POINTER(cudnnTensorDescriptor_t)]
libcudnn.cudnnCreateTensorDescriptor.restype  = ctypes.c_int

libcudnn.cudnnSetTensor4dDescriptor.argtypes = [
    cudnnTensorDescriptor_t, ctypes.c_int, ctypes.c_int,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]
libcudnn.cudnnSetTensor4dDescriptor.restype = ctypes.c_int

libcudnn.cudnnDestroyTensorDescriptor.argtypes = [cudnnTensorDescriptor_t]
libcudnn.cudnnDestroyTensorDescriptor.restype  = ctypes.c_int

# 辅助：把 cudnnStatus_t / cudaError_t 显示出来
def check(code, who, ok=0):
    if code != ok:
        raise RuntimeError(f"{who} failed -> status {code}")

def main():
    print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH",""))

    # 1) 初始化 CUDA 上下文（先 setDevice，再用 cudaFree(0) 触发 runtime init）
    check(libcudart.cudaSetDevice(0), "cudaSetDevice(0)")
    check(libcudart.cudaFree(ctypes.c_void_p(0)), "cudaFree(0)")

    # 2) 创建 cuDNN handle
    handle = cudnnHandle_t()
    check(libcudnn.cudnnCreate(ctypes.byref(handle)), "cudnnCreate")
    print("cudnnCreate OK ->", handle)

    # 3) 创建一个 4D tensor 描述符（NCHW = 1x1x64x64）
    desc = cudnnTensorDescriptor_t()
    check(libcudnn.cudnnCreateTensorDescriptor(ctypes.byref(desc)), "cudnnCreateTensorDescriptor")
    # mode=0: CUDNN_TENSOR_NCHW, dataType=0: CUDNN_DATA_FLOAT（这些枚举在 ctype 里用整型替代）
    check(libcudnn.cudnnSetTensor4dDescriptor(desc, 0, 0, 1, 1, 64, 64), "cudnnSetTensor4dDescriptor")
    print("cudnn tensor descriptor OK ->", desc)

    # 4) 清理
    check(libcudnn.cudnnDestroyTensorDescriptor(desc), "cudnnDestroyTensorDescriptor")
    check(libcudnn.cudnnDestroy(handle), "cudnnDestroy")
    print("All good ✅ cuDNN init works.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("❌ PROBE FAILED:", e)
        sys.exit(1)
