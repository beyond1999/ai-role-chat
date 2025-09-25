# environment


# 方案 A｜WSL 一键脚本（最贴合你现在的环境）

> 适用于：Windows + WSL2（Ubuntu 22.04），`nvidia-smi` 显示 CUDA 12.x（你的是 12.9），用 WSL 的 `/usr/lib/wsl/lib` 运行时。

## 1) bootstrap\_wsl.sh

创建 `bootstrap_wsl.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

# 0) 基础依赖
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv python3-pip ffmpeg libsndfile1 git

# 1) 安装对口的 cuDNN 12（WSL 驱动报告 12.x）
# 如果已装过会自动跳过；如果 network in CN 可以把源换成你常用的镜像
if ! ldconfig -p | grep -q 'libcudnn\.so'; then
  wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/nvidia-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
    | sudo tee /etc/apt/sources.list.d/cuda-nvidia.list
  sudo apt-get update
  sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12 libcudnn9-headers-cuda-12
  sudo ldconfig
fi

# 2) 项目目录（按你的路径）
ROOT="/mnt/d/github-repo/ai-role-chat"
cd "$ROOT"

# 3) Python 虚拟环境
if [ ! -d .venv-asr ]; then
  python3.10 -m venv .venv-asr
fi
source .venv-asr/bin/activate
python -m pip install --upgrade pip

# 4) 固定稳定版本（你机器上测过更稳的组合）
python -m pip install --no-cache-dir \
  "ctranslate2==4.5.0" \
  "faster-whisper==1.0.3"

# 5) 准备模型缓存目录
mkdir -p /mnt/d/hf-cache
echo "✅ bootstrap done."
```

执行：

```bash
chmod +x bootstrap_wsl.sh
./bootstrap_wsl.sh
```

## 2) run\_asr.sh（干净环境 + CUDA 12 运行时 + 可开/关 cuDNN）

创建 `run_asr.sh`（放仓库根目录）：

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# 默认启用 cuDNN（如需关闭，运行时加 DISABLE_CUDNN=1）
CUDNN_FLAGS=""
if [ "${DISABLE_CUDNN:-0}" = "1" ]; then
  CUDNN_FLAGS="CT2_USE_CUDNN=0 CTRANSLATE2_USE_CUDNN=0"
fi

# 强制使用 WSL 的 CUDA 12.x runtime，避免 /usr/local/cuda-13 污染
exec env -i \
  PATH=/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/x86_64-linux-gnu \
  CUDA_VISIBLE_DEVICES=0 \
  CTRANSLATE2_CUDA_ALLOCATOR=cuda_malloc_async \
  CTRANSLATE2_NUM_THREADS=$(nproc) \
  CT2_CUDA_TRUE_FP16_GEMM=0 \
  HF_HOME=/mnt/d/hf-cache \
  ${CUDNN_FLAGS} \
  PYTHONUNBUFFERED=1 \
  ./.venv-asr/bin/python backend/asr.py
```

```bash
chmod +x run_asr.sh
# 正常跑（启用 cuDNN）
./run_asr.sh
# 如需禁用 cuDNN 兜底
DISABLE_CUDNN=1 ./run_asr.sh
```

## 3) backend/asr.py（稳态模板 + 可选 warmup）

替换为（或对照修改）：

```python
# backend/asr.py
import os, time
from faster_whisper import WhisperModel

# 可选：打印 ct2 详细日志
# os.environ["CT2_VERBOSE"] = "1"

MODEL_DIR = "../models/faster-whisper-small"   # 本地模型目录（存在则离线加载）
AUDIO = "data/sound/test.m4a"

def build_model():
    return WhisperModel(
        MODEL_DIR if os.path.exists(MODEL_DIR) else "Systran/faster-whisper-small",
        device="cuda",
        device_index=0,
        compute_type="int8_float16",        # 速度/显存/精度平衡；想更快可试 "float16" 或 "int8"
        download_root=os.environ.get("HF_HOME", "/tmp"),
        local_files_only=os.path.exists(MODEL_DIR)
    )

def warmup(m):
    # 可选：跑一个极短的静音/小切片，触发 CUDA/cuDNN 预热
    try:
        m.transcribe(AUDIO, beam_size=1, vad_filter=False, suppress_blank=True, max_new_tokens=1)
    except Exception:
        pass

def main():
    t0 = time.time()
    model = build_model()
    print(f"[1] model loaded in {time.time()-t0:.2f}s")

    # 预热（第一次会慢，之后稳定）
    t1 = time.time()
    warmup(model)
    print(f"[2] warmup in {time.time()-t1:.2f}s")

    # 正式转写
    t2 = time.time()
    segs, info = model.transcribe(AUDIO, beam_size=1, vad_filter=False, language=None)
    txt = "".join(s.text for s in segs)
    dt = time.time()-t2
    print(f"[3] transcribe {AUDIO} in {dt:.3f}s; lang={info.language} p={info.language_probability:.2f}")
    print(txt.strip())

if __name__ == "__main__":
    main()
```

---

# 方案 B｜Docker 部署（上服务器最省心）

> 适用于：服务器装有 **NVIDIA 驱动 + nvidia-container-toolkit**，能 `docker run --gpus all`。
> 选择 CUDA 12.x + cuDNN 9 的官方基础镜像。

## 1) Dockerfile

创建 `Dockerfile`（项目根目录）：

```dockerfile
# CUDA 12.x + cuDNN 9 + Ubuntu 22.04（devel 包含编译工具，若只跑时用 runtime 也可）
FROM nvidia/cuda:12.4.1-cudnn9-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/hf-cache \
    PYTHONUNBUFFERED=1

# 基础依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-venv \
    ffmpeg libsndfile1 git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 工作目录
WORKDIR /workspace

# 复制项目（按需调整路径）
COPY . /workspace

# Python 依赖（钉住稳定版本）
RUN python3 -m pip install --upgrade pip \
 && python3 -m pip install "ctranslate2==4.5.0" "faster-whisper==1.0.3"

# 预创建缓存
RUN mkdir -p /workspace/backend /workspace/data/sound ${HF_HOME}

# 可选：把常用模型提前下好（避免首次拉网）
# RUN python3 - <<'PY'\nfrom faster_whisper import WhisperModel; WhisperModel("Systran/faster-whisper-small")\nPY

# 入口（可换成你的脚本）
ENV CTRANSLATE2_CUDA_ALLOCATOR=cuda_malloc_async \
    CTRANSLATE2_NUM_THREADS=8 \
    CT2_CUDA_TRUE_FP16_GEMM=0
CMD ["python3", "backend/asr.py"]
```

## 2) 启动命令

```bash
# 宿主机需要安装 nvidia-container-toolkit，然后：
docker build -t asr-cuda12 .
docker run --rm -it --gpus all \
  -e HF_HOME=/cache/hf \
  -v $(pwd):/workspace \
  -v /path/to/cache:/cache/hf \
  asr-cuda12
```

> 说明
>
> * `--gpus all` 一定要有，否则容器里没有 GPU。
> * 用 `-v` 把模型缓存、音频数据挂进去。
> * 也可以用 `docker-compose` 固化这些参数。

---

## 3) docker-compose.yml（可选）

```yaml
version: "3.8"
services:
  asr:
    build: .
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - HF_HOME=/cache/hf
      - CTRANSLATE2_CUDA_ALLOCATOR=cuda_malloc_async
      - CT2_CUDA_TRUE_FP16_GEMM=0
    volumes:
      - .:/workspace
      - /path/to/cache:/cache/hf
    command: ["python3", "backend/asr.py"]
```

---

## 性能/稳定性提示（两套都适用）

* **compute\_type**：`int8_float16`（推荐） → 更快/省显存；如追极限再试 `float16`。
* **beam\_size=1**：贪婪搜索最快；只在需要更好字幕质量时再开大。
* **warm-up**：服务启动时跑一小段音频，后续调用稳定低延迟。
* **cuDNN**：Whisper 以 GEMM 为主，cuDNN 加速有限；不稳定可临时关：`CT2_USE_CUDNN=0`。
* **日志**：卡住时用 `CT2_VERBOSE=1` 或 `LD_DEBUG=libs` 快速定位。

watch -n 0.1 nvidia-smi
