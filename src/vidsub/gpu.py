"""CUDA 运行库引导。

本机无系统级 cuDNN/cuBLAS，依赖 pip 包 nvidia-cublas-cu12 / nvidia-cudnn-cu12。
在 `import faster_whisper` 之前用 RTLD_GLOBAL 预加载这些 .so，
等价于设置 LD_LIBRARY_PATH，且无需修改 shell 环境。
"""

from __future__ import annotations

import ctypes
import glob
import importlib.util
import os
import sys

_PRELOADED = False


def _nvidia_dir(module_name: str) -> str | None:
    spec = importlib.util.find_spec(module_name)
    if spec is None or not spec.submodule_search_locations:
        return None
    return str(spec.submodule_search_locations[0])


def preload_cuda_libs(verbose: bool = False) -> bool:
    """成功（或已加载）返回 True。仅 Linux 生效。"""
    global _PRELOADED
    if _PRELOADED or sys.platform != "linux":
        return True

    lib_dirs: list[str] = []
    for mod in ("nvidia.cublas", "nvidia.cudnn"):
        d = _nvidia_dir(mod)
        if d:
            lib_dirs.append(os.path.join(d, "lib"))

    if not lib_dirs:
        if verbose:
            print("[gpu] 未找到 pip 版 nvidia cublas/cudnn，将使用系统库路径", file=sys.stderr)
        return False

    # 先加载全部库（依赖在前），关键库 libcublas/libcudnn 最后确认
    patterns = ["libcublasLt.so.*", "libcublas.so.*", "libcudnn_*.so.*", "libcudnn.so.*"]
    loaded: list[str] = []
    for pattern in patterns:
        for d in lib_dirs:
            for path in sorted(glob.glob(os.path.join(d, pattern))):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(path))
                except OSError:
                    # 依赖可能尚未就绪，最后再试一轮
                    pass

    # 第二轮：补上第一轮因依赖顺序失败的库
    for pattern in patterns:
        for d in lib_dirs:
            for path in sorted(glob.glob(os.path.join(d, pattern))):
                name = os.path.basename(path)
                if name in loaded:
                    continue
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(name)
                except OSError as exc:
                    if pattern.startswith("libcublas.") or pattern == "libcudnn.so.*":
                        raise RuntimeError(f"无法预加载 {name}: {exc}") from exc

    if verbose:
        print(f"[gpu] 已预加载 {len(loaded)} 个 CUDA 库")
    _PRELOADED = True
    return True
