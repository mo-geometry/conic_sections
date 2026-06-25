import sys
import numpy as _np

GPU_REQUESTED = '--gpu-on' in sys.argv
GPU_AVAILABLE = False
xp = _np

if GPU_REQUESTED:
    try:
        import cupy as _cp
        _cp.array([0]) + 1
        xp = _cp
        GPU_AVAILABLE = True
        print('GPU acceleration: ON (cupy)')
    except Exception as e:
        print(f'--gpu-on requested but GPU is unavailable ({e}); falling back to CPU.')
        xp = _np
        GPU_AVAILABLE = False
else:
    print('GPU acceleration: OFF (pass --gpu-on to enable)')


def to_gpu(array):
    return xp.asarray(array)


def to_cpu(array):
    if GPU_AVAILABLE:
        return _cp.asnumpy(array)
    return _np.asarray(array)
