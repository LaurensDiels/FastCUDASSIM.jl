# Run inside a Python environment with torch, pytorch_msssim, and fused_ssim
# installed

import sys
from importlib.metadata import version
import time

import torch
import pytorch_msssim
import fused_ssim


print("Info:")
print("  Python ", sys.version)
print("  PyTorch ", version("torch"))  # (or torch.__version__)
print("  pytorch_msssim ", "? (probably 1.0.0)")  
# Defines neither __version__ nor the metadata required for
# version("pytorch_msssim"). 
# To be certain, use a manual pip list to check
print("  fused_ssim ", version("fused_ssim"))
print()


print("Single full HD RGB image pair SSIM:")
c, h, w, b = 3, 1080, 1920, 1
imgs1 = torch.empty(b, c, h, w, dtype=torch.float32, device="cuda")
imgs2 = torch.empty_like(imgs1)

def time_ssim(func, imgs1, imgs2, nb_runs):
    times = []
    for _ in range(nb_runs):
        imgs1[:] = torch.rand_like(imgs1)
        imgs2[:] = torch.rand_like(imgs2)
        start = time.perf_counter()
        func(imgs1, imgs2)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # in ms
    times = torch.tensor(times)
    print("    min:    ", times.min().item(),    " ms")
    print("    median: ", times.median().item(), " ms")
    print("    mean:   ", times.mean().item(),   " ms")

print("  pytorch_msssim:")
time_ssim(pytorch_msssim.ssim, imgs1, imgs2, 1_000)
# (Uses valid (i.e. no) padding, so not completely fair)

print("  fused_ssim:")
time_ssim(
    lambda imgs1, imgs2: fused_ssim.fused_ssim(imgs1, imgs2, train=False), 
    imgs1, imgs2, 
    10_000)
print()


print("Grayscale batch DSSIMs and gradients:")
c, h, w, b = 1, 256, 256, 32
imgs1 = torch.empty(b, c, h, w, dtype=torch.float32, device="cuda")
imgs2 = torch.empty_like(imgs1)

def time_ssim_and_back(func, imgs1, imgs2, nb_runs):
    times = []
    for _ in range(nb_runs):
        imgs1 = torch.rand_like(imgs1, requires_grad=True)
        # (Cannot modify in-place if it requires a grad)
        imgs2[:] = torch.rand_like(imgs2)
        start = time.perf_counter()
        s = func(imgs1, imgs2)
        s.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)  # in ms
    times = torch.tensor(times)
    print("    min:    ", times.min().item(), " ms")
    print("    median: ", times.median().item(), " ms")
    print("    mean:   ", times.mean().item(), " ms")

print("  pytorch_msssim:")
time_ssim_and_back(
    lambda imgs1, imgs2: (1 - pytorch_msssim.ssim(imgs1, imgs2)).sum(),
    imgs1, imgs2,
    500)


print("  fused_ssim:")
time_ssim_and_back(
    lambda imgs1, imgs2: 
        (1 - fused_ssim.fused_ssim(imgs1, imgs2, train=True)).sum(),
    imgs1, imgs2,
    10_000)