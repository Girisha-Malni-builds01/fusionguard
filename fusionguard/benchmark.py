import torch
import time

def measure(model, device, x, runs=50):
    model.to(device)
    model.eval()
    x = x.to(device)

    with torch.no_grad():
        for _ in range(10):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.time()
        for _ in range(runs):
            model(x)

        if device.type == "cuda":
            torch.cuda.synchronize()

        end = time.time()

    return (end - start) / runs
