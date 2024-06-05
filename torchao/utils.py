import torch
import torch.utils.benchmark as benchmark
from typing import Tuple
from functools import reduce
from math import gcd


def benchmark_model(model, num_runs, input_tensor):
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    # benchmark
    for _ in range(num_runs):
        with torch.autograd.profiler.record_function("timed region"):
            model(input_tensor)

    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / num_runs

def profiler_runner(path, fn, *args, **kwargs):
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True) as prof:
        result = fn(*args, **kwargs)
    prof.export_chrome_trace(path)
    return result

def get_compute_capability():
    if torch.cuda.is_available():
        capability = torch.cuda.get_device_capability()
        return float(f"{capability[0]}.{capability[1]}")
    return 0.0

def skip_if_compute_capability_less_than(min_capability):
    import unittest
    def decorator(test_func):
        def wrapper(*args, **kwargs):
            if get_compute_capability() < min_capability:
                raise unittest.SkipTest(f"Compute capability is less than {min_capability}")
            return test_func(*args, **kwargs)
        return wrapper
    return decorator


def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    # Manual warmup

    f(*args, **kwargs)
    f(*args, **kwargs)

    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)",
        globals={"args": args, "kwargs": kwargs, "f": f},  # noqa: E501
    )
    measurement = t0.blocked_autorange()
    return measurement.mean * 1e6


def find_multiple(n: int, *args: Tuple[int]) -> int:
    k: int = reduce(lambda x, y: x * y // gcd(x, y), args + (1,))  # type: ignore[9]
    if n % k == 0:
        return n
    return n + k - (n % k)
