import torch
import psutil
import platform
import time
import os
import tempfile


def get_cpu_info():
    """Gathers CPU information using psutil."""
    # psutil.cpu_freq() can return None if frequency is not available
    freq = psutil.cpu_freq()
    current_freq = int(freq.current) if freq else "N/A"

    return {
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "frequency_mhz": current_freq,
        "type": platform.processor(),
    }


def get_memory_info():
    """Gathers memory information using psutil."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": f"{mem.total / (1024**3):.2f}",
        "available_gb": f"{mem.available / (1024**3):.2f}",
    }


def get_storage_info():
    """Gathers root partition storage info using psutil."""
    disk = psutil.disk_usage("/")
    return {
        "total_gb": f"{disk.total / (1024**3):.2f}",
        "available_gb": f"{disk.free / (1024**3):.2f}",
    }


def get_gpu_info():
    """Gathers basic GPU information using torch."""
    if not torch.cuda.is_available():
        return []
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append(
            {
                "name": props.name,
                "memory_gb": f"{props.total_memory / (1024**3):.2f}",
            }
        )
    return gpus


def test_gpu_allocation(device, size_gb=1):
    """Tests allocating a tensor of a given size on a GPU."""
    try:
        # 4 bytes per float32
        tensor_size_elements = int(size_gb * (1024**3) / 4)
        tensor = torch.empty(tensor_size_elements, dtype=torch.float32, device=device)
        # short operation to ensure memory is physically allocated
        tensor.fill_(1.0)
        torch.cuda.synchronize(device)
        del tensor
        return "Ok"
    except RuntimeError as e:
        return f"Failed: {e}"


def benchmark_gpu_speed(device, matrix_size=4096):
    """Performs a simple matrix multiplication to benchmark GPU speed."""
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Warm-up run
    torch.matmul(a, b)
    torch.cuda.synchronize(device)

    start_time = time.time()
    for _ in range(5):
        torch.matmul(a, b)
    torch.cuda.synchronize(device)
    end_time = time.time()

    duration = (end_time - start_time) / 5
    # Matmul FLOPS is 2 * N^3 for square matrices
    flops = 2 * (matrix_size**3)
    gflops = (flops / duration) / 1e9

    return gflops


def benchmark_transfer_speed(device, size_gb=1):
    """Benchmarks the data transfer speed between CPU and GPU."""
    tensor_size_bytes = int(size_gb * 1024**3)
    # Create a tensor on the CPU in pinned memory for faster transfer
    cpu_tensor = torch.empty(tensor_size_bytes, dtype=torch.uint8, pin_memory=True)

    torch.cuda.synchronize(device)
    start_time = time.time()
    gpu_tensor = cpu_tensor.to(device, non_blocking=True)
    torch.cuda.synchronize(device)
    end_time = time.time()

    # Transfer speed in GB/s
    speed = size_gb / (end_time - start_time)
    del gpu_tensor
    return speed


def test_filesystem():
    """Writes and reads a temporary file in the script's directory."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Create a temporary file handle securely
    fd, path = tempfile.mkstemp(dir=script_dir, prefix="fs_test_")

    try:
        # Test writing
        with os.fdopen(fd, "w") as tmp_file:
            tmp_file.write("test")
        print("  Writing: Ok")

        # Test reading
        with open(path, "r") as tmp_file:
            content = tmp_file.read()
            if content == "test":
                print("  Reading: Ok")
            else:
                print("  Reading: Failed (content mismatch)")

    except IOError as e:
        print(f"  Filesystem test failed: {e}")
    finally:
        # Ensure cleanup
        if os.path.exists(path):
            os.remove(path)


def main():
    """Main function to run the server diagnostics."""
    print("--- System Information ---")
    print(f"Python Version: {platform.python_version()}")
    print(f"Torch Version: {torch.__version__}")

    print("\n--- CPU Information ---")
    cpu_info = get_cpu_info()
    print(f"Type: {cpu_info['type']}")
    print(f"Cores: {cpu_info['cores']} (Threads: {cpu_info['threads']})")
    print(f"Frequency: {cpu_info['frequency_mhz']} MHz")

    print("\n--- Memory (RAM) Information ---")
    mem_info = get_memory_info()
    print(f"Total: {mem_info['total_gb']} GB (Available: {mem_info['available_gb']} GB)")

    print("\n--- Storage Information (Root) ---")
    storage_info = get_storage_info()
    print(f"Total: {storage_info['total_gb']} GB (Available: {storage_info['available_gb']} GB)")

    print("\n--- GPU Benchmarks ---")
    if not torch.cuda.is_available():
        print("No CUDA-enabled GPU found.")
    else:
        gpus = get_gpu_info()
        for i, gpu in enumerate(gpus):
            device = f"cuda:{i}"
            print(f"GPU: {gpu['name']} ({device}) | Memory: {gpu['memory_gb']} GB")

            alloc_status = test_gpu_allocation(device)
            print(f"  1GB Allocation: {alloc_status}")

            # Only run benchmarks if allocation succeeded
            if alloc_status == "Ok":
                gflops = benchmark_gpu_speed(device)
                print(f"  FP32 Perf (MatMul): {gflops:.2f} GFLOPS")

                transfer_speed = benchmark_transfer_speed(device)
                print(f"  Transfer (GPU<->CPU): {transfer_speed:.2f} GB/s")

    print("\n--- Filesystem Test ---")
    test_filesystem()


if __name__ == "__main__":
    main()
