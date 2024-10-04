import subprocess

def get_gpu_usage():
    try:
        # nvidia-smi 명령어 실행
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            encoding='utf-8'
        )
        gpu_usage = gpu_info.strip().split("\n")[0].split(", ")
        gpu_util = gpu_usage[0]  # GPU 사용률 (%)
        mem_util = gpu_usage[1]  # 메모리 사용률 (%)
        total_mem = gpu_usage[2]  # 총 메모리 (MB)
        used_mem = gpu_usage[3]  # 사용된 메모리 (MB)
        free_mem = gpu_usage[4]  # 사용 가능한 메모리 (MB)

        return gpu_util, mem_util, total_mem, used_mem, free_mem
    except Exception as e:
        print(f"Error retrieving GPU usage: {e}")
        return None, None, None, None, None