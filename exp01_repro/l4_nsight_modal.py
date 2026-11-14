"""Nsight Systems timeline of the certified step, and Nsight Compute on its kernels, on an L4.

  modal run l4_nsight_modal.py --model qwen3_8b
Saves profiles/<model>.nsys-rep, profiles/<model>_int4.ncu-rep (if the GPU counters are available) and
text summaries profiles/<model>_nsys.txt, profiles/<model>_ncu.txt.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("exp01-nsight")
vol = modal.Volume.from_name("exp01-repro")
VOL = "/vol"
REPO = "https://developer.download.nvidia.com/devtools/repos/ubuntu2204/amd64"
image = (modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
         .apt_install("wget", "gnupg", "ca-certificates")
         .run_commands(f"wget -qO /usr/share/keyrings/nv-devtools.asc {REPO}/nvidia.pub",
                       f"echo 'deb [signed-by=/usr/share/keyrings/nv-devtools.asc] {REPO}/ /' > /etc/apt/sources.list.d/nv-devtools.list",
                       "apt-get update && apt-get install -y nsight-systems-cli")
         .pip_install("torch", "safetensors", "huggingface_hub", "numpy")
         .env({"HF_HOME": f"{VOL}/hf"}))
for _f in ("cert_core.py", "a3_lib.py", "a3_modal.py", "subg_lib.py", "l4_kernels.py", "l4_cert.py", "l4_bench_modal.py"):
    image = image.add_local_file(os.path.join(HERE, _f), f"/root/{_f}")

DRIVER = r'''
import sys, torch, l4_cert
from a3_modal import MODELS, _load_head
from l4_bench_modal import T_STAR
model, n = sys.argv[1], int(sys.argv[2])
Wb = _load_head(MODELS[model]).cuda()
head = l4_cert.CertHead(Wb, T_STAR[model])
H = torch.load(f"/vol/states/{model}/h_greedy.pt")[:64].cuda()
for i in range(20):                                   # warm-up and Triton autotune, outside the capture range
    head.step(H[i:i + 1]); torch.argmax(H[i:i + 1] @ Wb.T, 1)
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
for i in range(20, 20 + n):
    torch.cuda.nvtx.range_push("bf16_head"); torch.argmax(H[i:i + 1] @ Wb.T, 1).item(); torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("certified_step"); head.step(H[i:i + 1])[0].item(); torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
'''


@app.function(image=image, volumes={VOL: vol}, gpu="L4", cpu=8, memory=64 * 1024, timeout=3600)
def run(model: str, n: int = 10):
    import glob
    import subprocess
    open("/root/driver.py", "w").write(DRIVER)
    out = {}

    def sh(cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd="/root")
        return r.returncode, (r.stdout + r.stderr)[-20000:]

    rc, log = sh(f"nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --trace=cuda,nvtx,osrt "
                 f"--force-overwrite=true -o /tmp/{model} python driver.py {model} {n}")
    out["nsys_log"] = log
    if os.path.exists(f"/tmp/{model}.nsys-rep"):
        out["nsys_rep"] = open(f"/tmp/{model}.nsys-rep", "rb").read()
        _, out["nsys_stats"] = sh(f"nsys stats --report nvtx_sum,cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum "
                                  f"--format column /tmp/{model}.nsys-rep")
    ncu = glob.glob("/usr/local/cuda*/bin/ncu") + glob.glob("/opt/nvidia/nsight-compute/*/ncu")
    if ncu:
        rc, log = sh(f"{ncu[0]} --target-processes all --kernel-name regex:'_int4_head|_gather_gemv|gemv|gemm|sm80_xmma|cutlass' "
                     f"--clock-control none --launch-skip 200 --launch-count 6 --set full --force-overwrite -o /tmp/{model}_ncu python driver.py {model} 3")
        out["ncu_log"] = log
        if os.path.exists(f"/tmp/{model}_ncu.ncu-rep"):
            out["ncu_rep"] = open(f"/tmp/{model}_ncu.ncu-rep", "rb").read()
            _, out["ncu_details"] = sh(f"{ncu[0]} --import /tmp/{model}_ncu.ncu-rep --page details "
                                       f"--section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats --section Occupancy")
    else:
        out["ncu_log"] = "ncu not found in the image"
    return out


@app.local_entrypoint()
def main(model: str = "qwen3_8b", n: int = 10):
    pdir = os.path.join(HERE, "profiles")
    os.makedirs(pdir, exist_ok=True)
    o = run.remote(model, n)
    if "nsys_rep" in o:
        open(f"{pdir}/{model}.nsys-rep", "wb").write(o["nsys_rep"])
    if "ncu_rep" in o:
        open(f"{pdir}/{model}_ncu.ncu-rep", "wb").write(o["ncu_rep"])
    open(f"{pdir}/{model}_nsys.txt", "w").write(o.get("nsys_stats", "") + "\n\n--- nsys log\n" + o["nsys_log"])
    open(f"{pdir}/{model}_ncu.txt", "w").write(o.get("ncu_details", "") + "\n\n--- ncu log\n" + o["ncu_log"])
    print(sorted(os.listdir(pdir)))
