"""Run head_mean_row.py (the same code as head_mean_row.ipynb) on a Modal CPU container.

  modal run run_head_mean_row_modal.py
Outputs land in ./outputs/: summary JSON, figure, and the printed log.
"""
import os

import modal

HERE = os.path.dirname(os.path.abspath(__file__))
app = modal.App("head-mean-row")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("numpy", "requests", "matplotlib")
         .add_local_file(os.path.join(HERE, "head_mean_row.py"), "/root/head_mean_row.py"))


@app.function(image=image, cpu=8, memory=16384, timeout=3600)
def run():
    import contextlib, io, runpy
    os.environ["HEAD_OUTDIR"] = "/tmp/heads"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path("/root/head_mean_row.py", run_name="__main__")
    out = {"log.txt": buf.getvalue().encode()}
    for f in ("head_mean_row_summary.json", "head_mean_row.png"):
        out[f] = open(f"/tmp/heads/{f}", "rb").read()
    return out


@app.local_entrypoint()
def main():
    res = run.remote()
    od = os.path.join(HERE, "outputs")
    os.makedirs(od, exist_ok=True)
    for name, data in res.items():
        open(os.path.join(od, name), "wb").write(data)
    print(res["log.txt"].decode())
