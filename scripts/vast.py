"""Rent a vast.ai GPU box, serve Qwen3.5-9B (bf16) with vLLM on it, and reach it through an SSH tunnel.

usage:
  python scripts/vast.py offers  [--gpu RTX_5090] [--n 1]           cheapest matching offers (free, read-only)
  python scripts/vast.py up      [--gpu RTX_5090] [--n 1] [--offer ID] [--max-price 1.0] --yes
                                                                    rent the box and start vLLM (billing starts)
  python scripts/vast.py status                                      instance state, $/h, vLLM health, last log lines
  python scripts/vast.py tunnel                                      forward localhost:8000 -> vLLM (keep it open)
  python scripts/vast.py restart                                     relaunch vLLM with current settings (no re-rent)
  python scripts/vast.py logs [-n 50]                                tail of the vLLM log on the box
  python scripts/vast.py target                                      print "user@host port" for ssh / rsync
  python scripts/vast.py down --yes                                  destroy the instance (billing stops; disk is deleted)

Then, in another terminal:
  VLLM_BASE_URL=http://localhost:8000/v1 uv run datamate run --backend vllm --split-dir <split> --workers 24

Design:
- vLLM listens on 127.0.0.1 inside the box: nothing is exposed publicly, the SSH tunnel is the only way in.
- n GPUs are served as ONE endpoint, so the client needs one URL: data parallel (one model copy per GPU) on >= 32 GB
  cards, tensor parallel (the model split across GPUs) on 24 GB cards such as 2x RTX_3090.
- bf16 weights are 19.3 GB; a single 24 GB card is refused (too little KV cache). RTX 5090 (the organisers' card) by default.
- The vLLM image is pinned (VLLM_IMAGE) so reruns use the same inference stack.
- KV cache: on >= 30 GB cards vLLM gets a fixed KV cache (VLLM_KV_CACHE_GB, default 10.5 GB) instead of
  --gpu-memory-utilization 0.92, which left 8.4 GB on a 5090 (260k tokens; 303k with 10.5 GB). The agents harness sends
  several requests per question, so a full cache meant preemptions (recomputed work) at 32 workers.
- State (instance id) is kept in .vast_instance.json at the repo root.

Running ON the box (no tunnel, best throughput): see docs/box_runs.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = Path(os.getenv("VAST_STATE") or ROOT / ".vast_instance.json")  # VAST_STATE=<file> manages a second box
IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:v0.31.0")
MODEL = os.getenv("VLLM_MODEL", "Qwen/Qwen3.5-9B")
PORT = int(os.getenv("VLLM_PORT", "8000"))
MAX_LEN = int(os.getenv("MAX_MODEL_LEN", "65536"))  # agent histories passed 32k on the 5090 test (400 errors -> abstain)
DISK_GB = int(os.getenv("VAST_DISK_GB", "80"))      # image (~12 GB) + weights (~19 GB) + headroom


def vast(*args: str, raw: bool = True):
    cmd = ["vastai", *args] + (["--raw"] if raw else [])
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"vastai {' '.join(args)} failed:\n{out.stderr or out.stdout}")
    return json.loads(out.stdout) if raw else out.stdout


def query(gpu: str, n: int, max_price: float | None) -> str:
    q = (f"gpu_name={gpu} num_gpus={n} rentable=true reliability>0.97 cuda_max_good>=12.8 "
         f"disk_space>={DISK_GB} inet_down>=300 direct_port_count>=1")
    return q + (f" dph_total<={max_price}" if max_price else "")


def offers(a) -> list[dict]:
    rows = vast("search", "offers", query(a.gpu, a.n, a.max_price), "-o", "dph_total")
    for o in rows[:10]:
        print(f"{o['id']:>10}  ${o['dph_total']:.3f}/h  {o['num_gpus']}x {o['gpu_name']} {o['gpu_ram'] / 1024:.0f}GB  "
              f"rel={o['reliability2']:.3f}  down={int(o['inet_down'])}Mb/s  cuda={o.get('cuda_max_good')}  "
              f"verified={o.get('verified')}  {o.get('geolocation')}")
    if not rows:
        print("no matching offers; try --gpu RTX_4090/L40S/A100_PCIE or raise --max-price")
    return rows


def parallel_flag(n: int, gpu_ram_gb: float) -> list[str]:
    """>= 32 GB cards hold a full bf16 copy each: one replica per GPU (data parallel, best throughput).
    Smaller cards (24 GB 3090/4090) leave ~1 GB of KV cache next to the 19.3 GB of weights, so the model is split
    across the GPUs instead (tensor parallel): ~9.7 GB of weights and ~11 GB of KV cache per card."""
    if n == 1 or gpu_ram_gb >= 30:
        return ["--data-parallel-size", str(n)]
    return ["--tensor-parallel-size", str(n)]


def memory_flag(gpu_ram_gb: float) -> list[str]:
    kv = os.getenv("VLLM_KV_CACHE_GB", "10.5" if gpu_ram_gb >= 30 else "")
    if kv:
        return ["--kv-cache-memory", str(int(float(kv) * 1e9))]
    return ["--gpu-memory-utilization", "0.92"]


def onstart(n: int, gpu_ram_gb: float = 32) -> str:
    serve = [
        "vllm", "serve", MODEL, "--host", "127.0.0.1", "--port", str(PORT), "--dtype", "bfloat16",
        "--max-model-len", str(MAX_LEN), *memory_flag(gpu_ram_gb), "--enable-prefix-caching",
        *parallel_flag(n, gpu_ram_gb), "--seed", "0", "--reasoning-parser", "qwen3",
    ]
    env = f"VLLM_BATCH_INVARIANT={os.getenv('VLLM_BATCH_INVARIANT', '0')}"
    return f"bash -c {shlex.quote(f'{env} nohup {shlex.join(serve)} > /root/vllm.log 2>&1 &')}"


def up(a) -> None:
    if STATE.exists():
        sys.exit(f"an instance is already recorded in {STATE.name}: run `status` or `down --yes` first")
    if a.offer:
        # `search offers id=N` returns [] for every id, so find the offer in the filtered listing instead
        offer = next((o for o in vast("search", "offers", query(a.gpu, a.n, a.max_price), "-o", "dph_total")
                      if o["id"] == a.offer), None)
        if offer is None:
            sys.exit(f"offer {a.offer} is no longer available")
    else:
        rows = offers(a)
        if not rows:
            sys.exit(1)
        offer = rows[0]
    ram = offer["gpu_ram"] / 1024
    if offer["num_gpus"] == 1 and ram < 30:
        sys.exit(f"1x {offer['gpu_name']} ({ram:.0f} GB) cannot hold bf16 Qwen3.5-9B (19.3 GB) plus a useful KV cache; "
                 "use --n 2 (tensor parallel) or a >= 32 GB card")
    print(f"\nselected offer {offer['id']}: {offer['num_gpus']}x {offer['gpu_name']} at ${offer['dph_total']:.3f}/h "
          f"({offer.get('geolocation')}), image {IMAGE}, disk {DISK_GB} GB")
    if not a.yes:
        sys.exit("dry run: add --yes to rent it (billing starts when the instance is created)")
    args = ["create", "instance", str(offer["id"]), "--image", IMAGE, "--disk", str(DISK_GB), "--ssh", "--direct",
            "--label", "datamate-vllm", "--onstart-cmd", onstart(int(offer["num_gpus"]), ram)]
    if os.getenv("HF_TOKEN"):  # optional: avoids anonymous Hugging Face rate limits on the weight download
        args += ["--env", f"-e HF_TOKEN={os.environ['HF_TOKEN']}"]
    res = vast(*args)
    iid = res.get("new_contract")
    if not iid:
        sys.exit(f"create failed: {res}")
    STATE.write_text(json.dumps({"id": iid, "offer": offer["id"], "dph": offer["dph_total"], "gpus": offer["num_gpus"],
                                 "gpu": offer["gpu_name"], "gpu_ram_gb": ram, "image": IMAGE, "model": MODEL, "created": time.time()}, indent=1))
    print(f"instance {iid} created; waiting for it to boot (image pull + weight download take ~5-15 min)")
    wait_running(iid)
    print("next: `python scripts/vast.py tunnel` in one terminal, then `python scripts/vast.py status` until vLLM is ready")


def instance(iid: int) -> dict:
    return vast("show", "instance", str(iid))


def wait_running(iid: int, timeout: int = 1200) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = instance(iid).get("actual_status")
        print(f"  {int(time.time() - t0):>4}s  status={st}")
        if st == "running":
            return
        time.sleep(20)
    print("still not running; check `status` (some hosts are slow to pull the image)")


def ssh_target(iid: int) -> tuple[str, str]:
    url = vast("ssh-url", str(iid), raw=False).strip()  # ssh://root@host:port
    host, port = url.removeprefix("ssh://").rsplit(":", 1)
    return host, port


def ssh(iid: int, remote: str, capture: bool = True) -> str:
    host, port = ssh_target(iid)
    cmd = ["ssh", "-p", port, "-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15", host, remote]
    return subprocess.run(cmd, capture_output=capture, text=True).stdout if capture else str(subprocess.call(cmd))


def state() -> dict:
    if not STATE.exists():
        sys.exit("no instance recorded; run `up` first")
    return json.loads(STATE.read_text())


def status(a) -> None:
    s = state()
    inst = instance(s["id"])
    hours = (time.time() - s["created"]) / 3600
    print(f"instance {s['id']}: {inst.get('actual_status')}  {s['gpus']}x {s['gpu']}  ${inst.get('dph_total', s['dph']):.3f}/h  "
          f"up {hours:.2f} h  ~${hours * inst.get('dph_total', s['dph']):.2f} so far")
    if inst.get("actual_status") == "running":
        code = ssh(s["id"], f"curl -s -o /dev/null -w '%{{http_code}}' localhost:{PORT}/v1/models; echo; tail -n 3 /root/vllm.log")
        lines = code.splitlines()
        print("vLLM:", "READY" if lines and lines[0] == "200" else "not ready yet", "\n" + "\n".join(lines[1:]))


def tunnel(a) -> None:
    s = state()
    host, port = ssh_target(s["id"])
    print(f"forwarding localhost:{PORT} -> {host}:{PORT} (Ctrl-C to stop); use VLLM_BASE_URL=http://localhost:{PORT}/v1")
    sys.exit(subprocess.call(["ssh", "-p", port, "-N", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30",
                              "-L", f"{PORT}:127.0.0.1:{PORT}", host]))


def restart(a) -> None:
    """Relaunch vLLM on the rented box with the current settings (e.g. a new MAX_MODEL_LEN); weights stay cached on disk."""
    s = state()
    inner = onstart(int(s["gpus"]), float(s.get("gpu_ram_gb", 32))).removeprefix("bash -c ")
    # two calls: a pkill pattern found anywhere in the same remote command line would kill that command itself
    ssh(s["id"], "pkill -f '[v]llm serve'; sleep 5")
    ssh(s["id"], f"bash -c {inner}")
    print("vLLM relaunched; run `status` until READY (~1-3 min)")


def logs(a) -> None:
    print(ssh(state()["id"], f"tail -n {a.lines} /root/vllm.log"))


def down(a) -> None:
    s = state()
    if not a.yes:
        sys.exit(f"this destroys instance {s['id']} (its disk too); add --yes")
    vast("destroy", "instance", str(s["id"]), "-y", raw=False)  # -y: without it the CLI prompts and silently aborts
    for _ in range(12):  # trust the instance list, not the command's exit code
        if not any(i["id"] == s["id"] for i in vast("show", "instances")):
            break
        time.sleep(10)
    else:
        sys.exit(f"instance {s['id']} is STILL listed after destroy: it may keep billing. Check `vastai show instances`.")
    hours = (time.time() - s["created"]) / 3600
    print(f"destroyed {s['id']} after {hours:.2f} h (~${hours * s['dph']:.2f}); confirmed gone")
    STATE.unlink()


def main() -> None:
    p = argparse.ArgumentParser(prog="vast.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("offers", offers), ("up", up)):
        sp = sub.add_parser(name)
        sp.add_argument("--gpu", default="RTX_5090")
        sp.add_argument("--n", type=int, default=1, help="GPUs on the box (served as one endpoint)")
        sp.add_argument("--max-price", type=float, help="$/h ceiling")
        if name == "up":
            sp.add_argument("--offer", type=int, help="rent this offer id instead of the cheapest match")
            sp.add_argument("--yes", action="store_true")
        sp.set_defaults(fn=fn)
    sub.add_parser("status").set_defaults(fn=status)
    sub.add_parser("tunnel").set_defaults(fn=tunnel)
    sub.add_parser("target").set_defaults(fn=lambda a: print(*ssh_target(state()["id"])))
    sub.add_parser("restart").set_defaults(fn=restart)
    lp = sub.add_parser("logs")
    lp.add_argument("-n", "--lines", type=int, default=50)
    lp.set_defaults(fn=logs)
    dp = sub.add_parser("down")
    dp.add_argument("--yes", action="store_true")
    dp.set_defaults(fn=down)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
