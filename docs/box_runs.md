# Running on a rented GPU box

The harness runs **on the box**, next to vLLM. Runs on the Mac through the SSH tunnel were limited by the Mac's 16 GB
of RAM: macOS killed the run at 6 workers. Results are copied back to the Mac.

## 1. Rent the box (billing starts)

```bash
python3 scripts/vast.py up --gpu RTX_5090 --max-price 0.8 --yes
python3 scripts/vast.py status        # until vLLM: READY
```

vLLM gets a fixed 10.5 GB KV cache on >= 30 GB cards (`VLLM_KV_CACHE_GB`). That's 303k tokens on a 5090, up from 260k
with `--gpu-memory-utilization 0.92`. After changing it on a running box: `python3 scripts/vast.py restart`.

## 2. Deploy the committed code and the data

```bash
scripts/box_deploy.sh            # pushes `main` over SSH to /root/repo, data to /root/data, builds splits/holdout6
```

The box gets no GitHub credentials, and `.env` is never copied (the vLLM backend needs no key). Commit first:
uncommitted changes are not deployed.

## 3. Run (on the box)

```bash
ssh -p <port> root@<host>        # python3 scripts/vast.py target
cd /root/repo
WORKERS=16 nohup setsid scripts/box_run.sh splits/holdout6 h6-agents-r7 150 > /dev/null 2>&1 < /dev/null &
WORKERS=16 nohup setsid scripts/box_run.sh data/public_test public-agents-p1 1000 > /dev/null 2>&1 < /dev/null &
```

`box_run.sh` resumes the run after crashes or vLLM restarts. Progress is in `result/box_logs/<name>.out`.

**Workers:** each question can send several requests at once (sample voting, checks). On one 5090:
- 32 workers filled the KV cache: 98% used, requests queued, 133 preemptions in 5 minutes.
- 16 workers was the starting point after that.

Check the load with:
```bash
curl -s 127.0.0.1:8000/metrics | grep -E "^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total)\{"
```
If `num_requests_waiting` stays above 0 or preemptions keep rising, lower `WORKERS`.

**Stopping:** stop both the loop and the run, or the loop restarts the run once vLLM answers again:
`pkill -f "[b]ox_run.sh"; pkill -f "[d]atamate run"`. The `[x]` keeps the pattern from matching the SSH command itself.

**Memory (CPU RAM):** the container has ~31 GB. The harness runs the coder's pandas code in-process, and one runaway
merge got it OOM-killed (twice alone, once with the vLLM engine, server down and 500s). The harness now caps its address
space (`DATAMATE_MEM_LIMIT_GB`, default 16, Linux). Check `grep oom_kill /sys/fs/cgroup/memory.events` on the box: a rising
count means a kill. Throughput with 16 workers on one 5090: ~28 s per question (dev300: 300 questions in ~2.3 h).

**Stale state:** `python3 scripts/vast.py up` refuses when `.vast_instance.json` exists, even if the instance is gone
(`vastai show instances` is the truth); move the file away first. `box_deploy.sh <branch>` takes the branch to deploy.

## 4. Copy results back and score (on the Mac)

```bash
scripts/box_sync.sh &            # result/runs/ and result/box_logs/ every 2 min
python3 scripts/subsplit.py ids data/train splits/holdout6 docs/holdout6_ids.txt    # once
.venv/bin/python scripts/holdout_report.py splits/holdout6 h6-agents-r7 h6-agents-r6 --detail
```

The public-test submission is `result/runs/<datetime>_public-agents-p1/submission.jsonl`. Upload it on the Zalo AI
Challenge portal (5 a day).

## 5. Stop billing

```bash
python3 scripts/vast.py down --yes   # destroys the instance and its disk
```
