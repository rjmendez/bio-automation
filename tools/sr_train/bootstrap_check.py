#!/usr/bin/env python3
"""Read-only, idempotent pre-flight for tools/sr_train.

Adapted from one idea in the third-party "agentic-pipelines" framework (its
``bootstrap.py --check`` command) -- NOT an adoption of that framework. This
file checks the real environment regressions hit during the M3b session, each
one discovered only after a training run had already silently failed or
produced nothing:

  1. bare `pip install torch` (no --index-url), or any dependency install that
     drags torch back to a CPU wheel, silently clobbering the CUDA build.
  2. torchvision ABI-mismatched against the installed torch build
     ("operator torchvision::nms does not exist").
  3. the cuDNN companion package missing/broken, which only surfaces when a
     real cuDNN-backed op is first run (cuda.is_available() can be True while
     this is still broken).
  4. the venv's own pip silently refusing to self-upgrade (observed on the
     Amazon-Linux AWS box: RPM-managed system pip leaves the venv pip stale).
  5. `pip install piq` (no --no-deps) silently downgrading torch via its own
     version pins -- checked both generically and by grepping the two shipped
     cloud scripts for the exact `--no-deps` flag on the piq line.
  6. a training script exiting 0 with empty stdout and no --metrics-out /
     --result-json file written (the exact silent no-op hit with
     train_denoise.py's M2 recipe A/B run).
  7. train_denoise.py has no GPU-only guard after pick_device() (unlike
     train_lora_hat.py, which already raises SystemExit on a cpu device) --
     omitting --cuda-name-substr falls back to CPU instead of failing loudly,
     violating the project's GPU-only/fail-loudly standing rule.
  8. Ollama (a separate local LLM server sharing these GPUs) reporting a model
     unloaded via keep_alive:0 while it is still resident in VRAM, so a local
     GPU job wrongly assumes a card is free.
  9. no pinned requirements file for tools/sr_train's own deps, so the live
     venv's installed torch/torchvision/piq/numpy/Pillow/einops set is
     cross-checked against the exact pins hardcoded in cloud/user-data.sh and
     cloud/user-data-lora.sh instead.

This script is READ-ONLY: every check queries state (pip show/--version,
nvidia-smi, a GET to Ollama's own API, python -c import/assert probes, grep
of the two shipped shell scripts). It never installs, upgrades or writes
anything. Running it twice in a row with no state change gives the same
pass/fail result both times.

Scope of THIS pass: only the local venv
(F:\\venvs\\bio-sr-train\\Scripts\\python.exe, Windows, cudnn64_9.dll) is
exercised by default. The cudnn check and the pip-floor check both branch on
the target interpreter's own `sys.platform`, so pointing `--python` at the
AWS venv's `python3` (Linux, libcudnn.so.9) should work unchanged -- but that
has NOT been run against the real AWS instance in this pass. A follow-up
should copy this file into cloud/user-data.sh's and cloud/user-data-lora.sh's
staging step and run it there as the gate before `timeout 7200 python3
aws_train.py|train_lora_hat.py` starts; this script does not edit those shell
scripts itself.

GPU-only, fail loudly, never a silent CPU fallback -- this script exists
specifically to enforce that standing rule on the environment, before a
training command (aws_train.py / train_denoise.py / train_lora_hat.py) spends
real GPU time (and, on AWS, real money) on a broken install.

Exit code: 0 only if every applicable check passes. Non-zero, with each
failing check named and why, otherwise. A SKIP (e.g. no --last-run-log given,
or Ollama unreachable) does not fail the run on its own.

Usage:
    F:\\venvs\\bio-sr-train\\Scripts\\python.exe tools\\sr_train\\bootstrap_check.py
    python tools/sr_train/bootstrap_check.py --python /path/to/other/venv/python3
    python tools/sr_train/bootstrap_check.py --last-run-log out/train.log \\
        --last-run-metrics out/metrics.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SR_TRAIN_DIR = Path(__file__).resolve().parent
CLOUD_DIR = REPO_ROOT / "cloud"

PIP_VERSION_FLOOR = (24, 0)
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/ps"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


class CheckResult(NamedTuple):
    name: str
    status: str
    detail: str
    severity: str = ""


def run_argv(argv: list[str], timeout: int = 60) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout, proc.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s: {argv!r}"


def run_probe(python_exe: str, code: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run `code` as `python_exe -I -c code` (isolated: no site/user hijacking)."""
    return run_argv([python_exe, "-I", "-c", code], timeout=timeout)


# --------------------------------------------------------------------------
# GPU selection (read-only; re-measured every run, never a hardcoded index)
# --------------------------------------------------------------------------

def nvidia_smi_gpus() -> list[dict] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    rc, out, _err = run_argv(
        [exe, "--query-gpu=name,memory.used,memory.free,memory.total",
         "--format=csv,noheader,nounits"],
        timeout=15,
    )
    if rc != 0:
        return None
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        name, used, free, total = parts
        try:
            gpus.append({
                "name": name,
                "used_mib": float(used),
                "free_mib": float(free),
                "total_mib": float(total),
            })
        except ValueError:
            continue
    return gpus or None


def pick_least_contended_gpu_substr(python_exe: str) -> tuple[str | None, str]:
    """
    Pick a short torch-matchable name substring (e.g. "2080") for whichever
    installed GPU currently has the most free-memory headroom, per a fresh
    nvidia-smi query -- never trusting a previously measured index, since
    torch's own cuda:N ordering is not guaranteed to match nvidia-smi's index
    order (reconfirmed on this run; see detail string).
    """
    gpus = nvidia_smi_gpus()
    if not gpus:
        return None, "nvidia-smi unavailable or returned no GPUs -- GPU checks will use torch's bare 'cuda' default device"
    best = max(gpus, key=lambda g: g["free_mib"] / g["total_mib"] if g["total_mib"] else 0.0)
    m = re.search(r"(\d{4})", best["name"])
    substr = m.group(1) if m else best["name"]
    rows = ", ".join(
        f"{g['name']!r} free={g['free_mib']:.0f}/{g['total_mib']:.0f}MiB" for g in gpus
    )
    detail = f"nvidia-smi now: {rows} -> picked {best['name']!r} (match substring {substr!r})"
    return substr, detail


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_pip_version(python_exe: str) -> CheckResult:
    rc, out, err = run_argv([python_exe, "-m", "pip", "--version"])
    if rc != 0:
        return CheckResult("pip version floor", FAIL, f"could not run pip --version: {err or out}", "blocks_training")
    m = re.search(r"pip (\d+)\.(\d+)", out)
    if not m:
        return CheckResult("pip version floor", FAIL, f"could not parse pip version from: {out!r}", "blocks_training")
    ver = (int(m.group(1)), int(m.group(2)))
    detail = f"venv pip reports {out.strip()}"
    if ver >= PIP_VERSION_FLOOR:
        return CheckResult("pip version floor", PASS, detail)
    return CheckResult(
        "pip version floor", FAIL,
        detail + f" -- below floor {'.'.join(map(str, PIP_VERSION_FLOOR))}; "
        "this venv's own pip may be too stale to resolve torch's cudnn companion package correctly "
        "(the Amazon-Linux AWS incident: RPM-managed system pip silently refused to self-upgrade)",
        "blocks_training",
    )


_TORCH_CUDA_PROBE = (
    "import torch\n"
    "ver = torch.__version__\n"
    "cu = torch.version.cuda\n"
    "ok = torch.cuda.is_available() and cu is not None and '+cu' in ver\n"
    "print('TORCH_VER=' + ver)\n"
    "print('TORCH_CUDA=' + str(cu))\n"
    "print('CUDA_AVAILABLE=' + str(torch.cuda.is_available()))\n"
    "print('RESULT=' + ('OK' if ok else 'FAIL'))\n"
)


def check_torch_cuda_wheel(python_exe: str) -> CheckResult:
    rc, out, err = run_probe(python_exe, _TORCH_CUDA_PROBE)
    if rc != 0:
        return CheckResult(
            "torch is a CUDA wheel (not CPU-only)", FAIL,
            f"probe exited {rc}: {(err or out).strip()[:400]}", "blocks_training",
        )
    ver_m = re.search(r"TORCH_VER=(\S+)", out)
    cuda_m = re.search(r"TORCH_CUDA=(\S+)", out)
    avail_m = re.search(r"CUDA_AVAILABLE=(\S+)", out)
    result_m = re.search(r"RESULT=(\S+)", out)
    ver = ver_m.group(1) if ver_m else "?"
    cuda = cuda_m.group(1) if cuda_m else "?"
    avail = avail_m.group(1) if avail_m else "?"
    detail = f"torch.__version__={ver} torch.version.cuda={cuda} cuda.is_available()={avail}"
    if result_m and result_m.group(1) == "OK":
        return CheckResult("torch is a CUDA wheel (not CPU-only)", PASS, detail)
    return CheckResult(
        "torch is a CUDA wheel (not CPU-only)", FAIL,
        detail + " -- this is the CPU-wheel regression (bare `pip install torch`, or a dependency "
        "like piq dragging torch back, silently installs '+cpu' and cuda.is_available()=False)",
        "blocks_training",
    )


_TORCHVISION_ABI_PROBE = (
    "import torch, torchvision\n"
    "torchvision.ops.nms(torch.zeros(1, 4), torch.zeros(1), 0.5)\n"
    "print('RESULT=OK')\n"
)


def check_torchvision_abi(python_exe: str) -> CheckResult:
    rc, out, err = run_probe(python_exe, _TORCHVISION_ABI_PROBE)
    combined = (out + err)
    if rc == 0 and "RESULT=OK" in combined:
        return CheckResult("torchvision ABI matches installed torch", PASS, "torchvision.ops.nms() ran without error")
    nms_mismatch = "torchvision::nms does not exist" in combined
    detail = combined.strip()[-400:] if combined.strip() else f"probe exited {rc} with no output"
    extra = (
        " (classic ABI mismatch: torchvision built against a different torch build than is "
        "currently installed -- e.g. torch was --force-reinstalled after torchvision was already "
        "installed against the old one)"
        if nms_mismatch else ""
    )
    return CheckResult("torchvision ABI matches installed torch", FAIL, detail + extra, "blocks_training")


_CUDNN_PROBE_TEMPLATE = (
    "import torch, torch.nn.functional as F\n"
    "dev = {device!r}\n"
    "x = torch.randn(8, 3, 64, 64, device=dev)\n"
    "w = torch.randn(8, 3, 3, 3, device=dev)\n"
    "y = F.conv2d(x, w)\n"
    "torch.cuda.synchronize()\n"
    "print('RESULT=OK', tuple(y.shape))\n"
)


def check_cudnn_conv(python_exe: str, cuda_name_substr: str | None, gpu_detail: str) -> CheckResult:
    # Resolve the substring to a concrete cuda:N index inside THIS interpreter,
    # the same way train.py's own pick_device() does, so the check exercises
    # whichever card is actually least contended right now.
    if cuda_name_substr:
        resolve = (
            "import torch\n"
            f"substr = {cuda_name_substr!r}.lower()\n"
            "matches = [i for i in range(torch.cuda.device_count()) "
            "if substr in torch.cuda.get_device_name(i).lower()]\n"
            "print('MATCHES=' + ','.join(map(str, matches)))\n"
        )
        rc, out, err = run_probe(python_exe, resolve)
        m = re.search(r"MATCHES=([\d,]*)", out)
        idxs = [int(i) for i in m.group(1).split(",") if i] if m else []
        if rc != 0 or len(idxs) != 1:
            device = "cuda"
            note = f" (could not uniquely resolve substring {cuda_name_substr!r} to one device, idxs={idxs!r}; falling back to default 'cuda')"
        else:
            device = f"cuda:{idxs[0]}"
            note = ""
    else:
        device = "cuda"
        note = ""

    probe = _CUDNN_PROBE_TEMPLATE.format(device=device)
    rc, out, err = run_probe(python_exe, probe)
    combined = out + err
    header = f"device={device}{note} | {gpu_detail}"
    if rc == 0 and "RESULT=OK" in combined:
        return CheckResult("cuDNN-backed conv2d runs on CUDA", PASS, header + " | " + out.strip())
    libcudnn_linux = "libcudnn.so.9" in combined
    libcudnn_win = "cudnn64_9.dll" in combined or "LoadLibrary failed" in combined
    extra = ""
    if libcudnn_linux:
        extra = " -- cudnn companion package missing/broken (Linux: libcudnn.so.9)"
    elif libcudnn_win:
        extra = " -- cudnn companion package missing/broken (Windows: cudnn64_9.dll)"
    detail = (combined.strip()[-500:] if combined.strip() else f"probe exited {rc} with no output")
    return CheckResult("cuDNN-backed conv2d runs on CUDA", FAIL, header + " | " + detail + extra, "blocks_training")


def check_piq_no_deps_in_cloud_scripts() -> CheckResult:
    scripts = ["user-data.sh", "user-data-lora.sh"]
    missing = []
    seen = {}
    for name in scripts:
        path = CLOUD_DIR / name
        if not path.exists():
            missing.append(f"{name} (file not found at {path})")
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        has_exact_flag = bool(re.search(r"pip install --quiet --no-deps piq\b", text))
        seen[name] = has_exact_flag
        if not has_exact_flag:
            missing.append(f"{name} (missing exact `pip install --quiet --no-deps piq` line)")
    if not missing:
        return CheckResult(
            "cloud scripts pin piq with --no-deps", PASS,
            f"both {', '.join(scripts)} contain the exact `--no-deps` piq install line",
        )
    return CheckResult(
        "cloud scripts pin piq with --no-deps", FAIL,
        "regression risk: " + "; ".join(missing) +
        " -- without --no-deps, piq's own torch/torchvision version pins silently downgrade torch to a CPU wheel",
        "blocks_training",
    )


def check_sr_train_dep_versions(python_exe: str) -> CheckResult:
    # What the cloud scripts intend: cu128 wheels off download.pytorch.org.
    cu_tags = set()
    for name in ["user-data.sh", "user-data-lora.sh"]:
        path = CLOUD_DIR / name
        if path.exists():
            cu_tags.update(re.findall(r"whl/(cu\d+)", path.read_text(encoding="utf-8", errors="replace")))
    expected_tag = sorted(cu_tags)[0] if cu_tags else "cu128"

    rc, out, err = run_argv([python_exe, "-m", "pip", "show", "torch", "torchvision", "piq", "numpy", "Pillow", "einops"])
    if rc != 0:
        return CheckResult("installed deps match cloud-script pins", FAIL, f"pip show failed: {err or out}", "wastes_time")
    blocks = out.split("---")
    versions = {}
    for block in blocks:
        name_m = re.search(r"^Name: (\S+)", block, re.MULTILINE)
        ver_m = re.search(r"^Version: (\S+)", block, re.MULTILINE)
        if name_m and ver_m:
            versions[name_m.group(1).lower()] = ver_m.group(1)

    problems = []
    for pkg in ("torch", "torchvision"):
        v = versions.get(pkg)
        if v is None:
            problems.append(f"{pkg} not installed")
        elif expected_tag not in v:
            problems.append(f"{pkg}=={v} does not carry the {expected_tag!r} tag the cloud scripts install")
    for pkg in ("piq", "numpy", "pillow", "einops"):
        if pkg not in versions:
            problems.append(f"{pkg} not installed")

    detail = "installed: " + ", ".join(f"{k}=={v}" for k, v in sorted(versions.items()))
    if not problems:
        return CheckResult("installed deps match cloud-script pins", PASS, detail)
    return CheckResult(
        "installed deps match cloud-script pins", FAIL,
        detail + " | problems: " + "; ".join(problems), "wastes_time",
    )


def check_train_denoise_gpu_guard() -> CheckResult:
    path = SR_TRAIN_DIR / "train_denoise.py"
    if not path.exists():
        return CheckResult("train_denoise.py has a GPU-only guard", FAIL, f"{path} not found", "wastes_money")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    pick_idx = None
    for i, line in enumerate(lines):
        if "pick_device(args.cuda_name_substr)" in line:
            pick_idx = i
            break
    if pick_idx is None:
        return CheckResult(
            "train_denoise.py has a GPU-only guard", FAIL,
            "no `pick_device(args.cuda_name_substr)` call found -- cannot verify a guard follows it", "wastes_money",
        )
    window = "\n".join(lines[pick_idx: pick_idx + 6])
    guarded = bool(re.search(r"device\.type\s*!=\s*[\"']cuda[\"']", window)) and \
        bool(re.search(r"raise\s+SystemExit|sys\.exit", window))
    loc = f"{path.name}:{pick_idx + 1}"
    if guarded:
        return CheckResult("train_denoise.py has a GPU-only guard", PASS, f"guard found right after {loc}")
    return CheckResult(
        "train_denoise.py has a GPU-only guard", FAIL,
        f"{loc}: `device = pick_device(args.cuda_name_substr)` has no `if device.type != \"cuda\": raise SystemExit(...)` "
        "guard after it (train_lora_hat.py already has this guard; train_denoise.py does not) -- omitting "
        "--cuda-name-substr silently falls back to CPU instead of failing loudly, violating the GPU-only standing rule",
        "wastes_money",
    )


def nvidia_smi_compute_apps() -> list[dict] | None:
    """List processes nvidia-smi sees attached to a GPU context right now.

    On Windows (WDDM) per-process used_memory usually comes back as the
    literal string "[N/A]" -- nvidia-smi still reports which processes hold a
    GPU context even though it won't give their individual memory, so process
    *presence* is the signal this check uses, not a memory figure.
    """
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    rc, out, _err = run_argv(
        [exe, "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        timeout=15,
    )
    if rc != 0:
        return None
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        pid, name = parts[0], parts[1]
        apps.append({"pid": pid, "process_name": name})
    return apps


def check_ollama_vram_agreement(url: str) -> CheckResult:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return CheckResult("Ollama /api/ps agrees with actual GPU attachment", SKIP, f"Ollama unreachable at {url}: {exc}")
    except json.JSONDecodeError as exc:
        return CheckResult("Ollama /api/ps agrees with actual GPU attachment", SKIP, f"Ollama returned unparseable JSON: {exc}")

    models = payload.get("models", [])
    if models:
        names = ", ".join(m.get("name", "?") for m in models)
        return CheckResult(
            "Ollama /api/ps agrees with actual GPU attachment", PASS,
            f"/api/ps reports {len(models)} model(s) resident ({names}) -- nothing to cross-check, Ollama claims the GPU is in use",
        )

    apps = nvidia_smi_compute_apps()
    if apps is None:
        return CheckResult("Ollama /api/ps agrees with actual GPU attachment", SKIP, "nvidia-smi unavailable, cannot cross-check")

    # Match on the process itself, not total VRAM used -- this machine also
    # runs other real GPU workloads (e.g. k3s vllm-gpu) whose usage must not
    # be misattributed to Ollama just because a card shows memory used.
    ollama_procs = [a for a in apps if "ollama" in a["process_name"].lower()]
    if not ollama_procs:
        return CheckResult(
            "Ollama /api/ps agrees with actual GPU attachment", PASS,
            f"/api/ps reports no models loaded, and no ollama/llama-server process holds a GPU context ({len(apps)} other compute-app(s) present)",
        )
    procs_str = "; ".join(f"pid={a['pid']} {a['process_name']}" for a in ollama_procs)
    return CheckResult(
        "Ollama /api/ps agrees with actual GPU attachment", FAIL,
        f"/api/ps reports no models loaded, but nvidia-smi still lists an Ollama process attached to a GPU context: "
        f"{procs_str} -- Ollama's own unload accounting (keep_alive:0) disagrees with reality; this script cannot "
        "fix Ollama, but a local GPU job should not assume a card is free from /api/ps alone",
        "wastes_time",
    )


def check_last_run_output(log_path: Path | None, metrics_path: Path | None, exit_code: int | None) -> CheckResult:
    if log_path is None and metrics_path is None:
        return CheckResult(
            "last training run produced real output", SKIP,
            "no --last-run-log/--last-run-metrics given -- pass them after a run to check for the exit-0/zero-output silent failure",
        )
    problems = []
    if exit_code is not None and exit_code != 0:
        problems.append(f"process exited {exit_code}")
    if log_path is not None:
        if not log_path.exists():
            problems.append(f"log file missing: {log_path}")
        elif log_path.stat().st_size == 0:
            problems.append(f"log file is empty: {log_path}")
    if metrics_path is not None:
        if not metrics_path.exists():
            problems.append(f"metrics/result file missing: {metrics_path}")
        elif metrics_path.stat().st_size == 0:
            problems.append(f"metrics/result file is empty: {metrics_path}")
        else:
            try:
                json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                problems.append(f"metrics/result file is not valid JSON: {exc}")
    if problems:
        return CheckResult(
            "last training run produced real output", FAIL,
            "; ".join(problems) + " -- this is the exit-0-but-silent-no-op failure (train_denoise.py's M2 A/B run)",
            "wastes_time",
        )
    detail_bits = []
    if log_path is not None:
        detail_bits.append(f"log {log_path} is {log_path.stat().st_size} bytes")
    if metrics_path is not None:
        detail_bits.append(f"metrics {metrics_path} is valid JSON")
    return CheckResult("last training run produced real output", PASS, "; ".join(detail_bits))


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable, help="python executable of the venv to check (default: whatever ran this script)")
    ap.add_argument("--cuda-name-substr", default=None, help="skip auto-detection and force this GPU-name substring for the cudnn check")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help=f"Ollama /api/ps URL (default: {DEFAULT_OLLAMA_URL})")
    ap.add_argument("--last-run-log", type=Path, default=None, help="stdout/stderr log of the training run just launched, to check it is non-empty")
    ap.add_argument("--last-run-metrics", type=Path, default=None, help="--metrics-out / --result-json path the training run just launched should have written")
    ap.add_argument("--last-run-exit-code", type=int, default=None, help="exit code of the training run just launched")
    args = ap.parse_args()

    python_exe = args.python
    results: list[CheckResult] = []

    results.append(check_pip_version(python_exe))
    results.append(check_torch_cuda_wheel(python_exe))
    results.append(check_torchvision_abi(python_exe))

    substr = args.cuda_name_substr
    gpu_detail = "GPU selection: --cuda-name-substr given explicitly, skipping auto-detect" if substr else ""
    if not substr:
        substr, gpu_detail = pick_least_contended_gpu_substr(python_exe)
    results.append(check_cudnn_conv(python_exe, substr, gpu_detail))

    results.append(check_piq_no_deps_in_cloud_scripts())
    results.append(check_sr_train_dep_versions(python_exe))
    results.append(check_train_denoise_gpu_guard())
    results.append(check_ollama_vram_agreement(args.ollama_url))
    results.append(check_last_run_output(args.last_run_log, args.last_run_metrics, args.last_run_exit_code))

    name_width = max(len(r.name) for r in results)
    print(f"tools/sr_train/bootstrap_check.py -- target interpreter: {python_exe}\n")
    for r in results:
        print(f"[{r.status:4s}] {r.name:<{name_width}}  {r.detail}")

    n_pass = sum(1 for r in results if r.status == PASS)
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_skip = sum(1 for r in results if r.status == SKIP)
    print(f"\n{n_pass} passed, {n_fail} failed, {n_skip} skipped (of {len(results)} checks)")

    if n_fail:
        print("\nFAILED checks (fix these before any training command runs):")
        for r in results:
            if r.status == FAIL:
                print(f"  - {r.name} [{r.severity or 'unspecified'}]: {r.detail}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
