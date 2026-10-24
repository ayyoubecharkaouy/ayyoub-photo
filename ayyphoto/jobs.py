"""Lancer un calcul local sans bloquer l'interface Streamlit."""

from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys

from .config import EXPERIMENTS, PROJECT_ROOT, read_json, write_json


ACTIVE_JOB = EXPERIMENTS / "active_job.json"
MODULES = {"train": "ayyphoto.train", "audit": "ayyphoto.audit", "data": "ayyphoto.download"}


def is_running(job):
    if not job or not job.get("pid"):
        return False
    try:
        pid = int(job["pid"])
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ")
        return (str(job["config_path"]).encode() in command
                and MODULES.get(job.get("kind"), "invalid-job-module").encode() in command)
    except (OSError, ValueError, TypeError):
        return False


def active_job():
    return read_json(ACTIVE_JOB)


def start_job(kind, config):
    if kind not in MODULES:
        raise ValueError("Type de calcul inconnu.")
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    lock_path = EXPERIMENTS / ".launch.lock"
    lock = lock_path.open("a")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Un lancement est déjà en cours ; réessayer dans un instant.") from exc
        if is_running(active_job()):
            raise ValueError("Un calcul local est déjà en cours. L’attendre ou l’arrêter avant d’en lancer un autre.")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        directory = EXPERIMENTS / f"{kind}_{stamp}"
        directory.mkdir(exist_ok=False)
        config_path = directory / "job_config.json"
        write_json(config_path, config)
        write_json(directory / "status.json", {"state": "starting", "message": "Préparation du calcul…"})
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", OMP_NUM_THREADS=str(config.get("threads", 2)),
                           MKL_NUM_THREADS=str(config.get("threads", 2)))
        with (directory / "run.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", MODULES[kind], "--config", str(config_path)],
                cwd=PROJECT_ROOT, env=environment, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        job = {"kind": kind, "pid": process.pid, "run_dir": str(directory), "config_path": str(config_path)}
        write_json(ACTIVE_JOB, job)
        return job
    finally:
        lock.close()  # Le système libère aussi ce verrou si le serveur s'arrête.


def stop_job(job):
    if not is_running(job):
        return False
    os.kill(int(job["pid"]), signal.SIGINT)
    return True


def tail_log(directory, limit=10000):
    try:
        with (Path(directory) / "run.log").open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
