"""Bridge to the PYTHIA 8 worker that runs inside WSL.

The worker (``hep/worker.py``) owns all research-grade physics: PYTHIA 8.3 hard
processes, QED/QCD showers, multiparton interactions, Lund string
hadronization, and hadron decays with physical decay vertices. This module
starts it through ``wsl.exe``, exchanges JSON lines on a background thread so
the window never blocks, and turns replies into :class:`PythiaEvent` objects.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import threading
from dataclasses import dataclass, field

from smlab.lorentz import FourVector

WSL_DISTRO = os.environ.get("SMLAB_WSL_DISTRO", "Ubuntu")
ENV_PYTHON = os.environ.get("SMLAB_HEP_PYTHON", "~/micromamba/envs/smlab-hep/bin/python")
_HERE = os.path.dirname(os.path.abspath(__file__))


def _project_root() -> str:
    # A PyInstaller one-file build unpacks hep/ into sys._MEIPASS, which WSL can
    # read through /mnt/<drive>/ like any other Windows folder.
    if getattr(sys, "frozen", False):
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    return os.path.dirname(_HERE)


def worker_path_in_wsl() -> str:
    """Translate the Windows path of hep/worker.py to its /mnt/<drive>/ form."""
    path = os.path.join(_project_root(), "hep", "worker.py")
    drive, rest = os.path.splitdrive(os.path.abspath(path))
    if not drive:
        return path
    return "/mnt/" + drive[0].lower() + rest.replace("\\", "/")


# PYTHIA status codes (Pythia 8 manual, "Particle Properties").
def is_final(status: int) -> bool:
    return status > 0


@dataclass(slots=True)
class PythiaParticle:
    index: int
    pdg: int
    status: int
    mother1: int
    mother2: int
    daughter1: int
    daughter2: int
    p4: FourVector
    mass: float
    vertex_mm: tuple[float, float, float]
    tau_mm: float
    charge: float

    @property
    def final(self) -> bool:
        return self.status > 0


@dataclass(slots=True)
class PythiaEvent:
    """One PYTHIA event, with the same core attributes the display expects."""

    seed: int
    sqrt_s: float
    beam_id: str
    process_id: str
    process_title: str
    particles: list[PythiaParticle]
    hard: list[tuple]
    jets: list[FourVector]
    jet_multiplicity: list[int]
    code: int
    name: str
    sqrt_s_hat: float
    sigma_pb: float
    sigma_err_pb: float
    engine: str = "pythia"
    reco: dict | None = None  # Delphes reconstruction, when the detector simulation is on
    weight: float = 1.0  # signed event weight (MC@NLO events can be negative)
    cos_theta: float = 0.0
    phi: float = 0.0
    isr_v: float = 0.0
    _finals: list[PythiaParticle] = field(default_factory=list)

    def finals(self) -> list[PythiaParticle]:
        if not self._finals:
            self._finals = [p for p in self.particles if p.final]
        return self._finals

    def beams(self) -> list[PythiaParticle]:
        return [p for p in self.particles if abs(p.status) == 12]

    def isr_photon(self):
        return None

    def daughters(self, particle: PythiaParticle) -> list[PythiaParticle]:
        if particle.daughter1 <= 0:
            return []
        last = max(particle.daughter1, particle.daughter2)
        return [self.particles[i] for i in range(particle.daughter1, last + 1) if 0 <= i < len(self.particles)]

    def decay_vertex_mm(self, particle: PythiaParticle) -> tuple[float, float, float] | None:
        kids = self.daughters(particle)
        if not kids:
            return None
        return kids[0].vertex_mm


@dataclass(frozen=True, slots=True)
class PythiaConservation:
    delta_e: float
    delta_p: float
    delta_charge_thirds: int
    missing_px: float
    missing_py: float
    delta_baryon_thirds: int = 0
    delta_lepton: tuple[int, int, int] = (0, 0, 0)
    checks_flavor: bool = False

    @property
    def ok(self) -> bool:
        return abs(self.delta_e) < 1.0e-3 and self.delta_p < 1.0e-3 and self.delta_charge_thirds == 0


def conservation(event: PythiaEvent) -> PythiaConservation:
    """Energy, momentum, and charge of the final state against the two beams.

    PYTHIA keeps kinematics in single-precision-safe doubles; the tolerance is
    1 MeV on the sum of hundreds of particles.
    """
    beams = event.beams()
    total_in = FourVector(0.0, 0.0, 0.0, 0.0)
    charge_in = 0.0
    for beam in beams:
        total_in = total_in + beam.p4
        charge_in += beam.charge
    total_out = FourVector(0.0, 0.0, 0.0, 0.0)
    charge_out = 0.0
    visible_px = 0.0
    visible_py = 0.0
    for particle in event.finals():
        total_out = total_out + particle.p4
        charge_out += particle.charge
        if abs(particle.pdg) not in (12, 14, 16):
            visible_px += particle.p4.px
            visible_py += particle.p4.py
    delta = total_out - total_in
    return PythiaConservation(
        delta_e=delta.e,
        delta_p=math.sqrt(delta.px**2 + delta.py**2 + delta.pz**2),
        delta_charge_thirds=int(round(3.0 * (charge_out - charge_in))),
        missing_px=total_in.px - visible_px,
        missing_py=total_in.py - visible_py,
    )


def event_from_reply(raw: dict, *, seed: int, sqrt_s: float, beam_id: str, process_id: str, title: str,
                     sigma_pb: float, sigma_err_pb: float) -> PythiaEvent:
    particles = []
    for index, row in enumerate(raw["particles"]):
        (pdg, status, m1, m2, d1, d2, px, py, pz, e, m, x, y, z, tau, charge) = row
        particles.append(
            PythiaParticle(index, int(pdg), int(status), int(m1), int(m2), int(d1), int(d2),
                           FourVector(e, px, py, pz), m, (x, y, z), tau, charge)
        )
    jets = [FourVector(j[3], j[0], j[1], j[2]) for j in raw.get("jets", [])]
    shat = raw.get("sqrt_s_hat") or sqrt_s
    # The primary angle: the first outgoing hard-process particle against the +z beam.
    cos_theta = 0.0
    phi = 0.0
    for row in raw.get("hard", []):
        pdg, status = row[0], row[1]
        if status == 23 or status == 22:
            px, py, pz = row[4], row[5], row[6]
            p = math.sqrt(px * px + py * py + pz * pz)
            if p > 0.0:
                cos_theta = pz / p
                phi = math.atan2(py, px)
            break
    return PythiaEvent(
        seed=seed,
        sqrt_s=sqrt_s,
        beam_id=beam_id,
        process_id=process_id,
        process_title=title,
        particles=particles,
        hard=[tuple(row) for row in raw.get("hard", [])],
        jets=jets,
        jet_multiplicity=[int(j[4]) for j in raw.get("jets", [])],
        code=int(raw.get("code", 0)),
        name=str(raw.get("name", "")),
        sqrt_s_hat=float(shat),
        sigma_pb=sigma_pb,
        sigma_err_pb=sigma_err_pb,
        cos_theta=cos_theta,
        phi=phi,
        reco=raw.get("reco"),
    )


def _children_die_with_us() -> bool:
    """Put SMLab in a Windows job object that kills its members when SMLab exits.

    Windows does not end child processes with their parent. Processes started
    after this call (wsl.exe and the helpers it spawns) inherit the job, so a
    crashed or force-closed SMLab cannot leave the WSL worker running: when the
    last handle to the job closes, every member is terminated and the worker
    sees end-of-input. Returns True when the job is in place.
    """
    if sys.platform != "win32" or _JOBS:
        return bool(_JOBS)
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class IO_COUNTERS(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BASIC_LIMIT(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class EXTENDED_LIMIT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BASIC_LIMIT),
            ("IoInfo", IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = EXTENDED_LIMIT()
    info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        kernel32.CloseHandle(job)
        return False
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        kernel32.CloseHandle(job)
        return False
    _JOBS.append(job)  # kept open until SMLab exits
    return True


_JOBS: list = []


class PythiaEngine:
    """Owns the WSL worker process. All calls return immediately; results arrive via ``poll``."""

    def __init__(self) -> None:
        self.state = "offline"  # offline, starting, ready, busy, error
        self.message = ""
        self.version = ""
        self.catalog: dict[str, dict] = {}
        self._proc: subprocess.Popen | None = None
        self._jobs: queue.Queue = queue.Queue()
        self._results: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._configured: dict | None = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self.state = "starting"
        self.message = "Starting PYTHIA in WSL…"
        self._thread = threading.Thread(target=self._run, name="pythia-bridge", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        """Ask the worker to quit, then make sure the wsl.exe relay is gone."""
        if self._thread is not None and self._thread.is_alive():
            self._jobs.put(("quit", None))
            self._thread.join(timeout)
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except OSError:
                pass

    def _spawn(self) -> subprocess.Popen:
        try:
            _children_die_with_us()
        except Exception:  # cleanup is a convenience; never block the engine
            pass
        command = f"exec {ENV_PYTHON} -u {worker_path_in_wsl()}"
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        proc = subprocess.Popen(
            ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", command],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
            creationflags=flags,
        )
        return proc

    def _call(self, request: dict) -> dict:
        assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
        self._proc.stdin.write(json.dumps(request) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("the PYTHIA worker stopped unexpectedly")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "PYTHIA worker error"))
        return reply

    def _run(self) -> None:
        try:
            self._proc = self._spawn()
            hello = self._call({"cmd": "hello"})
        except Exception as exc:
            self.state = "error"
            self.message = (
                "PYTHIA is not available. It runs in WSL (Ubuntu) from ~/micromamba/envs/smlab-hep; "
                f"run hep/setup_wsl.sh once. Details: {exc}"
            )
            self._results.put(("error", self.message))
            return
        self.version = hello.get("pythia", "")
        self.catalog = hello.get("processes", {})
        self.state = "ready"
        self.message = f"PYTHIA {self.version} ready"
        self._results.put(("ready", hello))
        while True:
            kind, payload = self._jobs.get()
            if kind == "quit":
                try:
                    self._call({"cmd": "quit"})
                except Exception:
                    pass
                return
            try:
                self.state = "busy"
                if kind == "run":
                    config, n = payload
                    if config != self._configured:
                        self.message = "Initializing PYTHIA…"
                        self._call({"cmd": "init", "config": config})
                        self._configured = dict(config)
                    self.message = f"Generating {n} event{'s' if n != 1 else ''}…"
                    reply = self._call({"cmd": "generate", "n": n})
                    self._results.put(("events", (config, reply)))
                self.state = "ready"
                self.message = f"PYTHIA {self.version} ready"
            except Exception as exc:
                self._configured = None
                self.state = "ready" if self._proc and self._proc.poll() is None else "error"
                self.message = str(exc)
                self._results.put(("error", str(exc)))
                if self.state == "error":
                    return

    # -- requests ------------------------------------------------------------

    def request_events(self, config: dict, n: int = 1) -> None:
        """Queue ``n`` events for ``config``. PYTHIA is re-initialized only when the config changes."""
        self._jobs.put(("run", (dict(config), int(n))))

    @property
    def pending(self) -> int:
        return self._jobs.qsize()

    def poll(self) -> list[tuple[str, object]]:
        out = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out
