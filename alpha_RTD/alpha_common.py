# alpha_common.py
# Shared core for Alpha RTD v9 (device, data, Env, simulate_wave)

import os
print("[file]", __file__)

# Force-request MPS inside this process unless caller already set a device.
if os.environ.get("ALPHA_RTD_DEVICE", "").strip() == "":
    os.environ["ALPHA_RTD_DEVICE"] = "mps"

# Optional hard requirement: set ALPHA_RTD_REQUIRE_MPS=1 to abort if MPS isn't selected.
REQUIRE_MPS = os.environ.get("ALPHA_RTD_REQUIRE_MPS", "0") == "1"

import sys
import json, math, random, copy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F  # not heavily used here but kept

print("[py]", sys.executable)
print("[in-script] mps:", hasattr(torch.backends, "mps"),
      "available:", torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else None)

import platform
print("[arch]", platform.machine(), platform.processor())
print("[torch]", torch.__version__, "cuda_ver=", getattr(torch.version, "cuda", None))
if hasattr(torch.backends, "mps"):
    print("[mps] built=", torch.backends.mps.is_built(), "available=", torch.backends.mps.is_available())

# force line-buffered stdout so prints appear immediately
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# --- Prefer GPU on NVIDIA (CUDA) or Apple Silicon (MPS), else CPU ---
# Allow CPU fallback for ops not yet implemented on MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _get_best_device():
    """Pick device with optional override.
    Override examples:
      ALPHA_RTD_DEVICE=mps   # force MPS if available
      ALPHA_RTD_DEVICE=cuda  # force CUDA if available
      ALPHA_RTD_DEVICE=cpu   # force CPU
    """
    override = os.environ.get("ALPHA_RTD_DEVICE", "").strip().lower()
    if override:
        if override == "cuda" and torch.cuda.is_available():
            print("[device override] using CUDA by request")
            return torch.device("cuda")
        if override == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[device override] using MPS by request")
            return torch.device("mps")
        if override == "cpu":
            print("[device override] using CPU by request")
            return torch.device("cpu")
        print(f"[device override] '{override}' requested but not available; falling back automatically")

    if torch.cuda.is_available():
        print("[device select] CUDA available -> cuda")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("[device select] MPS available -> mps")
        return torch.device("mps")
    print("[device select] neither CUDA nor MPS available -> cpu")
    return torch.device("cpu")


def _print_device_info(device):
    has_mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    print(f"[device] {device}")
    print(f"[accel] cuda={torch.cuda.is_available()} mps={has_mps}")


def _fast_mode():
    """Return True if ALPHA_RTD_FAST=1 is set."""
    return os.environ.get("ALPHA_RTD_FAST", "") == "1"


# =========================
# 데이터 로딩 (CSV/JSON)
# =========================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("ALPHA_RTD_DATA_DIR", THIS_DIR)
print(f"[data] BASE={BASE}")

UNITS: Dict[str, pd.DataFrame] = {}


def _load_units():
    U = {}

    def read(name, tier):
        p = os.path.join(BASE, name)
        if os.path.exists(p):
            df = pd.read_csv(p)
            U[tier] = df
        else:
            print(f"[warn] missing {p}; tier '{tier}' empty.")

    read("units_normals.csv", "normal")
    read("units_rares.csv", "rare")
    read("units_epics.csv", "epic")
    read("units_legends.csv", "legend")
    read("units_gods.csv", "god")
    return U


def _safe_read_csv(path):
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(ap)
    print(f"[data] load csv: {ap}")
    return pd.read_csv(ap)


def _safe_read_json(path):
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(ap)
    print(f"[data] load json: {ap}")
    with open(ap, "r", encoding="utf-8") as f:
        return json.load(f)


try:
    print("[data] cwd =", os.getcwd())
    print("[data] listing @BASE:", [n for n in os.listdir(BASE) if n.endswith((".csv", ".json"))])

    UNITS = _load_units()
    ROUNDS = _safe_read_csv(os.path.join(BASE, "rounds_1_50.csv"))
    CONFIG = _safe_read_json(os.path.join(BASE, "config.json"))

    PATH_DF = _safe_read_csv(os.path.join(BASE, "mob_path_waypoints_v2.csv")).sort_values(by="step")
    GRID_DF = _safe_read_csv(os.path.join(BASE, "grid with lane and slot.csv"))

    SLOTS_DF = GRID_DF[GRID_DF["type"] == "SLOT"].copy()
    SLOTS_DF["slot_id"] = range(1, len(SLOTS_DF) + 1)
    SLOTS_DF["max_stack"] = 1
    SLOTS = SLOTS_DF

except Exception as e:
    import traceback

    print(f"[fatal] data load error: {e}")
    print("[data] cwd =", os.getcwd())
    required = [
        "config.json",
        "rounds_1_50.csv",
        "mob_path_waypoints_v2.csv",
        "grid with lane and slot.csv",
        "units_normals.csv",
        "units_rares.csv",
        "units_epics.csv",
        "units_legends.csv",
        "units_gods.csv",
    ]
    for fname in required:
        ap = os.path.abspath(os.path.join(BASE, fname))
        print("  -", fname, "->", ap, ("OK" if os.path.exists(ap) else "MISSING"))
    traceback.print_exc()
    sys.exit(1)


def _seg_len(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


PL = [(float(x), float(y)) for x, y in zip(PATH_DF["x"], PATH_DF["y"])]
SEG = [_seg_len(PL[i], PL[i + 1]) for i in range(len(PL) - 1)]
CUM = [0.0]
for d in SEG:
    CUM.append(CUM[-1] + d)
L_TOTAL = CUM[-1]


def pos_on_path(dist: float) -> Tuple[float, float]:
    if dist <= 0:
        return PL[0]
    if dist >= L_TOTAL:
        return PL[-1]
    for i in range(len(SEG)):
        if CUM[i] <= dist <= CUM[i + 1]:
            a = PL[i]
            b = PL[i + 1]
            sd = SEG[i]
            t = (dist - CUM[i]) / sd if sd > 0 else 0.0
            return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)
    return PL[-1]


def dps_of(row, levels: Dict[str, int]) -> float:
    base = (row.damage * row.shots) / max(1e-6, row.weapon_period)
    bonus = float(row.approx_bonus_dps) if "approx_bonus_dps" in row and not pd.isna(row.approx_bonus_dps) else 0.0
    races = str(row.races_applied).split(";") if "races_applied" in row else []
    mult = 1.0
    up = CONFIG["economy"]["upgrade"]
    for r in races:
        r = r.strip()
        if not r:
            continue
        mult *= up["multiplier"] ** int(levels.get(r, 0))
    return (base + bonus) * mult


@dataclass
class Tower:
    tier: str
    name: str
    race_tag: str
    slot_id: int
    rng: float
    dps: float


@dataclass
class State:
    round_idx: int = 1
    time_sec: float = 0.0
    minerals: int = CONFIG["economy"]["start_minerals"]
    gas: int = CONFIG["economy"]["start_gas"]
    terrazine: int = 0
    levels: Dict[str, int] = field(default_factory=lambda: {"T": 0, "P": 0, "Z": 0})
    inv: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: {k: {} for k in ["normal", "rare", "epic", "legend", "god"]}
    )
    placed: Dict[int, Dict[Tuple[str, str], int]] = field(default_factory=dict)  # slot_id -> {(tier,name): count}
    missions_done: Set[str] = field(default_factory=set)
    pm_next_ready: Dict[str, float] = field(default_factory=lambda: {"pm1": 0.0, "pm2": 0.0, "pm3": 0.0})
    god_moved: Set[Tuple[str, int]] = field(default_factory=set)


class Env:
    def __init__(self, seed=7):
        self.rng = random.Random(seed)
        self.np = np.random.default_rng(seed)
        balance_cfg = CONFIG.get("balance", {})
        self.RANGE_SCALE = float(balance_cfg.get("range_sc2_to_tiles", 1.0))
        self.SPEED_SCALE = float(balance_cfg.get("speed_scale", 1.0))
        self.reset()

    def reset(self):
        self.s = State()
        return self.s

    def snapshot(self):
        return copy.deepcopy(self.s)

    def restore(self, state_snapshot: State):
        self.s = state_snapshot

    def draw_normal(self):
        assert self.s.minerals >= CONFIG["economy"]["draw_cost"]
        self.s.minerals -= CONFIG["economy"]["draw_cost"]
        if "normal" not in UNITS or UNITS["normal"].empty:
            return None
        pool = UNITS["normal"]["name"].tolist()
        name = self.rng.choice(pool)
        self.s.inv["normal"][name] = self.s.inv["normal"].get(name, 0) + 1
        self._auto_combine("normal", "rare")
        self._check_missions()
        return name

    def upgrade(self, race_tag):
        lv = self.s.levels[race_tag]
        cost = CONFIG["economy"]["upgrade"]["base"] + CONFIG["economy"]["upgrade"]["step"] * lv
        assert self.s.gas >= cost
        self.s.gas -= cost
        self.s.levels[race_tag] += 1
        return self.s.levels[race_tag]

    def sell_100m_for_gas(self):
        assert self.s.minerals >= 100
        self.s.minerals -= 100
        g = self.rng.randint(
            CONFIG["economy"]["sell_100m_for_gas"]["min"], CONFIG["economy"]["sell_100m_for_gas"]["max"]
        )
        self.s.gas += g
        return g

    def terra_to_minerals(self, count: int = 1):
        assert self.s.terrazine >= count
        self.s.terrazine -= count
        self.s.minerals += CONFIG["economy"]["terrazine"]["exchange"]["to_minerals"] * count
        return self.s.minerals

    def terra_to_gas(self, count: int = 1):
        assert self.s.terrazine >= count
        self.s.terrazine -= count
        self.s.gas += CONFIG["economy"]["terrazine"]["exchange"]["to_gas"] * count
        return self.s.gas

    def terra_buy_epic(self):
        costT = CONFIG["economy"]["terrazine"]["buy_epic"]["terrazine"]
        costM = CONFIG["economy"]["terrazine"]["buy_epic"]["minerals"]
        assert self.s.terrazine >= costT and self.s.minerals >= costM
        self.s.terrazine -= costT
        self.s.minerals -= costM
        if "epic" not in UNITS or UNITS["epic"].empty:
            return None
        pool = UNITS["epic"]["name"].tolist()
        name = self.rng.choice(pool)
        self.s.inv["epic"][name] = self.s.inv["epic"].get(name, 0) + 1
        self._auto_combine("epic", "legend")
        self._check_missions()
        return name

    def place_unit(self, tier: str, name: str, slot_id: int, count: int = 1):
        assert tier in self.s.inv and self.s.inv[tier].get(name, 0) >= count, "no such unit in inventory"
        max_stack = int(SLOTS.loc[SLOTS["slot_id"] == slot_id, "max_stack"].iloc[0])
        cur = sum(self.s.placed.get(slot_id, {}).values())
        assert cur + count <= max_stack, "slot capacity exceeded"
        self.s.inv[tier][name] -= count
        if self.s.inv[tier][name] <= 0:
            del self.s.inv[tier][name]
        slotmap = self.s.placed.setdefault(slot_id, {})
        slotmap[(tier, name)] = slotmap.get((tier, name), 0) + count
        return True

    def _auto_combine(self, from_t, to_t):
        if "combine_rules" not in CONFIG:
            return
        need = next((r["need"] for r in CONFIG["combine_rules"] if r["from"] == from_t), None)
        if need is None:
            return
        if to_t not in UNITS or UNITS[to_t].empty:
            return
        while True:
            changed = False
            for name, count in list(self.s.inv[from_t].items()):
                while count >= need:
                    self.s.inv[from_t][name] -= need
                    count -= need
                    pool = UNITS[to_t]["name"].tolist()
                    pick = self.rng.choice(pool)
                    self.s.inv[to_t][pick] = self.s.inv[to_t].get(pick, 0) + 1
                    changed = True
                    self._check_missions()
            if not changed:
                break

    def _check_missions(self):
        missions = CONFIG.get("missions", [])
        for m in missions:
            mid = m["id"]
            if mid in self.s.missions_done:
                continue
            try:
                if mid == "normal_8":
                    if "normal" in UNITS and "normal" in self.s.inv:
                        names = UNITS["normal"]["name"].tolist()
                        if all(self.s.inv["normal"].get(n, 0) > 0 for n in names):
                            self.s.missions_done.add(mid)
                            self.s.minerals += int(m.get("reward_minerals", 0))
                elif mid == "rare_pair":
                    if "rare" in self.s.inv:
                        have = set(self.s.inv["rare"].keys())
                        pairs = m.get("pairs", [])
                        if any(p[0] in have and p[1] in have for p in pairs):
                            self.s.missions_done.add(mid)
                            self.s.minerals += int(m.get("reward_minerals", 0))
                elif mid == "rare_8":
                    if "rare" in UNITS and "rare" in self.s.inv:
                        names = UNITS["rare"]["name"].tolist()
                        if all(self.s.inv["rare"].get(n, 0) > 0 for n in names):
                            self.s.missions_done.add(mid)
                            self.s.minerals += int(m.get("reward_minerals", 0))
            except Exception:
                pass

    def _make_towers(self, include_virtual: bool = False):
        towers: List[Tower] = []
        used_slots = set()
        for slot, slotmap in self.s.placed.items():
            used_slots.add(slot)
            for (tier, name), cnt in slotmap.items():
                if tier not in UNITS:
                    continue
                df = UNITS[tier].set_index("name")
                if name not in df.index:
                    continue
                row = df.loc[name]
                for _ in range(cnt):
                    dps_val = dps_of(row, self.s.levels)
                    rng = float(row.range) * self.RANGE_SCALE
                    towers.append(
                        Tower(
                            tier=tier,
                            name=name,
                            race_tag=getattr(row, "primary_race", "T"),
                            slot_id=slot,
                            rng=rng,
                            dps=dps_val,
                        )
                    )

        # capacity bookkeeping for optional virtual fill
        slots_free = [sid for sid in SLOTS["slot_id"].tolist()]
        cap = {int(r.slot_id): int(r.max_stack) for _, r in SLOTS.iterrows()}
        for sid in used_slots:
            c = sum(self.s.placed.get(sid, {}).values())
            cap[sid] = max(0, cap[sid] - c)

        # (optional) Virtually place leftover inventory ONLY when explicitly requested
        if include_virtual:
            for tier in ["god", "legend", "epic", "rare", "normal"]:
                if tier not in UNITS:
                    continue
                df = UNITS[tier].set_index("name")
                for name, count in self.s.inv.get(tier, {}).items():
                    for _ in range(count):
                        dst = None
                        for sid in slots_free:
                            if cap.get(sid, 0) > 0:
                                dst = sid
                                break
                        if dst is None:
                            break
                        cap[dst] -= 1
                        row = df.loc[name]
                        dps_val = dps_of(row, self.s.levels)
                        rng = float(row.range) * self.RANGE_SCALE
                        towers.append(
                            Tower(
                                tier=tier,
                                name=name,
                                race_tag=getattr(row, "primary_race", "T"),
                                slot_id=dst,
                                rng=rng,
                                dps=dps_val,
                            )
                        )
        return towers

    def start_round(self):
        r = int(self.s.round_idx)
        if r > 50:
            return True, 0.0
        row = ROUNDS.loc[ROUNDS["round"] == r].iloc[0]
        sp = float(row.speed) * self.SPEED_SCALE
        creeps = int(row.creeps)
        hp = float(row.hp)
        sh = float(row.shield)
        gap = float(row.spawn_interval_sec)
        vuln = float(row.vuln_mult) if "vuln_mult" in row.index and not pd.isna(row.vuln_mult) else 1.0
        towers = self._make_towers(include_virtual=False)
        if pd.isna(creeps) or pd.isna(hp):
            return False, None
        clear_t = simulate_wave(towers, creeps, hp, sh, sp, gap, t0=self.s.time_sec, vuln_mult=vuln)
        if clear_t is None:
            return False, None
        pause = (
            float(row.pause_after_sec)
            if ("pause_after_sec" in row.index and not pd.isna(row.pause_after_sec))
            else 0.0
        )
        self.s.time_sec = clear_t + pause
        self.s.minerals += CONFIG["economy"]["round_clear_minerals"]
        is_boss = ("boss" in row.index) and (not pd.isna(row.boss)) and (int(row.boss) == 1)
        if is_boss and r < 50:
            self.s.terrazine = min(
                self.s.terrazine + CONFIG["economy"]["terrazine"]["boss_kills_grant"],
                CONFIG["economy"]["terrazine"]["max_boss_terra"],
            )
        self.s.round_idx += 1
        return True, clear_t

    def trigger_pm(self, pm_id: str):
        pm_cfg = CONFIG.get("personal_missions", None)
        if not pm_cfg:
            return False, None
        pm = pm_cfg.get(pm_id)
        if not pm:
            return False, None
        if self.s.round_idx < pm.get("unlock_round", 1):
            return False, None
        if self.s.time_sec < self.s.pm_next_ready.get(pm_id, 0.0):
            return False, None
        towers = self._make_towers(include_virtual=False)
        sp = float(pm["speed"]) * self.SPEED_SCALE
        clear_t = simulate_wave(towers, 1, float(pm["hp"]), 0.0, sp, 0.5, t0=self.s.time_sec, vuln_mult=1.0)
        if clear_t is None:
            return False, None
        self.s.time_sec = clear_t
        self.s.minerals += int(pm["reward_minerals"])
        cd = float(pm_cfg["cooldown_sec"])
        self.s.pm_next_ready[pm_id] = self.s.time_sec + cd
        return True, clear_t


def simulate_wave(
    towers: List[Tower],
    n_creeps: int,
    hp: float,
    sh: float,
    speed: float,
    gap: float,
    t0: float = 0.0,
    dt: float = 0.02,
    tmax: float = 1800.0,
    vuln_mult: float = 1.0,
):
    class C:
        __slots__ = ("spawn", "hp", "sh", "dist", "alive")

        def __init__(self, spawn):
            self.spawn = spawn
            self.hp = hp
            self.sh = sh
            self.dist = 0.0
            self.alive = True

    creeps = [C(t0 + i * gap) for i in range(n_creeps)]
    t = t0
    last_kill = t0
    slot_xy = {int(r.slot_id): (float(r.x), float(r.y)) for _, r in SLOTS.iterrows()}

    def pos(dist: float):
        return pos_on_path(dist)

    while t <= t0 + tmax:
        n_alive = 0
        for c in creeps:
            if not c.alive:
                continue
            if t >= c.spawn:
                c.dist += speed * dt
                if c.dist >= L_TOTAL:
                    return None
            if c.hp > 0:
                n_alive += 1
        if n_alive == 0:
            return last_kill

        for tw in towers:
            sx, sy = slot_xy[int(tw.slot_id)]
            target = None
            bestd = -1.0
            for i, c in enumerate(creeps):
                if not c.alive or t < c.spawn:
                    continue
                x, y = pos(c.dist)
                if (x - sx) ** 2 + (y - sy) ** 2 <= tw.rng ** 2:
                    if c.dist > bestd:
                        bestd = c.dist
                        target = i
            if target is not None:
                c = creeps[target]
                dmg = tw.dps * dt * vuln_mult
                if c.sh > 0:
                    ds = min(c.sh, dmg)
                    c.sh -= ds
                    dmg -= ds
                if dmg > 0:
                    c.hp -= dmg
                if c.hp <= 0 and c.alive:
                    c.alive = False
                    last_kill = t
        t += dt
    return None