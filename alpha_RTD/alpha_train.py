# alpha_train.py
# ReplayBuffer, self-play, curriculum, main_improved

import os
import sys
import math
import time
import random
import pickle
from collections import deque
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from alpha_common import (
    _get_best_device,
    _print_device_info,
    _fast_mode,
    REQUIRE_MPS,
    CONFIG,
)
from alpha_env import NLRDT_GymEnv, ImprovedNLRDT_GymEnv
from alpha_model import ImprovedNNet

from alpha_mcts import MCTS

# ---- Save scheduler (time/turn/ep/iter based) ---------------------------------
class SaveScheduler:
    def __init__(
        self,
        out_dir: str,
        keep: int = 10,
        per_ep: bool = True,
        every_ep: int = 0,
        every_iter: int = 0,
        every_turns: int = 0,
        every_seconds: float = 0.0,
        prefix: str = "model_v9_iter_",
    ):
        self.dir = Path(out_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep = int(keep)
        self.per_ep = bool(per_ep)
        self.every_ep = int(every_ep)
        self.every_iter = int(every_iter)
        self.every_turns = int(every_turns)
        self.every_seconds = float(every_seconds)
        self.prefix = prefix

        self._last_save_t = time.time()
        self._turns_since = 0

    def bump_turns(self, n: int = 1):
        self._turns_since += int(n)

    def due(self, ep_idx: Optional[int] = None, iter_idx: Optional[int] = None) -> bool:
        now = time.time()
        by_time = (self.every_seconds > 0.0) and ((now - self._last_save_t) >= self.every_seconds)
        by_turns = (self.every_turns > 0) and (self._turns_since >= self.every_turns)

        by_ep = False
        if self.per_ep and (ep_idx is not None):
            by_ep = True
        if (self.every_ep > 0) and (ep_idx is not None):
            by_ep = by_ep or ((ep_idx + 1) % self.every_ep == 0)

        by_iter = False
        if (self.every_iter > 0) and (iter_idx is not None):
            by_iter = ((iter_idx + 1) % self.every_iter == 0)

        return by_time or by_turns or by_ep or by_iter

    def mark_saved(self):
        self._last_save_t = time.time()
        self._turns_since = 0

    def make_tagged_path(self, iter_idx: int, tag: str) -> Path:
        ts = int(time.time())
        return self.dir / f"{self.prefix}{iter_idx}_{tag}_{ts}.pth"

    def prune(self, pattern: str):
        files = sorted(self.dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files[self.keep:]:
            try:
                f.unlink()
            except Exception:
                pass
# ------------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buf = deque(maxlen=max_size)

    def add(self, batch):
        self.buf.extend(batch)

    def sample(self, n):
        n = min(n, len(self.buf))
        if n == 0:
            return []
        return random.sample(self.buf, n)

    def __len__(self):
        return len(self.buf)


def run_self_play_improved(
    env: ImprovedNLRDT_GymEnv,
    mcts: MCTS,
    rb: ReplayBuffer,
    n_sims_per_move: int,
    device,
    iter_idx: int,
    value_scale: float,
    on_turn: Optional[callable] = None,
):
    print(f"  [self-play] Ep (iter {iter_idx}) start", flush=True)
    game_hist: List[Tuple[np.ndarray, np.ndarray, float]] = []
    obs, _ = env.reset(seed=random.randint(0, 2**32 - 1))
    done = False
    turns = 0
    total_reward = 0.0

    while not done and turns < 250:
        mcts.nn.eval()

        s = env.base_env.env.s
        _tot_dps, _n_towers, _cov = env._summary_stats()
        print(
            f"    [turn {turns+1}] round={s.round_idx} bank={s.minerals}m/{s.gas}g "
            f"towers={int(_n_towers)} dps={_tot_dps:.1f} cov={_cov:.2f}",
            flush=True,
        )

        if turns == 0:
            print(f"    [self-play] running MCTS sims={n_sims_per_move}", flush=True)

        _snap = env.base_env.snapshot()
        pi = mcts.get_action_probs(
            env,
            n_simulations=n_sims_per_move,
            device=device,
            iter_idx=iter_idx,
            episode_idx=iter_idx,
            turn_idx=turns + 1,
        )

        try:
            _obs_now = env._get_obs()
            _mask_now = _obs_now["action_mask"]
            _tot_dps_now, _n_towers_now, _cov_now = env._summary_stats()

            # Heuristic 1: when START_ROUND is not yet available, avoid PASS-spam and
            # spread probability over any other enabled actions (including placements/upgrades).
            if _mask_now[11] == 0:
                pi = pi.copy()
                # temporarily forbid PASS while we are still in "build" phase
                pi[12] = 0.0
                _enabled = np.where((_mask_now == 1) & (np.arange(len(_mask_now)) != 12))[0]
                if len(_enabled) > 0:
                    pi[:] = 0.0
                    pi[_enabled] = 1.0 / len(_enabled)
                else:
                    # if literally nothing else is enabled, fall back to PASS
                    pi[:] = 0.0
                    pi[12] = 1.0

            # Heuristic 2: fusion hint. If any fusion actions are available, gently boost them
            # so that MCTS is more likely to explore actual on-board synthesis.
            fusion_actions: List[int] = []
            try:
                for slot_id in env.strategic_slots:
                    for tier in ["rare", "epic", "legend"]:
                        from_tier = {"rare": "normal", "epic": "rare", "legend": "epic"}[tier]
                        cands = env._has_fusion_candidates_at(slot_id, from_tier)
                        if cands:
                            a_idx = _action_index_for_place(env, tier, slot_id)
                            if a_idx is not None and _mask_now[a_idx] == 1:
                                fusion_actions.append(a_idx)
            except Exception:
                fusion_actions = []

            if fusion_actions:
                pi = pi.copy()
                for a_idx in fusion_actions:
                    pi[a_idx] = pi[a_idx] * 1.5 + 0.5
                ssum = pi.sum()
                if ssum > 0:
                    pi /= ssum
        except Exception:
            pass

        try:
            _obs_now2 = env._get_obs()
            _mask_now2 = _obs_now2["action_mask"]
            if _mask_now2[11] == 1:
                _enabled_non_pass = np.where(
                    (_mask_now2 == 1) & (np.arange(len(_mask_now2)) != 12)
                )[0]
                if len(_enabled_non_pass) == 1 and _enabled_non_pass[0] == 11:
                    pi[:] = 0.0
                    pi[11] = 1.0
        except Exception:
            pass

        try:
            env.base_env.restore(_snap)
            env._cache_fp = None
            env._cache_obs = None
            env._cache_stats = None
        except Exception:
            pass

        topk = np.argsort(pi)[-5:][::-1]

        def _decode_action(idx: int) -> str:
            if idx < 13:
                names = [
                    "DRAW",
                    "UP_T",
                    "UP_P",
                    "UP_Z",
                    "SELL_GAS",
                    "TERRA_M",
                    "TERRA_G",
                    "BUY_EPIC",
                    "PM1",
                    "PM2",
                    "PM3",
                    "START_ROUND",
                    "PASS",
                ]
                return names[idx]
            _idx = idx - 13
            _ti = _idx // env.N_STRATEGIC_SLOTS
            _si = _idx % env.N_STRATEGIC_SLOTS
            return f"PLACE_{env.tier_order[_ti].upper()}_SLOT_{env.strategic_slots[_si]}"

        print(
            "    [pi top5] "
            + ", ".join(f"{_decode_action(j)}:{pi[j]:.2f}" for j in topk),
            flush=True,
        )

        game_hist.append([obs["observation"].copy(), pi.copy(), 0.0])
        a = np.random.choice(len(pi), p=pi)

        if a < 13:
            action_names = [
                "DRAW",
                "UP_T",
                "UP_P",
                "UP_Z",
                "SELL_GAS",
                "TERRA_M",
                "TERRA_G",
                "BUY_EPIC",
                "PM1",
                "PM2",
                "PM3",
                "START_ROUND",
                "PASS",
            ]
            act_name = action_names[a]
        else:
            _idx = a - 13
            _ti = _idx // env.N_STRATEGIC_SLOTS
            _si = _idx % env.N_STRATEGIC_SLOTS
            act_name = f"PLACE_{env.tier_order[_ti].upper()}_SLOT_{env.strategic_slots[_si]}"

        print(f"    [act] {act_name} (a={a})", flush=True)
        obs, r, done, _, _ = env.step(a)
        print(
            f"    [step] reward={r:+.2f} done={done} next_round={env.base_env.env.s.round_idx}",
            flush=True,
        )
        total_reward += r
        turns += 1
        if on_turn is not None:
            try:
                on_turn()
            except Exception:
                pass

        if a == 11:
            print(
                f"    t={turns:3d} action=START_ROUND reward={r:+.2f} "
                f"round={env.base_env.env.s.round_idx}",
                flush=True,
            )

    final_round = env.base_env.env.s.round_idx
    score = total_reward + (final_round - 1) * 2.0
    z = math.tanh(score / value_scale)

    for i in range(len(game_hist)):
        game_hist[i][2] = z

    print(
        f"  [self-play] end: turns={turns}, round={final_round}, "
        f"score={score:.2f} (z={z:.4f})"
    )
    return game_hist, score, final_round, turns


def train_nn(nn: ImprovedNNet, rb: ReplayBuffer, opt, device, batch_size=64):
    if len(rb) < batch_size:
        return None
    nn.train()

    batch = rb.sample(batch_size)
    s, target_pi, target_v = zip(*batch)
    states = torch.FloatTensor(np.array(s)).to(device)
    target_pi = torch.FloatTensor(np.array(target_pi)).to(device)
    target_v = torch.FloatTensor(np.array(target_v)).unsqueeze(1).to(device)

    logits, pred_v = nn(states)
    logp = torch.nn.functional.log_softmax(logits, dim=1)
    loss_policy = -(target_pi * logp).sum(dim=1).mean()
    loss_value = torch.nn.functional.mse_loss(pred_v, target_v)
    loss = loss_policy + loss_value

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(nn.parameters(), 1.0)
    opt.step()
    return float(loss.item())


def _one_hot(size: int, idx: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _action_index_for_place(env: ImprovedNLRDT_GymEnv, tier: str, slot_id: int) -> Optional[int]:
    try:
        ti = env.tier_order.index(tier)
        si = env.strategic_slots.index(int(slot_id))
    except ValueError:
        return None
    return 13 + ti * env.N_STRATEGIC_SLOTS + si


def _script_round1_episode(
    env: ImprovedNLRDT_GymEnv,
    n_place: int = 5,
    value_scale: float = 100.0,
    action_size: Optional[int] = None,
    seed: Optional[int] = None,
):
    """
    Simple scripted episode for curriculum:
    - Reset env
    - Place up to `n_place` normal-tier towers on the first strategic slots (field-only)
    - Start round 1
    This uses only ImprovedNLRDT_GymEnv placement semantics (no inventory / DRAW action 0).
    """
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    obs, _ = env.reset(seed=seed)

    if action_size is None:
        action_size = env.N_TOTAL_ACTIONS

    hist: List[Tuple[np.ndarray, np.ndarray, float]] = []
    total_reward = 0.0
    turns = 0

    # How many normal placements can we afford?
    s = env.base_env.env.s
    draw_cost = env.base_env.CONFIG["economy"]["draw_cost"]
    max_affordable = int(s.minerals // draw_cost)
    n_place_eff = n_place
    if n_place_eff > 0:
        n_place_eff = min(n_place_eff, max_affordable, len(env.strategic_slots))

    # 1) Place normals on the earliest strategic slots
    placed = 0
    for slot_id in env.strategic_slots:
        if n_place_eff > 0 and placed >= n_place_eff:
            break
        a = _action_index_for_place(env, "normal", slot_id)
        if a is None:
            continue
        hist.append([obs["observation"].copy(), _one_hot(action_size, a), 0.0])
        obs, r, done, _, _ = env.step(a)
        total_reward += r
        turns += 1
        placed += 1
        if done:
            break

    # 2) Start round 1 if we are still alive
    if not done:
        a = 11  # START_ROUND
        hist.append([obs["observation"].copy(), _one_hot(action_size, a), 0.0])
        obs, r, done, _, _ = env.step(a)
        total_reward += r
        turns += 1

    final_round = env.base_env.env.s.round_idx
    score = total_reward + (final_round - 1) * 2.0
    z = math.tanh(score / value_scale)

    # Broadcast final value to all turns in this scripted episode
    for i in range(len(hist)):
        hist[i][2] = z

    success = final_round >= 2
    return success, hist, score, final_round, turns


def inject_round1_curriculum(
    env: ImprovedNLRDT_GymEnv,
    rb: ReplayBuffer,
    nn: ImprovedNNet,
    opt: optim.Optimizer,
    device,
    demos: int = 6,
    n_place: int = 5,
    warmup_epochs: int = 10,
    value_scale: float = 100.0,
    action_size: Optional[int] = None,
    batch_size: int = 64,
) -> int:
    if action_size is None:
        action_size = env.N_TOTAL_ACTIONS

    print(
        f"[Curriculum] injecting scripted Round-1 clears: "
        f"target={demos}, base_places={n_place}"
    )
    injected = 0
    attempts = 0
    max_attempts = max(demos * 5, 10)

    while injected < demos and attempts < max_attempts:
        attempts += 1
        success, hist, score, fr, turns = _script_round1_episode(
            env, n_place=n_place, value_scale=value_scale, action_size=action_size
        )
        if success:
            rb.add(hist)
            injected += 1
            print(
                f"[Curriculum] +demo {injected}/{demos}  "
                f"round={fr}  turns={turns}  score={score:.2f}"
            )
        else:
            n_place = min(n_place + 1, env.N_STRATEGIC_SLOTS - 1)
            print(
                f"[Curriculum] demo failed (round={fr}). "
                f"Increasing placements -> {n_place}"
            )

    if injected > 0 and warmup_epochs > 0:
        print(f"[Curriculum] warmup training for {warmup_epochs} epochs on injected demos")
        nn.train()
        for e in range(warmup_epochs):
            L = train_nn(nn, rb, opt, device, batch_size)
            if L is not None:
                print(f"  [warmup {e+1}/{warmup_epochs}] loss={L:.4f}")

    return injected


def quick_test():
    """10턴 테스트"""
    device = _get_best_device()
    _print_device_info(device)
    if REQUIRE_MPS and str(device) != "mps":
        print("[FATAL] ALPHA_RTD_REQUIRE_MPS=1 but selected device is", device)
        sys.exit(2)
    base = NLRDT_GymEnv(seed=42)
    env = ImprovedNLRDT_GymEnv(base, n_strategic_slots=16)
    nn = ImprovedNNet(
        state_size=env.STATE_VECTOR_SIZE, action_size=env.N_TOTAL_ACTIONS
    ).to(device)
    nn.eval()

    mcts = MCTS(nn, c_puct=1.0, action_start_round=11)
    print("[quick] obs size:", env.STATE_VECTOR_SIZE, "| actions:", env.N_TOTAL_ACTIONS)
    obs, _ = env.reset()
    _print_device_info(device)
    print(f"[mode] fast={_fast_mode()}", flush=True)
    for t in range(20):
        pi = mcts.get_action_probs(
            env,
            n_simulations=50,
            device=device,
            iter_idx=0,
            episode_idx=0,
            turn_idx=t + 1,
        )
        a = int(np.argmax(pi))

        if a < 13:
            action_name = [
                "DRAW",
                "UP_T",
                "UP_P",
                "UP_Z",
                "SELL_GAS",
                "TERRA_M",
                "TERRA_G",
                "BUY_EPIC",
                "PM1",
                "PM2",
                "PM3",
                "START_ROUND",
                "PASS",
            ][a]
        else:
            idx = a - 13
            ti = idx // env.N_STRATEGIC_SLOTS
            si = idx % env.N_STRATEGIC_SLOTS
            action_name = f"PLACE_{env.tier_order[ti].upper()}_SLOT_{env.strategic_slots[si]}"

        obs, r, done, _, _ = env.step(a)
        print(f"  t={t+1:02d} a={a:3d}({action_name}) r={r:+.2f} done={done}")
        if done:
            break

    print(
        "[quick] round:",
        env.base_env.env.s.round_idx,
        "minerals:",
        env.base_env.env.s.minerals,
        "gas:",
        env.base_env.env.s.gas,
    )


def main_improved():
    # --- hparams ---
    N_ITERATIONS = 1000
    N_EPISODES_PER_ITR = 5
    N_SIMS_PER_MOVE = 128 if _fast_mode() else 200
    N_EPOCHS_PER_ITR = 20
    BATCH_SIZE = 64
    LR = 1e-4
    VALUE_SCALE = 100.0

    # --- saving options (env-overridable) ---
    LATEST_CKPT = "model_v9_latest.pth"
    LATEST_REPLAY = "replay_v9_latest.pkl"
    SAVE_PREFIX = "model_v9_iter_"
    SAVE_EVERY = int(os.environ.get("ALPHA_RTD_SAVE_EVERY", "1"))
    SAVE_PER_EP = int(os.environ.get("ALPHA_RTD_SAVE_PER_EP", "1"))
    SAVE_EVERY_SEC = int(os.environ.get("ALPHA_RTD_SAVE_EVERY_SEC", "0"))
    OUT_DIR = os.environ.get("ALPHA_RTD_OUT_DIR", "./checkpoints")

    os.makedirs(OUT_DIR, exist_ok=True)
    LATEST_CKPT_PATH = os.path.join(OUT_DIR, LATEST_CKPT)
    LATEST_REPLAY_PATH = os.path.join(OUT_DIR, LATEST_REPLAY)
    SAVE_PREFIX_PATH = lambda n: os.path.join(OUT_DIR, f"{SAVE_PREFIX}{n}.pth")

    device = _get_best_device()
    _print_device_info(device)
    if REQUIRE_MPS and str(device) != "mps":
        print("[FATAL] ALPHA_RTD_REQUIRE_MPS=1 but selected device is", device)
        sys.exit(2)
    _req_dev = os.environ.get("ALPHA_RTD_DEVICE", "").lower()
    if _req_dev == "mps" and str(device) != "mps":
        print(
            "[WARN] ALPHA_RTD_DEVICE=mps requested but selected device is",
            device,
            "- ensure Python is arm64 and torch reports mps available.",
        )

    print(
        f"[mode] fast={_fast_mode()}  "
        f"(set ALPHA_RTD_FAST=1 to lighten sims; set ALPHA_RTD_DEVICE=mps to request MPS)",
        flush=True,
    )
    print(f"[Save targets] out_dir={os.path.abspath(OUT_DIR)}", flush=True)
    print(f"  latest_ckpt={os.path.abspath(LATEST_CKPT_PATH)}", flush=True)
    print(f"  replay={os.path.abspath(LATEST_REPLAY_PATH)}", flush=True)
    print(
        f"  every_iter={SAVE_EVERY}  per_ep={SAVE_PER_EP}  heartbeat_sec={SAVE_EVERY_SEC}",
        flush=True,
    )

    base = NLRDT_GymEnv(seed=42)
    env = ImprovedNLRDT_GymEnv(base, n_strategic_slots=(12 if _fast_mode() else 16))
    print(
        f"[Env] obs={env.STATE_VECTOR_SIZE}, actions={env.N_TOTAL_ACTIONS} "
        f"(13+{env.N_PLACE_ACTIONS})"
    )
    print(
        f"[Gate] START_ROUND gate={env.GATE_START}  min_towers={env.MIN_TOWERS_START}"
    )

    nn = ImprovedNNet(
        state_size=env.STATE_VECTOR_SIZE,
        action_size=env.N_TOTAL_ACTIONS,
        width=256,
        n_blocks=3,
    ).to(device)
    opt = optim.Adam(nn.parameters(), lr=LR)
    rb = ReplayBuffer(max_size=50000)
    start_iter = 0

    # ---- initialize SaveScheduler (env-overridable) ----
    scheduler = SaveScheduler(
        out_dir=OUT_DIR,
        keep=int(os.environ.get("ALPHA_RTD_SAVE_KEEP", "10")),
        per_ep=bool(int(os.environ.get("ALPHA_RTD_SAVE_PER_EP", "1"))),
        every_ep=int(os.environ.get("ALPHA_RTD_SAVE_EVERY_EP", "0")),
        every_iter=int(os.environ.get("ALPHA_RTD_SAVE_EVERY_ITER", "0")),
        every_turns=int(os.environ.get("ALPHA_RTD_SAVE_EVERY_TURNS", "0")),
        every_seconds=float(os.environ.get("ALPHA_RTD_SAVE_EVERY_SEC", os.environ.get("ALPHA_RTD_SAVE_EVERY_SECONDS", "0"))),
        prefix=SAVE_PREFIX,
    )

    def _save(iter_index: int, write_iter_file: bool = False, reason: str = "", extra_tag: Optional[str] = None):
        ck = {"model": nn.state_dict(), "optimizer": opt.state_dict(), "iter": iter_index}
        try:
            # always refresh "latest"
            torch.save(ck, LATEST_CKPT_PATH)
            # optional iteration-indexed snapshot
            if write_iter_file:
                torch.save(ck, SAVE_PREFIX_PATH(iter_index))
            # optional tagged snapshot with timestamp
            tagged_path = None
            if extra_tag:
                tagged_path = scheduler.make_tagged_path(iter_index, extra_tag)
                torch.save(ck, tagged_path)
            # replay (latest)
            with open(LATEST_REPLAY_PATH, "wb") as f:
                pickle.dump(rb, f)

            msg = f"  [Save] {reason} -> {LATEST_CKPT_PATH}"
            if write_iter_file:
                msg += f" (+iter:{SAVE_PREFIX_PATH(iter_index)})"
            if tagged_path:
                msg += f" (+tag:{tagged_path})"
            msg += " (+replay)"
            print(msg, flush=True)

            # prune old files matching our prefix
            try:
                scheduler.prune(pattern=f"{SAVE_PREFIX}*.pth")
            except Exception:
                pass
        except Exception as e:
            print(f"  [Save WARN] failed to save ({reason}): {e}", flush=True)

    if os.path.exists(LATEST_CKPT_PATH):
        try:
            ckpt = torch.load(LATEST_CKPT_PATH, map_location=device)
            nn.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            start_iter = ckpt.get("iter", 0)
            print(f"[Load] {LATEST_CKPT_PATH} (iter={start_iter})")
            if os.path.exists(LATEST_REPLAY_PATH):
                with open(LATEST_REPLAY_PATH, "rb") as f:
                    rb = pickle.load(f)
                print(f"[Load] {LATEST_REPLAY_PATH} (size={len(rb)})")
        except Exception as e:
            print(f"[Load WARN] {e}")

    _save(start_iter, write_iter_file=False, reason="Bootstrap Save")

    _force_curr = os.environ.get("ALPHA_RTD_FORCE_CURR", "0") == "1"
    _enable_curr = os.environ.get("ALPHA_RTD_CURRICULUM", "1") == "1"
    _small_replay = len(rb) < int(os.environ.get("ALPHA_RTD_CURR_MIN_REPLAY", "32"))

    if _enable_curr and (start_iter == 0 or _force_curr or _small_replay):
        demos = int(os.environ.get("ALPHA_RTD_CURR_DEMOS", "6"))
        place_cap = int(os.environ.get("ALPHA_RTD_CURR_PLACEMENTS", "5"))
        warmup_epochs = int(os.environ.get("ALPHA_RTD_CURR_EPOCHS", "10"))
        print(
            f"[Curriculum] injecting scripted Round-1 clears "
            f"(reason: start_iter={start_iter}, force={_force_curr}, "
            f"small_replay={_small_replay})"
        )
        injected = inject_round1_curriculum(
            env,
            rb,
            nn,
            opt,
            device,
            demos=demos,
            n_place=place_cap,
            warmup_epochs=warmup_epochs,
            value_scale=VALUE_SCALE,
            action_size=env.N_TOTAL_ACTIONS,
            batch_size=BATCH_SIZE,
        )
        print(f"[Curriculum] injected={injected}  replay_size={len(rb)}")

    mcts = MCTS(nn, c_puct=1.0, action_start_round=11)

    it = start_iter
    last_hb = time.time()
    try:
        for it in range(start_iter, N_ITERATIONS):
            print("\n" + "=" * 60)
            print(f"    Iter {it+1}/{N_ITERATIONS}")
            print("=" * 60)

            nn.eval()
            avg_r, avg_fr, avg_t = [], [], []
            for ep in range(N_EPISODES_PER_ITR):
                def _on_turn():
                    scheduler.bump_turns(1)
                    if scheduler.due(ep_idx=ep, iter_idx=it):
                        _save(it, write_iter_file=False, reason="Periodic", extra_tag="roll")
                        scheduler.mark_saved()

                traj, r, fr, turns = run_self_play_improved(
                    env,
                    mcts,
                    rb,
                    N_SIMS_PER_MOVE,
                    device,
                    iter_idx=it,
                    value_scale=VALUE_SCALE,
                    on_turn=_on_turn,
                )
                rb.add(traj)
                avg_r.append(r)
                avg_fr.append(fr)
                avg_t.append(turns)
                print(
                    f"  [EP {ep+1}] score={r:+7.2f} round={fr:2d} "
                    f"turns={turns:3d} buf={len(rb)}"
                )
                if SAVE_PER_EP or scheduler.per_ep:
                    _save(it, write_iter_file=False, reason=f"Autosave after EP {ep+1}", extra_tag=f"ep{ep+1}")
                    scheduler.mark_saved()
                if SAVE_EVERY_SEC > 0 and (time.time() - last_hb) >= SAVE_EVERY_SEC:
                    _save(it, write_iter_file=False, reason="Heartbeat Save", extra_tag="hb")
                    last_hb = time.time()

            print(
                f"  [Avg] score={np.mean(avg_r):.2f} "
                f"round={np.mean(avg_fr):.1f} "
                f"turns={np.mean(avg_t):.1f}"
            )

            nn.train()
            losses = []
            steps = 0
            for _ in range(N_EPOCHS_PER_ITR):
                L = train_nn(nn, rb, opt, device, BATCH_SIZE)
                if L is not None:
                    losses.append(L)
                    steps += 1
            if steps > 0:
                print(
                    f"  [Train] batches={steps} avg_loss={np.mean(losses):.4f}"
                )

            if (it + 1) % SAVE_EVERY == 0:
                _save(it + 1, write_iter_file=True, reason="End of iteration")
            if scheduler.every_iter > 0 and ((it + 1) % scheduler.every_iter == 0):
                _save(it + 1, write_iter_file=True, reason="Scheduler Iteration", extra_tag="iter_end")
                scheduler.mark_saved()
    except KeyboardInterrupt:
        print(
            "  [Save] Interrupted — writing latest checkpoint/replay...",
            flush=True,
        )
        try:
            _save(it, write_iter_file=False, reason="Interrupted")
        except Exception as e:
            print(f"  [Save WARN] Failed to save on interrupt: {e}")
        raise