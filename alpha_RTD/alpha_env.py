# alpha_env.py
# Gym 환경 래퍼 (13액션 + Improved 배치/피처/보상)

import os
import random
from typing import Dict, List, Tuple, Optional, Set
import json
from datetime import datetime

import numpy as np
import gymnasium as gym

from alpha_common import CONFIG, UNITS, SLOTS, Env, pos_on_path, L_TOTAL, _fast_mode


# =========================
# 1) Base Gym wrapper (13 actions)
# =========================

class NLRDT_GymEnv(gym.Env):
    """Base 13-action env; v9에서는 Improved 래퍼로 확장"""

    metadata = {"render.modes": []}

    def __init__(self, seed=42):
        super().__init__()
        self.env = Env(seed=seed)
        self.rng = random.Random(seed)
        self.CONFIG = CONFIG

        self.all_unit_names = []
        for tier in ["normal", "rare", "epic", "legend", "god"]:
            if tier in UNITS:
                self.all_unit_names.extend(UNITS[tier]["name"].tolist())
        self.unit_name_to_idx = {name: i for i, name in enumerate(self.all_unit_names)}
        self.N_UNITS_TOTAL = len(self.all_unit_names)
        self.all_mission_ids = [m["id"] for m in self.CONFIG.get("missions", [])]
        self.mission_id_to_idx = {name: i for i, name in enumerate(self.all_mission_ids)}
        self.N_MISSIONS_TOTAL = len(self.all_mission_ids)
        self.N_SLOTS_TOTAL = len(SLOTS)

        # configurable START_ROUND gating (shared with Improved wrapper)
        self.GATE_START = os.environ.get("ALPHA_RTD_GATE_START", "1") == "1"
        self.MIN_TOWERS_START = int(os.environ.get("ALPHA_RTD_MIN_TOWERS", "5"))
        self.REQUIRE_ZERO_MINERALS_START = os.environ.get("ALPHA_RTD_REQUIRE_ZERO_M", "1") == "1"
        # require at least this many total upgrades across T/P/Z before START_ROUND
        self.MIN_TOTAL_UPGRADE_LEVEL = int(os.environ.get("ALPHA_RTD_MIN_UPGRADE_TOTAL", "1"))
        self.action_space = gym.spaces.Discrete(13)
        self.STATE_VECTOR_SIZE = (
            3 + 3 + 1 + 3 + self.N_UNITS_TOTAL + self.N_SLOTS_TOTAL + self.N_MISSIONS_TOTAL
        )
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=0, high=np.inf, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(13,), dtype=np.int8),
            }
        )
        self.slot_id_to_vec_idx = {int(r.slot_id): i for i, r in SLOTS.iterrows()}

        # round result logging
        self.ENABLE_ROUND_LOG = os.environ.get("ALPHA_RTD_ROUND_LOG", "1") == "1"
        self._round_log_path = os.environ.get("ALPHA_RTD_LOG_PATH", "./round_log.txt")
        self._prev_missions_logged = set()
    def _snapshot_placements(self):
        s = self.env.s
        placements = []
        for sid, slotmap in s.placed.items():
            for (tier, name), cnt in slotmap.items():
                if cnt > 0:
                    placements.append({
                        "slot_id": int(sid),
                        "tier": tier,
                        "name": name,
                        "count": int(cnt),
                    })
        placements.sort(key=lambda x: (x["slot_id"], x["tier"], x["name"]))
        return placements

    def _log_after_round(self, round_before: int, clear_time: float, result: str):
        """Append one JSONL record per round transition (robust to None clear_time)."""
        if not self.ENABLE_ROUND_LOG:
            return
        s = self.env.s
        round_after = int(s.round_idx)
        cleared_round = int(round_after - 1) if round_after > round_before else int(round_before)

        missions_now = set(getattr(s, "missions_done", set()))
        just_completed = sorted(list(missions_now - self._prev_missions_logged))
        self._prev_missions_logged = set(missions_now)

        rec = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "result": result,  # "cleared" or "failed"
            "cleared_round": int(cleared_round),
            "next_round": int(round_after),
            "clear_time": (None if clear_time is None else float(clear_time)),
            "bank": {"minerals": int(s.minerals), "gas": int(s.gas), "terrazine": int(s.terrazine)},
            "upgrades": {"T": int(s.levels.get("T", 0)), "P": int(s.levels.get("P", 0)), "Z": int(s.levels.get("Z", 0))},
            "placements": self._snapshot_placements(),
            "missions_done": sorted(list(missions_now)),
            "missions_just_completed": just_completed,
        }
        try:
                        # ensure directory exists
            try:
                d = os.path.dirname(self._round_log_path)
                if d:
                    os.makedirs(d, exist_ok=True)
            except Exception:
                pass
            with open(self._round_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            # Logging should never crash the env
            pass

    def snapshot(self):
        return self.env.snapshot()

    def restore(self, state_snapshot):
        self.env.restore(state_snapshot)

    def _normalize_obs(self, state_vec):
        state_vec[0] /= 1000.0
        state_vec[1] /= 1000.0
        state_vec[2] /= 10.0
        state_vec[3:6] /= 20.0
        state_vec[6] /= 50.0
        state_vec[7:10] /= 300.0
        return state_vec

    def _get_obs(self):
        s = self.env.s
        state_vec = np.zeros(self.STATE_VECTOR_SIZE, dtype=np.float32)
        off = 0
        state_vec[off] = s.minerals
        off += 1
        state_vec[off] = s.gas
        off += 1
        state_vec[off] = s.terrazine
        off += 1
        state_vec[off] = s.levels.get("T", 0)
        off += 1
        state_vec[off] = s.levels.get("P", 0)
        off += 1
        state_vec[off] = s.levels.get("Z", 0)
        off += 1
        state_vec[off] = s.round_idx
        off += 1
        state_vec[off] = max(0.0, s.pm_next_ready.get("pm1", 0.0) - s.time_sec)
        off += 1
        state_vec[off] = max(0.0, s.pm_next_ready.get("pm2", 0.0) - s.time_sec)
        off += 1
        state_vec[off] = max(0.0, s.pm_next_ready.get("pm3", 0.0) - s.time_sec)
        off += 1

        for tier, units in s.inv.items():
            for name, count in units.items():
                idx = self.unit_name_to_idx.get(name, None)
                if idx is not None:
                    state_vec[off + idx] = count
        off += self.N_UNITS_TOTAL

        for slot_id, slotmap in s.placed.items():
            count = sum(slotmap.values())
            vec_idx = self.slot_id_to_vec_idx.get(slot_id, None)
            if vec_idx is not None:
                state_vec[off + vec_idx] = count
        off += self.N_SLOTS_TOTAL

        for mission_id in getattr(s, "missions_done", set()):
            idx = self.mission_id_to_idx.get(mission_id, None)
            if idx is not None:
                state_vec[off + idx] = 1.0
        off += self.N_MISSIONS_TOTAL

        mask = np.zeros(13, dtype=np.int8)
        cfg_econ = self.CONFIG["economy"]
        cfg_pm = self.CONFIG.get("personal_missions", {})
        if s.minerals >= cfg_econ["draw_cost"]:
            mask[0] = 1
        cost_t = cfg_econ["upgrade"]["base"] + cfg_econ["upgrade"]["step"] * s.levels.get("T", 0)
        if s.gas >= cost_t:
            mask[1] = 1
        cost_p = cfg_econ["upgrade"]["base"] + cfg_econ["upgrade"]["step"] * s.levels.get("P", 0)
        if s.gas >= cost_p:
            mask[2] = 1
        cost_z = cfg_econ["upgrade"]["base"] + cfg_econ["upgrade"]["step"] * s.levels.get("Z", 0)
        if s.gas >= cost_z:
            mask[3] = 1
        if s.minerals >= 100:
            mask[4] = 1
        if s.terrazine >= 1:
            mask[5] = 1
        if s.terrazine >= 1:
            mask[6] = 1
        cfg_terra_buy = cfg_econ["terrazine"]["buy_epic"]
        if s.terrazine >= cfg_terra_buy["terrazine"] and s.minerals >= cfg_terra_buy["minerals"]:
            mask[7] = 1
        pm1_cfg = cfg_pm.get("pm1", {})
        if s.round_idx >= pm1_cfg.get("unlock_round", 0) and s.time_sec >= s.pm_next_ready.get(
            "pm1", 0.0
        ):
            mask[8] = 1
        pm2_cfg = cfg_pm.get("pm2", {})
        if s.round_idx >= pm2_cfg.get("unlock_round", 0) and s.time_sec >= s.pm_next_ready.get(
            "pm2", 0.0
        ):
            mask[9] = 1
        pm3_cfg = cfg_pm.get("pm3", {})
        if s.round_idx >= pm3_cfg.get("unlock_round", 0) and s.time_sec >= s.pm_next_ready.get(
            "pm3", 0.0
        ):
            mask[10] = 1
        mask[11] = 1
        mask[12] = 1

        # START_ROUND hard gate for the base env as well:
        # require: (1) ≥ MIN_TOWERS_START towers, (2) minerals == 0 (if enabled), (3) ≥ MIN_TOTAL_UPGRADE_LEVEL upgrades across T/P/Z.
        if self.GATE_START and mask[11] == 1:
            try:
                towers = self.env._make_towers(include_virtual=False)
                n_towers = len(towers)
            except Exception:
                n_towers = 0
            require_zero_m = self.REQUIRE_ZERO_MINERALS_START
            levels_total = (
                int(self.env.s.levels.get("T", 0))
                + int(self.env.s.levels.get("P", 0))
                + int(self.env.s.levels.get("Z", 0))
            )
            missing_upg = levels_total < self.MIN_TOTAL_UPGRADE_LEVEL
            if (n_towers < self.MIN_TOWERS_START) or (require_zero_m and self.env.s.minerals > 0) or missing_upg:
                mask[11] = 0

        # package observation
        state_vec = self._normalize_obs(state_vec)
        ret = {"observation": state_vec, "action_mask": mask}
        return ret
        
    def _ok_to_start(self) -> bool:
        """Hard gate for START_ROUND at the environment level (safety even if mask is ignored)."""
        if not self.GATE_START:
            return True
        try:
            n_towers = len(self.env._make_towers(include_virtual=False))
        except Exception:
            n_towers = 0
        if n_towers < self.MIN_TOWERS_START:
            return False
        if self.REQUIRE_ZERO_MINERALS_START and self.env.s.minerals > 0:
            return False
        levels_total = int(self.env.s.levels.get("T", 0)) + int(self.env.s.levels.get("P", 0)) + int(self.env.s.levels.get("Z", 0))
        if levels_total < self.MIN_TOTAL_UPGRADE_LEVEL:
            return False
        return True

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env = Env(seed=seed)
            self.rng = random.Random(seed)
        self.env.reset()
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        done = False
        info = {}
        s = self.env.s
        t_before = s.time_sec
        try:
            if action == 0:
                self.env.draw_normal()
            elif action == 1:
                self.env.upgrade("T")
            elif action == 2:
                self.env.upgrade("P")
            elif action == 3:
                self.env.upgrade("Z")
            elif action == 4:
                self.env.sell_100m_for_gas()
            elif action == 5:
                self.env.terra_to_minerals()
            elif action == 6:
                self.env.terra_to_gas()
            elif action == 7:
                self.env.terra_buy_epic()
            elif action == 8:
                success, clear_time = self.env.trigger_pm("pm1")
                if not success:
                    reward = -100.0
                    done = True
                else:
                    reward = (
                        CONFIG["personal_missions"]["pm1"]["reward_minerals"] / 100.0
                    ) - (clear_time - t_before)
            elif action == 9:
                success, clear_time = self.env.trigger_pm("pm2")
                if not success:
                    reward = -100.0
                    done = True
                else:
                    reward = (
                        CONFIG["personal_missions"]["pm2"]["reward_minerals"] / 100.0
                    ) - (clear_time - t_before)
            elif action == 10:
                success, clear_time = self.env.trigger_pm("pm3")
                if not success:
                    reward = -100.0
                    done = True
                else:
                    reward = (
                        CONFIG["personal_missions"]["pm3"]["reward_minerals"] / 100.0
                    ) - (clear_time - t_before)
            elif action == 11:
                # START_ROUND: enforce gate here as well (do not terminate episode on gate fail)
                if not self._ok_to_start():
                    reward = -1.0
                    done = False
                    try:
                        n_towers = len(self.env._make_towers(include_virtual=False))
                    except Exception:
                        n_towers = 0
                    levels_total = int(self.env.s.levels.get("T", 0)) + int(self.env.s.levels.get("P", 0)) + int(self.env.s.levels.get("Z", 0))
                    info = {
                        "reason": "gate_blocked",
                        "need": {
                            "min_towers": self.MIN_TOWERS_START,
                            "minerals_zero": bool(self.REQUIRE_ZERO_MINERALS_START),
                            "min_upgrade_total": self.MIN_TOTAL_UPGRADE_LEVEL,
                        },
                        "cur": {
                            "towers": int(n_towers),
                            "minerals": int(self.env.s.minerals),
                            "upgrade_total": int(levels_total),
                        },
                    }
                else:
                    round_before = int(self.env.s.round_idx)
                    success, clear_time = self.env.start_round()
                    if not success:
                        reward = -100.0
                        done = True
                        # write round failure log
                        self._log_after_round(round_before, clear_time, result="failed")
                    else:
                        reward = -(clear_time - t_before)
                        # write round completion log
                        self._log_after_round(round_before, clear_time, result="cleared")
                        if self.env.s.round_idx > 50:
                            reward += 1000.0
                            done = True
            elif action == 12:
                # PASS
                pass
        except AssertionError:
            reward = -0.1
        obs = self._get_obs()
        return obs, reward, done, False, info


# =========================
# 2) v9 Improved wrapper (placement+features+reward shaping)
# =========================

class ImprovedNLRDT_GymEnv(gym.Env):
    """v9 확장: 배치 액션(5티어×k슬롯), 상태에 [total_dps, n_towers, coverage], 즉시 보상, 진행 보너스,
    ● 필드 합성 규칙: 2×normal→1×rare, 2×rare→1×epic, 2×epic→1×legend, 2×legend→1×god (god은 종단, 더 이상 합성 불가)
    ● 미션(일반/희귀/특급/전설 전종 수집) 보상: R≤10:+300/600/1200/2400m, R>10:+500/900/1800/3600m
    ● v9: field-only normal draw, on-board multi-tier fusion at chosen slot (no inventory)"""

    tier_order = ["normal", "rare", "epic", "legend", "god"]

    def __init__(self, base_env: NLRDT_GymEnv, strategic_slots=None, n_strategic_slots=20):
        super().__init__()
        self.base_env = base_env
        self.CONFIG = base_env.CONFIG

        if strategic_slots is None:
            self.strategic_slots = (
                SLOTS["slot_id"].head(n_strategic_slots).astype(int).tolist()
            )
        else:
            self.strategic_slots = [int(x) for x in strategic_slots[:n_strategic_slots]]
        self.N_STRATEGIC_SLOTS = len(self.strategic_slots)

        # --- catalogue lists (field-only gameplay; draw places directly, no inventory) ---
        try:
            self.normal_names = UNITS["normal"]["name"].tolist()
        except Exception:
            self.normal_names = []
        try:
            self.rare_names = set(UNITS["rare"]["name"].tolist())
        except Exception:
            self.rare_names = set()
        try:
            self.epic_names = set(UNITS["epic"]["name"].tolist())
        except Exception:
            self.epic_names = set()
        try:
            self.legend_names = set(UNITS["legend"]["name"].tolist())
        except Exception:
            self.legend_names = set()
        try:
            self.god_names = set(UNITS["god"]["name"].tolist())
        except Exception:
            self.god_names = set()

        # fusion graph: rare→epic→legend→god (god is terminal)
        self._next_tier = {"rare": "epic", "epic": "legend", "legend": "god"}

        # cache slot coordinates once to avoid expensive pandas.iterrows calls
        self._slot_xy = {int(r.slot_id): (float(r.x), float(r.y)) for _, r in SLOTS.iterrows()}
        self._slot_max_stack = {int(r.slot_id): int(r.max_stack) for _, r in SLOTS.iterrows()}

        self._fast = _fast_mode()

        self.N_PLACE_ACTIONS = 5 * self.N_STRATEGIC_SLOTS
        self.N_TOTAL_ACTIONS = 13 + self.N_PLACE_ACTIONS
        self.action_space = gym.spaces.Discrete(self.N_TOTAL_ACTIONS)

        # configurable START_ROUND gating (can disable via ALPHA_RTD_GATE_START=0)
        self.GATE_START = os.environ.get("ALPHA_RTD_GATE_START", "1") == "1"
        self.MIN_TOWERS_START = int(os.environ.get("ALPHA_RTD_MIN_TOWERS", "5"))
        # require at least this many total upgrades across T/P/Z before START_ROUND
        self.MIN_TOTAL_UPGRADE_LEVEL = int(os.environ.get("ALPHA_RTD_MIN_UPGRADE_TOTAL", "1"))
        # optional additional thresholds to discourage "suicidal" early START_ROUND
        self.MIN_COVERAGE_START = float(os.environ.get("ALPHA_RTD_MIN_COVERAGE", "0.25"))
        self.MIN_DPS_START = float(os.environ.get("ALPHA_RTD_MIN_DPS", "30.0"))

        self.STATE_VECTOR_SIZE = self.base_env.STATE_VECTOR_SIZE + 3  # +[total_dps, n_towers, coverage]
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(
                    low=0, high=np.inf, shape=(self.STATE_VECTOR_SIZE,), dtype=np.float32
                ),
                "action_mask": gym.spaces.Box(
                    low=0, high=1, shape=(self.N_TOTAL_ACTIONS,), dtype=np.int8
                ),
            }
        )

        # --- simple caches to avoid recomputing coverage/obs every MCTS call ---
        self._cache_fp = None
        self._cache_obs = None
        self._cache_stats = None

        # shaping
        self.PLACE_REWARD = {"normal": 1.0, "rare": 3.0, "epic": 6.0, "legend": 10.0, "god": 15.0}
        self.PAIR_BONUS_NORMAL = 5.0  # bonus when placing the second copy of a normal unit anywhere on board
        self.COVERAGE_BONUS_SCALE = 2.0
        self._pass_count = 0 # for escalating PASS penalty

        # --- synthesis & mission shaping ---
        self.ENABLE_AUTO_SYNTH_RARE = True
        # mission: collect all distinct normal-tier units (inventory + placed)
        self.MISSION_NORMAL_SET_ID = "collect_normals_all"
        self.MISSION_NORMAL_SET_BONUS_PRE10 = 300  # minerals
        self.MISSION_NORMAL_SET_BONUS_POST10 = 500  # minerals
        self._mission_normal_set_done = False
        # cache: how many different normal types exist in data
        try:
            self.N_NORMAL_TYPES = int(len(UNITS.get("normal", {}).get("name", [])))
        except Exception:
            # UNITS["normal"] is likely a DataFrame
            try:
                self.N_NORMAL_TYPES = int(UNITS["normal"]["name"].nunique())
            except Exception:
                self.N_NORMAL_TYPES = 0

        # 추가 세트 미션(희귀/특급/전설) — 보드 전종 수집
        self.MISSION_RARE_SET_ID = "collect_rares_all"
        self.MISSION_RARE_SET_BONUS_PRE10 = int(os.environ.get("ALPHA_RTD_RARE_PRE10", "600"))
        self.MISSION_RARE_SET_BONUS_POST10 = int(os.environ.get("ALPHA_RTD_RARE_POST10", "900"))

        self.MISSION_EPIC_SET_ID = "collect_epics_all"
        self.MISSION_EPIC_SET_BONUS_PRE10 = int(os.environ.get("ALPHA_RTD_EPIC_PRE10", "1200"))
        self.MISSION_EPIC_SET_BONUS_POST10 = int(os.environ.get("ALPHA_RTD_EPIC_POST10", "1800"))

        self.MISSION_LEGEND_SET_ID = "collect_legends_all"
        self.MISSION_LEGEND_SET_BONUS_PRE10 = int(os.environ.get("ALPHA_RTD_LEGEND_PRE10", "2400"))
        self.MISSION_LEGEND_SET_BONUS_POST10 = int(os.environ.get("ALPHA_RTD_LEGEND_POST10", "3600"))

        self._mission_rare_set_done = False
        self._mission_epic_set_done = False
        self._mission_legend_set_done = False

        try:
            self.N_RARE_TYPES = int(UNITS["rare"]["name"].nunique())
        except Exception:
            self.N_RARE_TYPES = 0
        try:
            self.N_EPIC_TYPES = int(UNITS["epic"]["name"].nunique())
        except Exception:
            self.N_EPIC_TYPES = 0
        try:
            self.N_LEGEND_TYPES = int(UNITS["legend"]["name"].nunique())
        except Exception:
            self.N_LEGEND_TYPES = 0

    def _fingerprint_state(self):
        """Build a lightweight immutable signature of the current env state for caching."""
        s = self.base_env.env.s
        levels = (s.levels.get("T", 0), s.levels.get("P", 0), s.levels.get("Z", 0))
        placed_sig = tuple(
            sorted(
                (sid, tuple(sorted(((t, n), c) for (t, n), c in slot.items())))
                for sid, slot in s.placed.items()
            )
        )
        return (
            s.round_idx,
            s.time_sec,
            s.minerals,
            s.gas,
            s.terrazine,
            levels,
            placed_sig,
        )

    def _summary_stats(self):
        fp = self._fingerprint_state()
        if self._cache_fp == fp and self._cache_stats is not None:
            return self._cache_stats

        towers = self.base_env.env._make_towers(include_virtual=False)
        total_dps = float(sum(t.dps for t in towers))
        n_towers = float(len(towers))
        if n_towers == 0:
            self._cache_fp = fp
            self._cache_stats = (0.0, 0.0, 0.0)
            return self._cache_stats

        samples = 60 if self._fast else 120
        hit = 0
        for i in range(samples):
            dist = L_TOTAL * (i / (samples - 1))
            x, y = pos_on_path(dist)
            covered = False
            for tw in towers:
                sid = int(tw.slot_id)
                if sid not in self._slot_xy:
                    continue
                sx, sy = self._slot_xy[sid]
                if (x - sx) ** 2 + (y - sy) ** 2 <= tw.rng ** 2:
                    covered = True
                    break
            if covered:
                hit += 1

        cov = hit / samples
        self._cache_fp = fp
        self._cache_stats = (total_dps, n_towers, cov)
        return self._cache_stats

    def _has_fusion_candidates_at(self, slot_id: int, from_tier: str):
        """
        Return list of unit NAMES at slot_id (of tier `from_tier`) that have a second copy
        somewhere on the BOARD (any slot). Only names that exist in the NEXT tier's catalogue
        are considered valid, e.g., rare→epic, epic→legend, legend→god. God is terminal.
        """
        if from_tier == "god":
            return []

        # which names are valid to fuse up from `from_tier`?
        if from_tier == "normal":
            valid_names = self.rare_names
        elif from_tier == "rare":
            valid_names = self.epic_names
        elif from_tier == "epic":
            valid_names = self.legend_names
        elif from_tier == "legend":
            valid_names = self.god_names
        else:
            valid_names = set()

        s = self.base_env.env.s

        # names present at this slot in the required tier
        here = {}
        slotmap = s.placed.get(slot_id, {})
        for (tier, name), cnt in slotmap.items():
            if tier == from_tier and cnt > 0:
                here[name] = cnt
        if not here:
            return []

        # total copies on the board for that tier
        total_by_name = {}
        for sm in s.placed.values():
            for (tier, name), cnt in sm.items():
                if tier == from_tier and cnt > 0:
                    total_by_name[name] = total_by_name.get(name, 0) + cnt

        cands = []
        for name in here:
            if total_by_name.get(name, 0) >= 2 and ((not valid_names) or (name in valid_names)):
                cands.append(name)
        return cands

    def _best_name_by_dps(self, names: List[str], to_tier: str) -> Optional[str]:
        if not names:
            return None
        try:
            tbl = UNITS.get(to_tier, None)
            if tbl is None:
                return names[0]
            best = None
            best_score = -1e9
            for nm in names:
                row = tbl.loc[tbl["name"] == nm]
                score = float(row["dps"].iloc[0]) if (not row.empty and "dps" in row.columns) else 0.0
                if score > best_score:
                    best_score, best = score, nm
            return best if best is not None else names[0]
        except Exception:
            return names[0]

    def _fuse_at_slot(self, slot_id: int, from_tier: str) -> Optional[str]:
        """
        Consume two copies of the SAME unit from the BOARD at tier `from_tier`
        (one copy must be at `slot_id`), and create +1 unit of the NEXT tier
        at `slot_id`. Returns the fused unit name on success, else None.
        God-tier is terminal (no fusion).
        """
        if from_tier == "god":
            return None
        to_tier = self._next_tier.get(from_tier, None)
        if to_tier is None:
            return None

        s = self.base_env.env.s
        cands = self._has_fusion_candidates_at(slot_id, from_tier)
        if not cands:
            return None
        # pick best name by the DPS of the RESULT tier
        name = self._best_name_by_dps(cands, to_tier)

        # remove one from the base slot
        if not self._dec_unit_at_slot(slot_id, from_tier, name, 1):
            return None

        # remove the second copy (prefer same slot first)
        removed_second = False
        if self._dec_unit_at_slot(slot_id, from_tier, name, 1):
            removed_second = True
        else:
            for other_sid, slotmap in list(s.placed.items()):
                if other_sid == slot_id:
                    continue
                if slotmap.get((from_tier, name), 0) > 0:
                    removed_second = self._dec_unit_at_slot(other_sid, from_tier, name, 1)
                    if removed_second:
                        break
        if not removed_second:
            # roll back
            self._inc_unit_at_slot(slot_id, from_tier, name, 1)
            return None

        # add one at the NEXT tier on the chosen slot
        self._inc_unit_at_slot(slot_id, to_tier, name, 1)
        return name

    def _apply_collection_mission_bonus_normal(self) -> int:
        """Mission: collect all distinct NORMAL units ON THE BOARD (no inventory)."""
        if self._mission_normal_set_done or self.N_NORMAL_TYPES <= 0:
            return 0
        s = self.base_env.env.s
        have = set()
        for slot in s.placed.values():
            for (tier, name), cnt in slot.items():
                if tier == "normal" and cnt > 0:
                    have.add(name)
        if len(have) < self.N_NORMAL_TYPES:
            return 0
        bonus = self.MISSION_NORMAL_SET_BONUS_PRE10 if int(s.round_idx) <= 10 else self.MISSION_NORMAL_SET_BONUS_POST10
        s.minerals = int(s.minerals) + int(bonus)
        self._mission_normal_set_done = True
        try:
            s.missions_done.add(self.MISSION_NORMAL_SET_ID)
        except Exception:
            pass
        return int(bonus)

    def _apply_collection_mission_bonus_for(self, tier: str, need_count: int, mission_id: str, pre10: int, post10: int, done_attr: str) -> int:
        """Generic board-only set-collection mission for a given tier."""
        if need_count <= 0 or getattr(self, done_attr, False):
            return 0
        s = self.base_env.env.s
        have = set()
        for slot in s.placed.values():
            for (t, name), cnt in slot.items():
                if t == tier and cnt > 0:
                    have.add(name)
        if len(have) < int(need_count):
            return 0
        bonus = pre10 if int(s.round_idx) <= 10 else post10
        s.minerals = int(s.minerals) + int(bonus)
        setattr(self, done_attr, True)
        try:
            s.missions_done.add(mission_id)
        except Exception:
            pass
        return int(bonus)

    def _inc_unit_at_slot(self, slot_id: int, tier: str, name: str, delta: int = 1) -> None:
        s = self.base_env.env.s
        s.placed.setdefault(slot_id, {})
        key = (tier, name)
        s.placed[slot_id][key] = int(s.placed[slot_id].get(key, 0)) + int(delta)
        if s.placed[slot_id][key] <= 0:
            try:
                del s.placed[slot_id][key]
            except Exception:
                pass

    def _dec_unit_at_slot(self, slot_id: int, tier: str, name: str, delta: int = 1) -> bool:
        s = self.base_env.env.s
        slotmap = s.placed.get(slot_id, {})
        key = (tier, name)
        cur = int(slotmap.get(key, 0))
        if cur < delta:
            return False
        newv = cur - int(delta)
        if newv > 0:
            slotmap[key] = newv
        else:
            try:
                del slotmap[key]
            except Exception:
                pass
        return True

    def _get_obs(self):
        fp = self._fingerprint_state()
        if self._cache_fp == fp and self._cache_obs is not None:
            return self._cache_obs

        base = self.base_env._get_obs()
        total_dps, n_towers, cov = self._summary_stats()
        ext = np.array([total_dps / 1000.0, n_towers / 50.0, cov], dtype=np.float32)
        obs_vec = np.concatenate([base["observation"], ext])
        mask = np.zeros(self.N_TOTAL_ACTIONS, dtype=np.int8)
        mask[:13] = base["action_mask"]
        # disable bare DRAW (action 0) and inventory-only epic-buy (action 7)
        mask[0] = 0
        mask[7] = 0

        # placement actions mask
        s = self.base_env.env.s
        base_off = 13
        draw_cost = self.CONFIG["economy"]["draw_cost"]
        for ti, tier in enumerate(self.tier_order):
            for si, slot_id in enumerate(self.strategic_slots):
                ai = base_off + ti * self.N_STRATEGIC_SLOTS + si
                max_stack = self._slot_max_stack.get(slot_id, 1)
                cur = sum(self.base_env.env.s.placed.get(slot_id, {}).values())

                if tier == "normal":
                    # pay draw cost and place directly on board (no inventory)
                    mask[ai] = 1 if (cur < max_stack and s.minerals >= draw_cost and len(self.normal_names) > 0) else 0
                elif tier in ("rare", "epic", "legend"):
                    # fusion at this slot is possible if this slot holds a unit of `tier`'s source
                    # rare<= from_tier=normal, epic<= rare, legend<= epic (god is terminal)
                    from_tier = {"rare": "normal", "epic": "rare", "legend": "epic"}[tier]
                    can_fuse_here = len(self._has_fusion_candidates_at(slot_id, from_tier)) > 0
                    mask[ai] = 1 if can_fuse_here else 0
                else:
                    # god-tier placement/fusion is disabled (terminal)
                    mask[ai] = 0

        # --- START_ROUND hard gate: require (towers >= MIN_TOWERS_START), zero minerals, and min upgrades ---
        if self.GATE_START and mask[11] == 1:
            _, n_towers_cur, _ = self._summary_stats()
            s = self.base_env.env.s
            levels_total = int(s.levels.get("T", 0)) + int(s.levels.get("P", 0)) + int(s.levels.get("Z", 0))
            if (n_towers_cur < self.MIN_TOWERS_START) or (s.minerals > 0) or (levels_total < self.MIN_TOTAL_UPGRADE_LEVEL):
                mask[11] = 0

        ret = {"observation": obs_vec, "action_mask": mask}
        self._cache_fp = fp
        self._cache_obs = ret
        return ret

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.base_env.reset(seed=seed, options=options)
        else:
            self.base_env.reset()
        self._cache_fp = None
        self._cache_obs = None
        self._cache_stats = None
        self._mission_normal_set_done = False
        self._mission_rare_set_done = False
        self._mission_epic_set_done = False
        self._mission_legend_set_done = False
        self._pass_count = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._cache_fp = None
        self._cache_obs = None
        self._cache_stats = None

        pre_total, pre_n, pre_cov = self._summary_stats()

        if action < 13:
            # Strict START_ROUND guard at the improved wrapper level (before calling base_env)
            if action == 11:
                tot, n_t, cov = self._summary_stats()
                s = self.base_env.env.s
                levels_total = int(s.levels.get("T", 0)) + int(s.levels.get("P", 0)) + int(s.levels.get("Z", 0))
                if self.GATE_START and ((n_t < self.MIN_TOWERS_START) or (s.minerals > 0) or (levels_total < self.MIN_TOTAL_UPGRADE_LEVEL)):
                    obs = self._get_obs()
                    return obs, -1.0, False, False, {
                        "reason": "gate_blocked_strict",
                        "need": {
                            "min_towers": self.MIN_TOWERS_START,
                            "minerals_zero": True,
                            "min_upgrade_total": self.MIN_TOTAL_UPGRADE_LEVEL,
                        },
                        "cur": {
                            "towers": int(n_t),
                            "minerals": int(s.minerals),
                            "upgrade_total": int(levels_total),
                        },
                    }

            _, reward, terminated, truncated, info = self.base_env.step(action)

            if action == 12:  # PASS
                self._pass_count += 1
                reward -= min(0.06 * (self._pass_count ** 1.3), 0.5)  # escalating PASS penalty
            else:
                self._pass_count = 0

            # missions: set-collection (normal/rare/epic/legend), board-only
            mission_paid = 0
            mission_paid += self._apply_collection_mission_bonus_normal()
            mission_paid += self._apply_collection_mission_bonus_for("rare",   self.N_RARE_TYPES,   self.MISSION_RARE_SET_ID,   self.MISSION_RARE_SET_BONUS_PRE10,   self.MISSION_RARE_SET_BONUS_POST10,   "_mission_rare_set_done")
            mission_paid += self._apply_collection_mission_bonus_for("epic",   self.N_EPIC_TYPES,   self.MISSION_EPIC_SET_ID,   self.MISSION_EPIC_SET_BONUS_PRE10,   self.MISSION_EPIC_SET_BONUS_POST10,   "_mission_epic_set_done")
            mission_paid += self._apply_collection_mission_bonus_for("legend", self.N_LEGEND_TYPES, self.MISSION_LEGEND_SET_ID, self.MISSION_LEGEND_SET_BONUS_PRE10, self.MISSION_LEGEND_SET_BONUS_POST10, "_mission_legend_set_done")
            if mission_paid > 0:
                reward += float(mission_paid) / 100.0

            obs = self._get_obs()
        else:
            # placement actions
            place_idx = action - 13
            tier_idx = place_idx // self.N_STRATEGIC_SLOTS
            slot_idx = place_idx % self.N_STRATEGIC_SLOTS
            tier = self.tier_order[tier_idx]
            slot_id = self.strategic_slots[slot_idx]

            s = self.base_env.env.s
            reward = 0.0
            terminated = False
            truncated = False
            info = {}

            try:
                if tier == "normal":
                    # Spend minerals and directly place a randomly drawn normal at this slot (no inventory)
                    draw_cost = self.CONFIG["economy"]["draw_cost"]
                    max_stack = self._slot_max_stack.get(slot_id, 1)
                    cur = sum(s.placed.get(slot_id, {}).values())

                    if s.minerals < draw_cost or cur >= max_stack or len(self.normal_names) == 0:
                        reward -= 0.02  # safety penalty; should be masked
                    else:
                        s.minerals = int(s.minerals) - int(draw_cost)
                        # uniform 1/N draw among normal pool (original game: 1/8)
                        name = self.base_env.rng.choice(self.normal_names)
                        # count how many copies of this normal already exist on the board (all slots)
                        key = ("normal", name)
                        pre_count = 0
                        for sid, slotmap in s.placed.items():
                            pre_count += int(slotmap.get(key, 0))
                        # place the new unit
                        self._inc_unit_at_slot(slot_id, "normal", name, 1)
                        reward += self.PLACE_REWARD["normal"]
                        # give a small bonus precisely when we complete a pair (1 -> 2 copies on board)
                        if pre_count == 1:
                            reward += self.PAIR_BONUS_NORMAL
                elif tier in ("rare", "epic", "legend"):
                    # Field fusion for higher tiers: 2x from_tier → 1x to_tier at this slot
                    from_tier = {"rare": "normal", "epic": "rare", "legend": "epic"}[tier]
                    fused = self._fuse_at_slot(slot_id, from_tier)
                    if fused is None:
                        reward -= 0.02  # should be masked
                    else:
                        reward += self.PLACE_REWARD[tier] + 1.0  # slight bonus for crafting
                else:
                    # god-tier placement/fusion is disabled (terminal)
                    reward -= 0.02
            except AssertionError:
                reward -= 0.02

            # coverage shaping
            post_total, post_n, post_cov = self._summary_stats()
            dcov = max(0.0, post_cov - pre_cov)
            reward += self.COVERAGE_BONUS_SCALE * dcov

            # any placement resets PASS streak
            self._pass_count = 0

            # missions: set-collection (normal/rare/epic/legend), board-only
            mission_paid = 0
            mission_paid += self._apply_collection_mission_bonus_normal()
            mission_paid += self._apply_collection_mission_bonus_for("rare",   self.N_RARE_TYPES,   self.MISSION_RARE_SET_ID,   self.MISSION_RARE_SET_BONUS_PRE10,   self.MISSION_RARE_SET_BONUS_POST10,   "_mission_rare_set_done")
            mission_paid += self._apply_collection_mission_bonus_for("epic",   self.N_EPIC_TYPES,   self.MISSION_EPIC_SET_ID,   self.MISSION_EPIC_SET_BONUS_PRE10,   self.MISSION_EPIC_SET_BONUS_POST10,   "_mission_epic_set_done")
            mission_paid += self._apply_collection_mission_bonus_for("legend", self.N_LEGEND_TYPES, self.MISSION_LEGEND_SET_ID, self.MISSION_LEGEND_SET_BONUS_PRE10, self.MISSION_LEGEND_SET_BONUS_POST10, "_mission_legend_set_done")
            if mission_paid > 0:
                reward += float(mission_paid) / 100.0

            obs = self._get_obs()

        # small progress bonus when START_ROUND doesn't immediately die
        if action == 11 and not terminated:
            r_now = self.base_env.env.s.round_idx
            reward += float(r_now - 1) * 2.0

        return obs, reward, terminated, truncated, {}