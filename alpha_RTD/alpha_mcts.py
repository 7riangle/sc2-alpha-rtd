# alpha_mcts.py
# MCTS + NN 캐싱 + START_ROUND 특별 처리

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

# Helper: strict mask + renormalize, fallback to uniform if all masked
def _mask_and_normalize(vec, mask):
    """
    Apply an action mask to a probability-like vector and renormalize.
    If everything gets masked out, fall back to uniform over valid actions.
    """
    v = np.asarray(vec, dtype=np.float32)
    m = np.asarray(mask, dtype=np.float32)

    # Elementwise mask
    v *= m

    s = float(v.sum())
    if s > 1e-12:
        v /= s
    else:
        ms = float(m.sum())
        if ms > 0.0:
            v = m / ms
        else:
            # no valid actions (extremely rare); return zeros
            v = np.zeros_like(v, dtype=np.float32)
    return v

from alpha_common import _fast_mode
from alpha_env import ImprovedNLRDT_GymEnv


class Node:
    def __init__(self, parent, action, prior_p):
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.Q = 0.0
        self.P = prior_p

    def is_leaf(self):
        return len(self.children) == 0

    def select(self, c_puct):
        best_score = -1e9
        best_child = None
        for a, ch in self.children.items():
            score = ch.Q + c_puct * ch.P * (math.sqrt(self.N) / (1 + ch.N))
            if score > best_score:
                best_score = score
                best_child = ch
        return best_child

    def expand(self, policy_probs):
        for a, p in enumerate(policy_probs):
            if p > 0:
                self.children[a] = Node(self, a, p)

    def backup(self, value):
        self.N += 1
        self.Q += (value - self.Q) / self.N
        if self.parent:
            self.parent.backup(value)


class MCTS:
    def __init__(self, nnet_model, c_puct=1.0, action_start_round=11):
        self.nn = nnet_model
        self.c_puct = c_puct
        self.ACTION_START_ROUND = action_start_round

        if _fast_mode():
            self.HARD_EVALS_FIRST = 0
            self.REEVAL_EVERY = 10
        else:
            self.HARD_EVALS_FIRST = 1
            self.REEVAL_EVERY = 20

        self._turn_counter = 0
        self.nn_cache = {}
        self.start_cache = {}
        self._hard_eval_count = {}

        # --- START_ROUND dampening + logging verbosity ---
        self.sr_dampen_base = 0.25   # reduce START_ROUND prior when other actions exist (0.1~0.3 recommended)
        self.sr_dampen_min  = 0.02   # keep small non-zero floor so SR never becomes exactly 0
        self.VERBOSE = False         # set True to print MCTS progress; False speeds up a bit


    # --- Dampens START_ROUND probability at root if any other valid action exists ---
    def _dampen_start_round_at_root(self, pi, action_mask, root_obs_key):
        a_sr = self.ACTION_START_ROUND
        if a_sr is None:
            return pi
        # if SR is out of range or invalid under the mask, do nothing
        if not (0 <= a_sr < len(action_mask)) or action_mask[a_sr] == 0:
            return pi

        # if SR is the only valid action, do not dampen
        other_valid = np.array(action_mask, dtype=np.float32).copy()
        other_valid[a_sr] = 0.0
        if other_valid.sum() <= 0:
            return pi

        # increase dampening if we've already been here in recent turns
        cnt = int(self.start_cache.get(root_obs_key, 0))
        factor = max(self.sr_dampen_min, float(self.sr_dampen_base ** (1 + cnt)))

        pi = np.asarray(pi, dtype=np.float32).copy()
        pi[a_sr] *= factor
        s = float(pi.sum())
        if s > 1e-12:
            pi /= s
        else:
            m = np.asarray(action_mask, dtype=np.float32)
            ms = float(m.sum())
            pi = m / ms if ms > 0 else np.zeros_like(m, dtype=np.float32)

        self.start_cache[root_obs_key] = cnt + 1
        return pi

    def _get_nn(self, obs_dict, device):
        key = obs_dict["observation"].tobytes()
        if key in self.nn_cache:
            return self.nn_cache[key]
        with torch.no_grad():
            logits, value = self.nn(
                torch.FloatTensor(obs_dict["observation"]).to(device)
            )
        value_scalar = value.squeeze(0).item()
        policy_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        self.nn_cache[key] = (policy_probs, value_scalar)
        return self.nn_cache[key]

    def get_action_probs(
        self,
        gym_env,
        n_simulations=100,
        device=torch.device("cpu"),
        iter_idx=0,
        episode_idx=None,
        turn_idx=None,
    ):
        self._turn_counter += 1
        if self._turn_counter % self.REEVAL_EVERY == 0:
            self.start_cache = {}
            self._hard_eval_count = {}
        self.nn_cache = {}

        if episode_idx is not None and turn_idx is not None:
            _prog_prefix = f"[ep {episode_idx} turn {turn_idx}] "
        elif turn_idx is not None:
            _prog_prefix = f"[turn {turn_idx}] "
        else:
            _prog_prefix = ""

        if isinstance(gym_env, ImprovedNLRDT_GymEnv):
            root_snapshot = gym_env.base_env.snapshot()
            root_obs = gym_env._get_obs()
        else:
            root_snapshot = gym_env.snapshot()
            root_obs = gym_env._get_obs()

        policy_probs, root_value = self._get_nn(root_obs, device)
        action_mask = root_obs["action_mask"]

        # Root prior with strict masking
        policy_probs = _mask_and_normalize(policy_probs, action_mask)

        # Dirichlet noise for exploration (root only), then mask again
        noise = np.random.dirichlet([0.3] * len(policy_probs))
        pi = 0.75 * policy_probs + 0.25 * noise
        pi = _mask_and_normalize(pi, action_mask)
        # Damp START_ROUND at the root if there are other valid options
        root_key = root_obs["observation"].tobytes()
        pi = self._dampen_start_round_at_root(pi, action_mask, root_key)

        root = Node(None, None, 1.0)
        root.expand(pi)

        for i in range(n_simulations):
            hb = max(1, n_simulations // 4)
            if (i + 1) % hb == 0:
                if self.VERBOSE:
                    print(f"    [MCTS] {_prog_prefix}progress {i+1}/{n_simulations}", flush=True)

            if isinstance(gym_env, ImprovedNLRDT_GymEnv):
                gym_env.base_env.restore(root_snapshot)
            else:
                gym_env.restore(root_snapshot)
            node = root

            while not node.is_leaf():
                # Current state's valid action mask
                try:
                    cur_obs_for_mask = gym_env._get_obs()
                    cur_mask = np.asarray(cur_obs_for_mask.get("action_mask", None), dtype=np.float32)
                except Exception:
                    cur_mask = None

                # Choose child by UCT but restricted to currently valid actions
                if cur_mask is not None:
                    best_score = -1e18
                    best_child = None
                    for a, ch in node.children.items():
                        if 0 <= a < cur_mask.shape[0] and cur_mask[a] > 0:
                            score = ch.Q + self.c_puct * ch.P * (math.sqrt(max(1, node.N)) / (1 + ch.N))
                            if score > best_score:
                                best_score = score
                                best_child = ch
                    if best_child is None:
                        # No existing child is valid under current mask -> pick a valid random action and create it
                        valid = np.where(cur_mask > 0)[0]
                        if valid.size == 0:
                            # Pathological: no valid actions
                            node = None
                            break
                        a = int(np.random.choice(valid))
                        best_child = node.children.get(a)
                        if best_child is None:
                            best_child = Node(node, a, 1.0)
                            node.children[a] = best_child
                    node = best_child
                    a = node.action
                else:
                    # Fallback: original selection (no mask available)
                    node = node.select(self.c_puct)
                    a = node.action

                # Step the environment with the chosen (mask-valid) action
                obs, rew, done, _, _ = gym_env.step(a)
                if done:
                    node.backup(rew)
                    node = None
                    break

            if node is None:
                continue

            cur_obs = gym_env._get_obs()
            p, v = self._get_nn(cur_obs, device)
            am = cur_obs["action_mask"]
            p = _mask_and_normalize(p, am)

            node.expand(p)
            node.backup(v)

        counts = [root.children.get(a, Node(None, None, 0)).N for a in range(len(policy_probs))]
        sc = sum(counts)

        try:
            if isinstance(gym_env, ImprovedNLRDT_GymEnv):
                gym_env.base_env.restore(root_snapshot)
                gym_env._cache_fp = None
                gym_env._cache_obs = None
                gym_env._cache_stats = None
            else:
                gym_env.restore(root_snapshot)
        except Exception:
            pass

        # Mask-safe final policy from visit counts
        if sc == 0:
            am = gym_env._get_obs()["action_mask"]
            return am.astype(float) / max(1, am.sum())

        pi = np.array(counts, dtype=np.float32)
        pi = _mask_and_normalize(pi, root_obs["action_mask"])
        return pi.astype(np.float32)