# Alpha RTD.py
# Entry point for AlphaRTD v9 (modularized) with sane defaults for gating, autosave, and round logging

import os
from pathlib import Path

# ---- sensible defaults (override via environment if you want) ----
os.environ.setdefault("ALPHA_RTD_GATE_START", "1")             # strictly gate premature START_ROUND
os.environ.setdefault("ALPHA_RTD_MIN_TOWERS", "5")             # need ≥5 towers before starting
os.environ.setdefault("ALPHA_RTD_REQUIRE_ZERO_M", "1")         # spend minerals before starting
os.environ.setdefault("ALPHA_RTD_MIN_UPGRADE_TOTAL", "1")      # at least one upgrade across T/P/Z

os.environ.setdefault("ALPHA_RTD_ROUND_LOG", "0")              # write JSONL round logs
os.environ.setdefault("ALPHA_RTD_LOG_PATH", str(Path("./round_log.jsonl").resolve()))

# training/runtime QoL
# ---- save policy toggles (all optional, can be set in shell) ----
os.environ.setdefault("ALPHA_RTD_SAVE_DIR", "./checkpoints")
os.environ.setdefault("ALPHA_RTD_SAVE_KEEP", "10")            # 보관 개수 N
os.environ.setdefault("ALPHA_RTD_SAVE_REPLAY", "1")           # 리플까지 함께 저장

# 주기 선택: 아래 중 편한 것(들)을 켜면 된다
os.environ.setdefault("ALPHA_RTD_SAVE_PER_EP", "1")           # 에피소드마다 1=저장
os.environ.setdefault("ALPHA_RTD_SAVE_EVERY_EP", "10")         # N 에피소드마다 저장 (0=off)
os.environ.setdefault("ALPHA_RTD_SAVE_EVERY_ITER", "0")       # N 이터마다 저장 (0=off)
# --- resume & progress defaults (so we don't start from scratch if a checkpoint exists) ---
os.environ.setdefault("ALPHA_RTD_RESUME", "1")               # resume from latest if present
os.environ.setdefault("ALPHA_RTD_CKPT_PATH", str(Path(os.environ.get("ALPHA_RTD_SAVE_DIR", "./checkpoints")) / "model_v9_latest.pth"))
os.environ.setdefault("ALPHA_RTD_PRINT_MCTS", "1")           # show MCTS progress 50/200 style if alpha_mcts supports it
os.environ.setdefault("ALPHA_RTD_MCTS_SIMS", "200")          # default sims (lower to 64/100 for faster dev)
#os.environ.setdefault("ALPHA_RTD_SAVE_EVERY_TURNS", "0")      # N 턴마다 저장 (0=off)
#os.environ.setdefault("ALPHA_RTD_SAVE_EVERY_SECONDS", "0")    # N 초마다 저장 (0=off)
from alpha_train import main_improved  # , quick_test


if __name__ == "__main__":
    # 빠른 테스트만 하고 싶으면 아래 줄 주석 해제:
    # quick_test()
    main_improved()
