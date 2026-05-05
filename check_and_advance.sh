#!/bin/bash
# Called after each run to log results and advance to next run.
# Usage: bash check_and_advance.sh <run_number> <commit_sha> <pre_accuracy>
set -e
WORKTREE="/home/james/autoresearch/.claude/worktrees/dazzling-shannon-b70ca4"
LOG="$WORKTREE/run.log"

post=$(grep "^post_accuracy:" "$LOG" | awk '{print $2}')
delta=$(grep "^delta_accuracy:" "$LOG" | awk '{print $2}')
vram=$(grep "^peak_vram_mb:" "$LOG" | awk '{print $2}')
commit=$(git -C "$WORKTREE" rev-parse --short HEAD)

echo "post=$post delta=$delta vram=$vram commit=$commit"
