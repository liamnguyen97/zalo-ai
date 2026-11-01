#!/usr/bin/env bash
# Put this repo's committed code (and the competition data) on the rented GPU box, ready for scripts/box_run.sh.
# usage (on the Mac, repo root): scripts/box_deploy.sh [branch=main]
# The box gets no GitHub credentials: the commit is pushed over SSH into a bare repo there, and /root/repo is a clone
# of it whose `origin` points at GitHub. Uncommitted changes are NOT deployed. .env is never copied.
set -euo pipefail
cd "$(dirname "$0")/.."
BRANCH=${1:-main}
read -r HOST PORT < <(python3 scripts/vast.py target)
SSH="ssh -p $PORT -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new"
REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")

echo "data -> $HOST:/root/data"
$SSH "$HOST" 'mkdir -p /root/data' 2>/dev/null
rsync -az -e "$SSH" --exclude '*.zip' data/train data/public_test "$HOST":/root/data/

echo "code ($BRANCH @ $(git rev-parse --short "$BRANCH")) -> $HOST:/root/repo"
$SSH "$HOST" "test -d /root/datamate.git || git init -q --bare /root/datamate.git; git -C /root/datamate.git symbolic-ref HEAD refs/heads/$BRANCH" 2>/dev/null
git push -q "ssh://$HOST:$PORT/root/datamate.git" "$BRANCH:$BRANCH"
$SSH "$HOST" "set -e
  if [ -d /root/repo/.git ]; then git -C /root/repo fetch -q /root/datamate.git $BRANCH && git -C /root/repo checkout -q -B $BRANCH FETCH_HEAD
  else git clone -q -b $BRANCH /root/datamate.git /root/repo; fi
  cd /root/repo
  [ -n '$REMOTE_URL' ] && git remote set-url origin '$REMOTE_URL' || true
  ln -sfn /root/data data
  [ -d splits/holdout6 ] || { mkdir -p splits; python3 scripts/subsplit.py ids data/train splits/holdout6 docs/holdout6_ids.txt; }
  UV_LINK_MODE=copy uv sync -q --frozen --python 3.11
  .venv/bin/python -m pytest -q tests 2>&1 | tail -1
  git log --oneline -1" 2>&1 | grep -v "^Welcome\|^Have fun\|^$"
