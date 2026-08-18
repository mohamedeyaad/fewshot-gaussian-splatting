#!/usr/bin/env bash
# Make the history read as solely Mohamed Eyad's work.
#
#   1. sets the repo's committer identity for all FUTURE commits
#   2. rewrites author+committer on every EXISTING commit to that identity
#      (they were recorded as "mooeyad <moeyad.khalaf@gmail.com>")
#   3. strips the Co-Authored-By trailer, which GitHub parses to build the
#      repository's contributor list
#
# Safe: nothing has been pushed, so no other clone can be desynced by the
# rewrite, and filter-branch stashes pre-rewrite refs under refs/original/.
set -eu
cd "$HOME/fewshot_gs"

NAME="Mohamed Eyad"
MAIL="mooeyad@gmail.com"

# filter-branch refuses to run with a dirty tree, and .gitignore has an
# uncommitted edit from the viewer work.
if [ -n "$(git status --porcelain)" ]; then
  echo "--- committing pending changes first ---"
  git status --short
  git add -A
  git -c user.name="$NAME" -c user.email="$MAIL" commit -q -m \
"Ignore viewer/: regenerated point clouds are not the experimental record

src/run_viewer_models.sh retrains seven conditions purely to produce .ply
files for the SuperSplat viewer. They go to viewer/ rather than runs/
because training is not bit-deterministic (noise floor sigma = 0.039 dB),
so retraining into runs/ would overwrite results.json and drift every
published number away from the committed record."
fi

git config user.name "$NAME"
git config user.email "$MAIL"

echo
echo "before: $(git log --format='%H' --grep='Co-Authored-By' | wc -l) of $(git rev-list --count HEAD) commits carry a co-author trailer"
git tag -f backup-before-identity-rewrite HEAD >/dev/null
echo "backup tag: backup-before-identity-rewrite -> $(git rev-parse --short backup-before-identity-rewrite)"

export FILTER_BRANCH_SQUELCH_WARNING=1
export NAME MAIL
git filter-branch -f \
  --env-filter '
    export GIT_AUTHOR_NAME="$NAME"
    export GIT_AUTHOR_EMAIL="$MAIL"
    export GIT_COMMITTER_NAME="$NAME"
    export GIT_COMMITTER_EMAIL="$MAIL"
  ' \
  --msg-filter 'sed "/^Co-[Aa]uthored-[Bb]y:/d"' \
  -- --all

echo
echo "after : $(git log --format='%H' --grep='Co-Authored-By' | wc -l) of $(git rev-list --count HEAD) commits carry a co-author trailer"
echo
echo "--- every author/committer identity in history ---"
git log --format='author:    %an <%ae>' | sort -u
git log --format='committer: %cn <%ce>' | sort -u
echo
echo "--- history ---"
git log --oneline
echo
echo "--- most recent message, verbatim ---"
git log -1 --format='%B' | sed 's/^/    /'
