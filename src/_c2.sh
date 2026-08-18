#!/usr/bin/env bash
set -eu
cd "$HOME/fewshot_gs"
git add -A
git commit -q -F - <<'MSG'
Depth across the crossover: constraint does not cross over, coverage does

Extends the depth 2x2 to k=10 and k=20, completing a 3x2x2 factorial
(subset size x outpainting x depth prior), 3 seeds, 36 runs, paired within
seed.

                     k=5              k=10             k=20
  +depth        +0.259 +-0.111*  +0.163 +-0.073*  +0.155 +-0.039*
  +outpaint     +0.285 +-0.069*  -0.003 +-0.064   -0.618 +-0.195*
  +both         +0.714 +-0.202*  +0.294 +-0.045*  -0.355 +-0.198*
  interaction   +0.169 +-0.094*  +0.134 +-0.136   +0.108 +-0.031*

The depth prior is positive and separated from zero at EVERY subset size,
including k=20 where outpainting costs 0.618 dB. This was a prediction, not
a discovery: the report argues a synthetic view supplies coverage and
inconsistency in fixed proportion, coverage losing value as real views
accumulate while inconsistency does not. An intervention supplying
constraint with no extra view carries no inconsistency, so it should not
cross over. It does not.

The shape confirms the same model. Depth's benefit decays with k
(+0.259 -> +0.163 -> +0.155), exactly as diminishing returns on constraint
predict - it simply never turns negative, because there is no contradiction
term. Coverage and inconsistency are separable, and only one of them
reverses sign.

Practical consequence, now evidenced rather than argued: at 5 or 10 real
views use both; at 20 use the depth prior alone, since adding synthetic
views costs half a dB. At k=10 outpainting alone is flat (-0.003) yet
+both (+0.294) beats depth alone (+0.163) - the prior makes otherwise
worthless synthetic views worth having.

Drivers parameterised by K/NF so the same sweep runs at any subset size.
MSG
git log --oneline -1
git log -1 --format='%an <%ae>'
