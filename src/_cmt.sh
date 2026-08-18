#!/usr/bin/env bash
set -eu
cd "$HOME/fewshot_gs"
git add -A
git commit -q -F - <<'MSG'
Depth regularisation: the strongest result in the study

A depth prior supplies geometric constraint with no extra view, so unlike a
synthetic image it carries no multi-view inconsistency. Tested as a 2x2 at
k=5, three seeds, paired within seed.

  baseline   15.197
  +depth     +0.259 +- 0.111 *
  +outpaint  +0.285 +- 0.069 *
  +both      +0.714 +- 0.202 *      <- best result anywhere in this study

The two combine super-additively: per-seed interaction +0.196 / +0.247 /
+0.065, mean +0.169 +- 0.094, positive on every seed. They are fixing
different deficiencies - outpainting adds peripheral CONTENT the real views
never saw, the depth prior adds CONSTRAINT on geometry already observed - so
the gains compound instead of overlapping. That agrees with the duplication
control, which found 80-85% of outpainting's benefit comes from fabricated
borders rather than repeated views.

Depth is also the cleaner intervention. It improves all three metrics
(SSIM 0.5375 -> 0.5593, LPIPS 0.4096 -> 0.3901) where outpainting buys PSNR
while degrading SSIM, and costs +6.6% Gaussians against outpainting's +103%.

Two bugs had to be fixed first; both produced runs that completed normally
and reproduced the baseline exactly, which is indistinguishable from a
genuine null result:

  - add_depths.py wrote each scene's depth_params.json through a symlink into
    the shared dataset, so the first scene's 37 entries replaced the
    dataset's 251 and every later scene read the truncated file.
  - gaussian-splatting/utils/make_depth_scale.py slices its cv2.remap output
    as [..., 0], which on OpenCV 5 takes the first COLUMN of a (1, N) result
    rather than a channel, collapsing 4117 samples to one. The mean absolute
    deviation of a single value is 0, so every image got scale=inf, which
    poisons the median in dataset_readers.py and fails every image at the
    reliability gate in cameras.py. src/make_depth_params.py replaces it with
    correct sampling and a least-squares fit (median R^2 0.964).

src/check_depth_coverage.py now gates the sweep: it counts how many TRAINING
views actually survive to be supervised and refuses to train if any seed has
none. Counting all images in a scene is what hid the fault - it read "37 real
supervised" while all five training views were masked.
MSG
git log --oneline -1
echo "--- identity / trailers ---"
git log -1 --format='%an <%ae>'
git log -1 --format='%B' | grep -ci 'co-authored-by' || true
