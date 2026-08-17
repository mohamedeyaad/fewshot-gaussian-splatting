# Capturing a custom scene

Notes for reconstructing a second scene with COLMAP, to test whether the
augmentation findings generalise beyond Tanks & Temples `truck`.

## Subject selection

COLMAP matches SIFT features between images. No texture, no features, no
reconstruction.

| works | fails |
|---|---|
| parked car / motorbike | blank walls, plain floors |
| statue, monument, bench | glass, mirrors, chrome, water |
| textured machinery, bicycle | anything moving (people, foliage in wind) |
| patterned surfaces generally | strongly reflective or specular surfaces |

A parked vehicle is the best choice here: it matches the object class of the
reference scene, so a two-scene comparison isolates *capture* as the variable
rather than confounding it with subject type.

Overcast light is ideal. Hard sunlight casts shadows that move with the
photographer, and moving shadows are interpreted as moving geometry.

## Camera settings

1. Lock exposure and focus (AE/AF lock). Drifting auto-exposure between frames
   breaks photometric consistency.
2. HDR off — it fuses sub-frames inconsistently.
3. Portrait / bokeh mode off — synthetic depth-of-field destroys geometry.
4. **Never change lens mid-capture.** No pinch-zoom, no 0.5x/2x tap. COLMAP
   assumes a single shared camera model; switching lenses changes the intrinsics
   and is one of the most common causes of a failed reconstruction.
5. Live Photo / motion photo off.

## Capture pattern

Three complete rings at different heights:

| ring | camera height | frames |
|---|---|---|
| 1 | knee | ~45 |
| 2 | chest | ~45 |
| 3 | overhead | ~45 |

≈135 frames total. The project spec requires N >= 100 *registered* images, and
COLMAP will reject some, so capture with margin.

- One small step between consecutive frames (~70-80% overlap).
- Subject fills roughly two thirds of the frame; leave background visible, since
  background features aid registration.
- Constant distance within each ring.
- Pause before each shot. Motion blur is invisible on a phone screen and fatal
  to feature matching.

## Processing

```bash
# 1. Downsample (7.6 GB RAM will not comfortably match full-resolution frames)
python src/prepare_capture.py --input <raw_photos> --out data/custom/<name> --max-dim 1600

# 2. Sparse reconstruction
bash src/run_colmap.sh data/custom/<name>
```

Check the registration rate before committing to training. Below ~80% of
captured frames registered indicates a capture problem — most often lens
switching, motion blur, or insufficient overlap — and it is cheaper to reshoot
than to train on a bad model.
