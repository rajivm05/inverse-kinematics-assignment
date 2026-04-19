# CSCI 520 — Assignment 3: Skinning, Forward Kinematics, Inverse Kinematics

Rajiv Murali — rajivmur@usc.edu

## Overview

This project implements a character animation pipeline: a triangle mesh is
deformed by a joint skeleton. At runtime the user drags an IK handle; the
driver solves for the joint angles that move the handle toward the target,
then skins the mesh to those new joint transforms.

Three components were implemented from scratch:

1. **Skinning** (`skinning.cpp`) — linear blend skinning, plus dual-quaternion
   skinning as extra credit.
2. **Forward kinematics** (`FK.cpp`) — Maya-style joint hierarchy:
   `localR = R_jointOrient · R_euler(rotateOrder)`, chained root → leaves via
   `jointUpdateOrder`.
3. **Inverse kinematics** (`IK.cpp`) — Tikhonov-regularized damped least
   squares, with the Jacobian supplied by ADOL-C automatic differentiation of
   the FK function.

## Build & Run

```
make
./driver armadillo/skin.config     # or dragon/, hand/
```

The Makefile links OpenGL/GLUT (as frameworks on macOS), ADOL-C, and the
bundled Vega utilities.

## Core Algorithms

### Linear Blend Skinning

For each mesh vertex:

```
p_deformed = Σ_j w_{j,v} · M_skin_j · p_rest
```

where `M_skin_j = M_global_j(current) · M_global_j(rest)^-1`. Weights are
static; the joint index table is precomputed and the inner loop iterates over the fixed number
of influencing joints per vertex.

### Forward Kinematics

The local transform of a joint is built from its rest translation and two
rotations — the joint orientation (XYZ) and the user-controlled Euler angles
(in the joint's declared rotate order). These match Maya's convention. The
global transform is the parent's global times the local. `jointUpdateOrder`
is precomputed so the traversal visits parents before children. For
skinning, the global is multiplied by the inverse rest-pose global transform.

### Inverse Kinematics (Tikhonov)

Let `θ` be the flattened joint Euler angles and `b(θ)` the concatenated
world-space handle positions. Given a target `b*`, the solver computes:

```
(J^T J + α I) Δθ = J^T (b* − b(θ))
```

with small α > 0 (Tikhonov regularization) to keep the normal matrix
non-singular when handles are under-constrained. The Jacobian `J = ∂b/∂θ` is
produced by ADOL-C: the FK function is traced once into an ADOL-C tape, and
the Jacobian is evaluated on demand. The linear system is solved with
Eigen's `ldlt()`.

Two ADOL-C tapes coexist in the process — tag `1` traces joint-handle FK and
tag `2` traces vertex-handle FK (see extra credit).

## Controls

| Key        | Action                                                     |
|------------|------------------------------------------------------------|
| left-drag  | Rotate camera                                              |
| mid-drag   | Pan camera                                                 |
| right-drag | Zoom                                                       |
| click handle + drag | Move IK target                                   |
| `r` / `0`  | Reset pose to rest + reset camera                          |
| `a`        | Toggle world axes                                          |
| `w`        | Toggle wireframe                                           |
| `e`        | Toggle mesh rendering                                      |
| `s`        | Toggle skeleton rendering                                  |
| `=`        | Cycle highlighted joint                                    |
| `d`        | Toggle LBS / DQS skinning                                  |
| `v`        | Toggle joint-handle / vertex-handle IK                     |
| `i`        | Toggle IK sub-stepping on/off                              |
| `c`        | Toggle rainbow bone colors                                 |
| `p`        | Record current pose as a keyframe                          |
| space      | Play / pause pose playback                                 |
| `x`        | Clear recorded poses                                       |
| `F`        | Toggle frame capture to `frames/*.ppm`                     |
| esc        | Quit                                                       |

## Extra Credit

### 1. Dual-Quaternion Skinning

LBS is simple and fast but collapses volume at large twists — the classic
"candy-wrapper" defect on a twisted forearm. DQS blends unit dual
quaternions (rotation + translation fused into an 8-component number), then
renormalizes. The blended transform stays rigid (no non-uniform scale), so
volume is preserved through bends.

Implementation: `skinning.cpp::applyDQS` converts each
`RigidTransform4d` into a dual quaternion (rotation to quaternion via
Shepperd's method; translation half-multiplied with the rotation
quaternion). Per vertex, influencing joints' dual quaternions are blended
with skinning weights, flipping sign on any whose real part is anti-parallel
to the first (antipodality correction), then normalized. The result
transforms the rest position.

**Comparison:** on the armadillo's forearm and the dragon's tail, LBS shows
visible pinching where joint rotations approach 90°. DQS keeps cross-section
volume and looks noticeably rounder. DQS costs more per vertex (quaternion
blend + normalize vs matrix blend), but it's still O(K) per vertex for K
influencing joints. Toggle live with `d` to compare.

### 2. IK Handles at Mesh Vertices

Instead of solving to move skeleton joints, this mode solves to move
specific mesh vertices. This is what a real rig actually wants — grabbing a
fingertip instead of a wrist joint.

Implementation: a second IK object is built with its own ADOL-C tape. The
traced function first computes joint global transforms (same as joint mode),
then skins the selected vertex IDs by linear blend through the joint
transforms. The output is the concatenated world positions of those
vertices. Everything downstream (Jacobian, Tikhonov solve, sub-stepping) is
identical to joint mode.

The driver preloads both IK objects at startup and switches between them at
runtime with `v` — no restart needed. Vertex handle IDs are configured via
`*IKVertexIDs` in the `.config` file. A small red dot marks each handle on
the rendered mesh.

### 3. IK Sub-Stepping

Tikhonov IK is based on a first-order linearization. When the user yanks a
handle a large distance in a single frame, the Jacobian evaluated at the
current pose is a poor predictor of the target pose, and the solve overshoots
or gets stuck. Sub-stepping fixes this by breaking large motions into
smaller, linearly-trustworthy steps.

Implementation: at the start of `doIK`, the maximum displacement across all
handles is measured. If it exceeds `maxStepDistance` (tuned to
`modelRadius * 0.05` by the driver), the motion is split into
`ceil(maxDisplacement / maxStepDistance)` sub-steps (capped at
`maxSubSteps = 20`). Each sub-step linearly interpolates the target from the
starting position to the final target, re-evaluates the Jacobian at the
current pose, and applies a Tikhonov Δθ.

Console log:

```
IK sub-steps: 7 (max handle drag 0.352391)
IK sub-steps: 17 (max handle drag 0.947651)
IK sub-steps: 20 (max handle drag 1.77575)
```

Toggle with `i` to A/B compare. With sub-stepping off, large drags produce
wildly overshooting or twisted poses; with it on, the pose stays in the
locally-valid neighborhood that the Jacobian describes.

## Algorithms Implemented

One core IK solver — Tikhonov damped least squares with α = 0.01 — exposed
in two modes, with an optional sub-stepping wrapper. The comparisons below
are between these variants.

### Single-step Tikhonov vs sub-stepped Tikhonov

Same solver, same α. The only difference is whether a large handle
displacement is fed to the solver all at once, or split into
`ceil(maxDrag / maxStepDistance)` sub-steps (up to 20) that each get their
own Jacobian evaluation and Tikhonov solve.

| Aspect                  | Single step                                         | Sub-stepped                                                     |
|-------------------------|-----------------------------------------------------|------------------------------------------------------------------|
| Small drags (< threshold) | Identical — no sub-stepping kicks in              | Identical                                                        |
| Large drags             | Jacobian is evaluated far from target; pose overshoots or twists into a bad local minimum | Each sub-step stays in the linearization's trust region; pose tracks the target smoothly |
| Cost per frame          | One Jacobian + one solve                            | Up to 20× that (but only on the frames that actually need it)    |
| Failure mode            | Visible snapping / flipping limbs on big yanks      | Graceful; still subject to the underlying limitations below      |

The sub-step count is logged whenever it exceeds 1, e.g.
`IK sub-steps: 17 (max handle drag 0.947651)`. Toggle the wrapper with `i`
to A/B compare on the same drag. Sub-stepping does not change what the
solver is "allowed" to produce — it just keeps each linearization honest.

### Joint-handle IK vs vertex-handle IK

Both modes use the same Tikhonov solver and the same sub-stepping wrapper.
They differ only in what the ADOL-C-traced forward function outputs:

- **Joint mode (tape tag 1):** output is the 3D world position of each
  listed skeleton joint. Handle count = number of joints named in
  `*IKJointIDs`.
- **Vertex mode (tape tag 2):** output is the 3D world position of each
  listed mesh vertex, obtained by first computing joint globals and then
  skinning (linear blend) the rest vertex position through them. Handle
  count = number of vertices named in `*IKVertexIDs`.

| Aspect                     | Joint handles                                         | Vertex handles                                             |
|----------------------------|-------------------------------------------------------|-------------------------------------------------------------|
| What you grab              | Skeleton pivot points (elbow, wrist, ankle…)          | Specific points on the skin (fingertip, nose tip, tail tip) |
| Intuitiveness for posing   | Lower — you're pulling the bone, not the surface      | Higher — directly manipulates what you can see             |
| Tape setup cost            | Cheap — small output dimension                        | Slightly larger output dim; one extra skinning pass in the trace |
| Per-step solve cost        | Same                                                  | Same (Jacobian shape matches handle count in both cases)   |
| Interaction with DQS render | Faithful — joint global positions are identical under LBS and DQS | Inconsistent — trace skins with LBS even when render is DQS (see Limitations) |

Runtime switching is wired to `v`; both IK objects are preloaded at startup
so there's no rebuild cost when toggling.

### Skinning: LBS vs DQS

Independent of IK, two skinning implementations share the same vertex data
and are toggled live with `d`:

| Aspect              | Linear blend (LBS)                          | Dual quaternion (DQS)                                        |
|---------------------|---------------------------------------------|---------------------------------------------------------------|
| Blend quantity      | 4×4 rigid matrices                          | Unit dual quaternions (with antipodality correction)          |
| Cost per vertex     | Cheapest — matrix-vector blend              | ~2–3× more (quaternion blend + normalization)                 |
| Volume at bends     | Collapses ("candy-wrapper" at ~90°)         | Preserved — blended transform stays rigid                     |
| Artifacts           | Pinching in twisted forearms, elbows        | Slight bulging at extreme blends; no pinching                 |
| Where it matters    | Barely visible at small joint angles        | Most visible on the armadillo forearm, dragon tail            |

### Known Limitations

- **No joint limits.** The solver treats each Euler angle as unbounded, so
  knees can bend backwards and shoulders can rotate past anatomical range.
- **No self-collision.** When dragging one leg across the body, it will pass
  through the other leg/torso without resistance. Expected; fixing it would
  require a collision term in the objective.
- **LBS in the vertex-IK trace.** The ADOL-C tape for vertex-handle IK
  implements skinning as linear blend, even when the render toggle is set to
  DQS. So when solving in vertex mode with DQS rendering, there is a small
  inconsistency: the visible mesh follows DQS, but the solver's internal
  model assumes LBS. The handle ends up in roughly the right place but not
  exactly where DQS would put it. Fixing this would require an ADOL-C-typed
  dual-quaternion blend; out of scope for this assignment.

## Additional Features

These are quality-of-life and demo aids added on top of the assignment.

### Rainbow Bone Coloring (`c`)

Each joint is assigned a distinct hue using golden-ratio hue spacing (so the
colors stay visually separated regardless of joint count). Per-vertex colors
are a skinning-weight-blend of the influencing joints' palette colors.
Vertices influenced cleanly by one bone show as a solid hue; vertices with
shared influence show as a blend. This is a direct visualization of the
skinning weight map.

### Pose Recording & Playback (`p` / space / `x`)

- `p` takes a snapshot of all joint Euler angles.
- space toggles playback, which loops through the recorded keyframes with a
  cosine-smoothstep interpolation at 1 second per segment.
- `x` clears the recording.

During playback, IK drag input is ignored — the pose is driven entirely by
the keyframes — and IK handle world positions are re-synced each frame so the
user can resume dragging seamlessly when they pause.

### Frame Capture (`F`)

`F` toggles writing the back buffer to `frames/frame_NNNN.ppm` each rendered
frame (P6 binary). PPM is chosen to avoid pulling in a JPEG encoder
dependency; 

## File Map

| File                  | What's in it                                                        |
|-----------------------|---------------------------------------------------------------------|
| `skinning.cpp/.h`     | LBS + DQS; vertex skinning data accessors for vertex-IK             |
| `FK.cpp/.h`           | Joint hierarchy, local/global/skin transforms, rest transforms      |
| `IK.cpp/.h`           | Two IK modes (joint/vertex), ADOL-C tapes, Tikhonov solve, sub-stepping |
| `driver.cpp`          | GLUT app: rendering, input, pose recorder, frame capture            |
| `skeletonRenderer.*`  | Bone rendering (provided)                                           |
| `handleControl.*`     | Mouse handle manipulation (provided)                                |
| `vega/`               | Vega utility library (provided)                                     |
| `eigen/`, `adolc/`    | Third-party dependencies                                            |
| `armadillo/`, `dragon/`, `hand/` | Sample models and configs                                |
