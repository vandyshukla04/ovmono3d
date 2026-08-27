# TA-DETR vs the field — positioning recce for the paper
*(intro / related-work / experimental-comparison groundwork; drafted 2026-08-27 while A1 trains.
Sections marked ⏳ are being filled from live literature sweeps; everything in §1–§3 is measured
in-project and citable to our own artifacts.)*

## 0. The claim we are building toward (one paragraph)

Wildlife monitoring from drones needs, per animal, **where it is in 3D and which way it faces** —
for counting under occlusion, for viewpoint-gated individual identification, and for planning
flights that see every side of an animal without disturbance. Existing drone-wildlife computer
vision is (as far as the literature shows — §6) entirely 2D, and existing monocular 3D detectors —
including the open-vocabulary generation built on foundation models — fail outright in this regime
(§1). We contribute (a) WildBox, the first aerial wildlife benchmark with 3D boxes, and (b)
TA-DETR, a detector built around the structure of the aerial regime instead of against it: depth is
**computed by differentiable ray–terrain intersection, never regressed**, orientation is factorized
into an axis and a head/tail sign so that pseudo-label ambiguities cannot poison it, and scale is
resolved jointly across the herd. The core empirical surprise motivating the design: **a
training-free geometric lift already beats every fine-tuned detector's depth on WildBox** (§2).

## 1. Why generalist mono3D fails here — MEASURED, not asserted

All numbers are ours, on WildBox val (66,951 annotations, 6 species), official Omni3D evaluator.

| system | protocol | AP3D / BEV@0.50 | NHD-z (depth) | reading |
|---|---|---|---|---|
| OVMono3D-LIFT (open-vocab, zero-shot) | GT 2D boxes given | **0.00** | z ≈ 11.48 | complete failure with PERFECT 2D — the failure is the 3D lift, not detection |
| OVMono3D-LIFT, fine-tuned on WildBox | own 2D | 13.17 / 8.68 (3-seed) | 5.878–6.494 | the incumbent; depth is 84.5–91.8% of its residual error (disentangled) |
| DetAny3D, fine-tuned | own 2D | BEV 8.33@0.25 / 1.99@0.50 | — | the promptable-3D generation transfers WORSE than the older lift |
| trivial floor (GT 2D + z≡1 + class-median dims) | oracle ranking | BEV 23.66@0.25 / 8.94@0.50 | — | ties the fine-tuned model's published BEV — most of what fine-tuning learns here is ranking, not geometry |

Mechanism, measured over the released videos: oblique aerial viewing (median depression 15.8°, 0%
nadir), animals at a median 142 px in 1920×1080, and **digital zoom that makes focal length
effectively unknown** (true fx varies 10.9× pooled, up to 3.4× within one video) — exactly the
regime where the apparent-size depth cue that generalist detectors lean on becomes unidentifiable
(a large far animal and a small near one project identically). Within a segment, depth spans only
±10–20%: the problem is not depth STRUCTURE (a per-frame-anchor-free depth error of 1.09% is
achievable by the incumbent) but the **per-frame depth anchor**, which apparent size cannot supply
under zoom.

## 2. Our design, and the evidence each element already has

| element | what it does | measured evidence so far |
|---|---|---|
| **terrain-anchored depth** (never regressed) | per-segment height field from the dataset's own reconstruction; per-animal 2D ground-contact point; depth = differentiable ray–terrain intersection, landing directly in the label gauge | training-free A0 with GT 2D: **NHD-z 5.317 vs 5.878 for the BEST fine-tuned model**; raw z err 1.31% vs 2.67%; on real GroundingDINO detections still 6.079 (inside the fine-tuned band). GT bottoms sit within 0.142·H of the terrain (345/345 segments) |
| **contact, not size** | the depth cue is where the animal touches the ground, not how big it appears — zoom-invariant by construction | the zoom-unidentifiability analysis above; A0's z is achieved with CLASS-MEDIAN dims (size carries no depth information in A0 at all) |
| **height-residual δh** | absorbs the contact-vs-terrain gap (hidden feet, mounds, reconstruction bias) | measured species-dependent bias: box bottoms sit +0.11–0.14·H above terrain for rhino/elephant (legs under-reconstructed at 518-px), ≈0 for zebras — a learned residual's exact job |
| **axis × sign orientation** | body axis mod π supervised densely from boxes; head/tail sign only from motion-derived labels | pseudo-label rotations are sign-arbitrary (10.9% frame flips) AND standard metrics are exactly 180°-flip-blind (proven: byte-identical evaluator output under global flip) — supervising yaw directly fails SILENTLY; the factorization is the only honest route. Prior project: elephant sign 96.2% vs 78.7% floor with this recipe |
| **herd latent scale** | same-species animals in frame share a latent scale (amortized posterior); individuals deviate within allometric variance | untested yet (ablation A4); the falsifiable prediction is posterior std ∝ 1/√N |
| **telemetry conditioning** | drone SRT gives exact fx (focal/36×width), gimbal pitch, altitude — free at deployment | gimbal-derived ground normal matches GT normal to p50 0.73°; the dataset's own recovered intrinsics are off by 0.10–0.71× vs telemetry truth |
| **frozen-FM backbone + small head** | DINOv2 ViT-L frozen; 8.9M trainable | the heading result above was achieved with frozen-DINO features + a tiny head; bet inherited |

## 3. The dataset argument (WildBox; why no one could do this before)

- No prior aerial dataset annotates animals in 3D (⏳ §6 verifies against the literature). WildBox:
  60k frames, 345 video segments, 6 savanna species, ~237k 3D boxes (pseudo-labels from an offline
  VGGT+SAM reconstruction pipeline), video-level splits, DJI telemetry.
- Honesty ledger the paper must carry: GT is pseudo-labeled (known biases: dims aspect collapse,
  under-reconstructed legs ⇒ shin-height box bottoms, per-segment scale normalization ⇒ metric
  claims out of scope); a human-audited junk population (|off-plane|>1.0H = 95% fragments) is
  masked from training and gates; terrain and labels share the reconstruction (circularity),
  defended by an independent-terrain arm (CUT3R, different backbone) and by deployment tiers that
  use telemetry-plane terrain with no reconstruction at all.

## 4. Aerial & elevated-camera monocular 3D detection
*(from sweep A, 2026-08-27; venues verified except where flagged)*

### 4.1 Aerial monocular 3D detectors (all vehicles/people; none animals)
- **DVDET / AM3D (RA-L'23)** — the canonical first "aerial mono3D": image+BEV dual view,
  geo-deformable BEV warp, categorical ALTITUDE estimation; AM3D-Sim/Real (DJI M300 + LiDAR GT,
  40–80 m, vehicles). No terrain, no zoom.
- **CDrone / GroundMix (GCPR'24)** — the most oblique-diverse aerial 3D benchmark (synthetic;
  6.9–60.6 m, near-nadir→near-horizontal; 6 classes); detector = MonoCon-style with intrinsics-
  normalized "virtual depth" REGRESSION + ground-plane-consistent copy-paste. Closest detector in
  viewpoint spirit; still regresses depth. **Candidate for our transfer arm.**
- **MGF3D / LA3D (KBS'24)** — RTK-GPS altitude FUSED into depth for boats over LAKES — the
  existing telemetry-conditioned aerial mono3D, i.e., the flat-water degenerate case of our
  terrain conditioning. **Scopes our telemetry claim.**
- **DHD (NeurIPS'24)** — multi-drone collaborative; ground-prior BEV from inclined geometry (flat
  plane, sim). **DPETR ('25)** — PETR-style UAV queries (thin details, unverified).
- **MonoLAA/LAA3D ('25)** — 3D of AIRCRAFT from ground-based ZOOM cameras — the inverse geometry,
  notable as mono3D under variable focal length.

### 4.2 Elevated/roadside — the nearest methodological neighbors, and the exact contrast
- **Rope3D (CVPR'22), DAIR-V2X-I (CVPR'22)** — the roadside benchmarks (fixed mounts, LiDAR GT).
- **BEVHeight (CVPR'23)** — THE sharpest contrast for us: observes that from an elevated camera,
  per-pixel depth collapses with distance but height-above-ground is distance-invariant ⇒
  REGRESSES categorical height over an assumed FLAT ground with KNOWN camera height.
  Our sentence: *they regress height above an assumed plane; we regress nothing — depth is the
  intersection of a ray with a measured terrain field.* (+ BEVHeight++, CoBEV — height/depth
  fusion; venues partially unverified.)
- **MonoGAE (T-ITS'24)** — per-pixel refined ground-plane-EQUATION map as embeddings (still a
  plane, from calibration). **MonoUNI (NeurIPS'23)** — normalized depth removing pitch/focal
  diversity — the main prior treating focal ambiguity in elevated detection. **MOSE ('24)** — the
  closest roadside step toward non-planar ground: learns a scene-specific height CORRECTION to a
  virtual plane (learned, not measured; static camera).
- **SGV3D ('24)** — documents how badly roadside detectors overfit one camera/ground config —
  supporting evidence for the generalization framing.

### 4.3 Aerial 3D datasets (and the animal gap)
AM3D-Sim/Real (vehicles) · CDrone (synthetic, 6 classes) · UAV3D (NeurIPS'24 D&B; 500k synthetic
images, collaborative) · CoPerception-UAVs (sim) · LA3D (boats/lakes) · **DSC3D ('25)** — 175k+
real 6-DoF trajectories from drone footage via an OFFLINE SfM+MVS pipeline (the offline-pipeline
counterpoint to online detection; implicit scene mesh) · Archangel (2D humans, but ships
telemetry — precedent) · UAV-MM3D / LAA3D (drones/aircraft as TARGETS).
**Animal verdict (both sweeps concur): no aerial RGB dataset with 3D animal boxes as a detection
benchmark exists besides WildBox; the only other aerial animal 3D boxes anywhere are WildLIFT's
pipeline-generated pseudo-labels (2,581 frames, no detector, no AP).** Ground-based animal-3D
adjacents for completeness: WildDepth ('26 preprint; RGB+LiDAR, ground, no boxes), WildPose
(J. Exp. Biol.'25; telephoto+LiDAR ground rig — also evidence that "telephoto depth" is today
solved with LiDAR, not vision).

### 4.4 Unknown/variable focal length (the zoom regime)
Metric3D/v2 (canonical-camera transform) · UniDepth (CVPR'24) · **Depth Pro (ICLR'25** — focal
FROM the image**)** · Tame a Wild Camera (NeurIPS'23) · **GeoCalib (ECCV'24** — focal+gravity;
nobody found using it inside a 3D detector yet**)** · MonoIA (CVPR'26 — intrinsics-conditioned
detection, ego-view) · **AerialMetric (ECCV'26)** — UAV metric-depth benchmark showing severe
degradation of ground-trained metric depth from the air — **direct external corroboration of our
depth-is-the-error finding.** Not found despite searching: any 3D DETECTION work targeting
telephoto/digital-zoom depth degradation. Our fx-from-telemetry (exact, free) + contact-not-size
depth is unclaimed territory there.

## 5. Method landscape: ground priors, DETR-3D, open-vocab 3D, scale-from-context, orientation
*(from sweep B, 2026-08-27; ~20 targeted searches; unverifiable details flagged)*

### 5.1 Ground priors in mono3D — the novelty-critical lineage
- **Hoiem/Efros/Hebert, Putting Objects in Perspective (CVPR'06)** — the ancestor: contact + ground
  ⇒ depth, probabilistic, flat-ground, pre-deep.
- **Mono3D (CVPR'16)** — proposals sampled on a flat ground plane; prior outside any depth path.
- **Deep3DBox (CVPR'17)** — translation from 2D⊃3D projective constraints (geometric fit, no ground).
- **GUPNet/GUPNet++ (ICCV'21)** — depth = f·H3D/h2D with uncertainty; object-height geometry, no ground.
- **Ground-Aware Mono3D (RA-L'21)** — flat-ground per-pixel depth prior AS FEATURES; depth regressed.
- **MonoGround (CVPR'22)** — "local ground plane" = the GT box's own bottom face (flat, per-object);
  dense supervision + refinement; depth still predicted. NOT scene terrain despite the name.
- **Homography Loss (CVPR'22)** — flat-ground homography as a train-time consistency loss only.
- **GPENet (arXiv 2211.01556)** — *closest on the contact side*: predicts explicit ground-contact
  points + estimates a (tilt-corrected) plane from the horizon; back-projects. Single plane, and
  back-projection as deduction (gradient flow unverified).
- **MoGDE (NeurIPS'22) / GEDepth (ICCV'23) / MonoGAE (roadside) / MonoCD (CVPR'24) / YOLOBU (RA-L'24)**
  — ground-depth features/embeddings/complementary branches; GEDepth even predicts a ground-SLOPE
  residual (a step past flat) — but all ultimately regress depth.
- **CHARM3R (arXiv 2508.11185, '25)** — *the motivation gift*: shows regressed depth FAILS to
  extrapolate under camera-height change (exactly why driving-trained models die on drones);
  remedies by AVERAGING a regressed branch with analytic ray∩flat-ground depth. Flat plane, hybrid.
- **DEM ray-cast UAV geolocation (ESWA'24; Remote Sensing'25)** — TRUE height-field ray
  intersection… post-hoc, non-differentiable, on 2D detections, outside any network.
- **SC-Lane ('25)** — road height-field estimation for 3D lanes: the field is moving past flat
  planes, but not for object depth.

**Sweep verdict (quote-ready): every detector-internal ground prior found is a flat or tilted
PLANE; every true HEIGHT-FIELD intersection found is post-hoc and non-differentiable. No prior
work does differentiable ray–height-field intersection as the sole depth mechanism inside a
detector, driven by a predicted per-object contact point. TA-DETR's claim is the conjunction —
and must be stated as such, citing GPENet + CHARM3R + DEM-raycast as the nearest single-piece
holders.**

### 5.2 DETR-family camera-only 3D
MonoDETR (ICCV'23; depth-guided queries, regressed) → SSD-MonoDETR → **MonoDGP (CVPR'25**;
geometric depth + regressed error-correction, KITTI SOTA**)** → Mono3DV ('26, unverified detail).
**MonoDINO-DETR (IV'25, 2502.00315)** — DINOv2 backbone + DPT depth head + DETR: the closest prior
for our backbone/head shape; depth remains dense-regression hybrid. Multi-view lineage for context
only: DETR3D (CoRL'21), PETR/v2 (ECCV'22/ICCV'23). Cube R-CNN/Omni3D (CVPR'23) as the non-DETR
reference generalist (virtual depth = intrinsics normalization, not viewpoint normalization).

### 5.3 Open-vocab / foundation-model 3D — who we measured against, and why they fail here
- **OVMono3D-LIFT (3DV'26)** — GroundingDINO 2D + class-agnostic lift trained on Omni3D base
  classes. Novel CATEGORIES transfer; novel viewpoint-scale REGIMES do not (our measured 0.00
  zero-shot with GT 2D; 13.17 fine-tuned).
- **OVM3D-Det (NeurIPS'24)** — pseudo-LiDAR from metric-depth FMs + LLM size priors; inherits the
  FM's driving/indoor metric prior — no anchor in high-GSD savanna.
- **DetAny3D (ICCV'25)** — promptable 3D from SAM+depth-FM features, claims arbitrary intrinsics;
  fine-tunes WORSE than the older lift on wildlife (our measured 8.33/1.99) — plausibly fighting
  its pretrained metric prior. (+ DetAny4D '25/26, streaming.)
- **UniMODE (CVPR'24)** — unified BEV mono; BEV grids presume near-horizontal ground below the
  camera — degenerate oblique-aerial. (No "UniMODE-v2" exists; the extension is MM-UniMODE, same
  arXiv line — cite that.)
- **3D-MOOD (ICCV'25)** — open-set end-to-end, canonical image space; depth regressed in canonical
  space. **LocateAnything3D ('25/26)** — VLM next-token 3D ("chain-of-sight"); depth as token
  statistics. **WildDet3D ('26)** — 1M+ images/13.5K categories scaling bet; "wild" ≠ wildlife.
- **The four-part failure story for §1, now citable**: (a) CHARM3R: regression doesn't extrapolate
  camera height; (b) virtual-depth/canonical tricks normalize intrinsics, not viewpoint geometry;
  (c) depth-FM metric priors have no anchor in textureless high-GSD savanna; (d) category-size
  dictionaries break on juvenile/adult size variance within species. Terrain supplies exactly the
  missing anchor.

### 5.4 Scale from context (the herd module's neighbors)
Hoiem'06 and **Kar et al., Amodal Completion & Size Constancy (ICCV'15)** (multi-instance size
coupling, energy-based) · pedestrian-height autocalibration (population size prior fixes scale) ·
object-size priors in SLAM/BA (Frost T-RO'18; dual-quadric scale '22) · **ZeroDepth (ICCV'23)** and
**WorDepth (CVPR'24)** — amortized variational SCALE latents (per-scene, for dense depth) — the
mechanistic cousins. **Unclaimed conjunction: per-class latent scale shared across con-specific
instances, amortized posterior, inside a detector.**

### 5.5 Orientation
MultiBin (CVPR'17) and 3D-RCNN (CVPR'18) fix the allocentric-α convention · FCOS3D encodes yaw as
sin/cos + direction class (a 2π-space magnitude×sign, not mod-π axis) · Prokudin et al. (ECCV'18)
von Mises mixtures for pose uncertainty · the aerial oriented-box line **CSL (ECCV'20) / GWD
(ICML'21) / KLD (NeurIPS'21)** establishes mod-π AXIS as the well-posed overhead target — but
discards heading sign entirely · UAV livestock OBB works append θ with no sign/ambiguity handling.
**No detector found that factorizes heading into axial estimate × binary sign (absence
unverified — phrase as "to our knowledge").** Our extra twist is WHY the factorization is forced:
pseudo-label signs are arbitrary AND the standard 3D metrics are provably 180°-flip-blind, so
sign supervision must come from an independent channel (motion) — that argument appears nowhere.

### 5.6 The domain neighbor to cite and contrast
**WildLIFT** — oriented 3D boxes of large mammals from monocular drone VIDEO via multi-view scene
geometry: an OFFLINE reconstruction/annotation pipeline (it authored our labels), not a
single-image detector. The contrast sentence: WildLIFT shows aerial wildlife 3D is *annotatable*;
TA-DETR makes it *detectable* — one image, one forward pass, with the reconstruction demoted to a
training-time teacher and a per-segment terrain map. (Self-citation handling per §6.3.)

## 6. Drone wildlife monitoring: the 2D state of practice, and which tasks need 3D
*(from sweep C, 2026-08-27; agent verified the no-3D claim against the ~30-dataset
agentmorris/drone-wildlife-datasets registry item-by-item and per-paper self-descriptions)*

### 6.1 The established 2D pipeline (detection → counting → tracking → re-ID → behavior)
- **Drones as monitoring instrument:** Linchant et al. (Mammal Review 2015); Hodgson et al. (MEE
  2018 — drone counts 43–96% more accurate than ground counts); recent reviews Guo et al. (2025),
  Yaney-Keller et al. (Biol. Reviews 2025).
- **Detection/counting:** Kellenberger et al. (RSE 2018; TGRS 2019 — savanna UAV, class imbalance,
  active learning); Torney et al. (MEE 2019 — wildebeest within 1% of expert count); Delplanque
  et al. (RSEC 2022; **HerdNet**, ISPRS J. 2023 — dense-herd point counting).
- **Tracking / re-ID / behavior:** **BIRDSAI** (WACV 2020, aerial thermal); **BuckTales** (NeurIPS
  2024 D&B — 1.2M UAV annotations, 730 blackbuck IDs); **WildLive** (2025 — onboard 17.8 fps
  tracking, explicitly flying HIGHER to cut disturbance); **MMLA** (2025 — 811k boxes, 6 species);
  **KABR** (WACVW 2024 — drone behavior recognition, Mpala); kabr-tools (2025).
- **The one geometry-adjacent ecology pipeline:** Koger et al. (J. Animal Ecology 2023) — SfM
  TERRAIN + georeferenced 2D detections. 3D exists only for the terrain; animals stay 2D. (Cite
  carefully — closest external relative of our terrain idea, and still not 3D detection.)

**Every output in this literature is a 2D box, point, mask, track, ID, count, or behavior label.**

### 6.2 The tasks that need geometry (currently special-purpose photogrammetry, or unsolved)
- **Metric body size/condition:** the whale drone-photogrammetry line (Christiansen et al. MEE
  2019 → Bierlich et al. 2024, ~99% length accuracy; Bagchi et al. 2025); terrestrial ports exist
  but are nadir + per-species (Sumatran elephants, Sci. Reports 2023; Nile crocodiles 2023).
  A metric 3D box is the general-purpose first-order version of all of these.
- **Distance sampling:** camera-derived distances exist only as bolt-on monocular-depth pipelines
  for camera traps (Haucke et al. 2022; Henrich et al. 2024); no drone pipeline outputs per-animal
  distance — a 3D detection gives it for free.
- **Disturbance rules are metric:** approach thresholds are specified in metres (Mulero-Pázmány
  et al. 2017 meta-analysis; 50–80 m AGL guidance; species flight-initiation distances 5–170 m) —
  a drone cannot ENFORCE them autonomously without the animal's 3D position.
- **Viewpoint-seeking flight control — the killer motivation:** Sun, Berger-Wolf, Kline
  (CV4Animals @ CVPR 2026): a UAV steered to expose the zebra's LATERAL FLANK for re-ID using
  YOLO + coarse 2D pose classes — our exact downstream task, currently done without geometry.
  Plus disturbance-aware RL tracking (2026), WildWing autonomous behavior-filming UAS (MEE 2026).
- **Re-ID is side-gated:** ATRW treats left/right tiger flanks as distinct identities; a 2026
  jaguar study shows flip augmentation actively corrupts flank embeddings; Wildbook-line systems
  classify viewpoint before matching; giraffe systems index sides separately. "Which side is the
  camera seeing" — our heading output — is a first-class re-ID input.

### 6.3 Aerial 3D animals: verified empty (external to our own line)
3D animal pose/shape (SMAL CVPR 2017 → BARC/BITE → Animal3D ICCV 2023 → 2024–25 video lines) is
ground-level; NO third-party aerial animal-3D work surfaced. The only aerial-3D thread is our own
(ISPRS 2024 oblique SMAL fitting; WildLIFT offline 3D lifting; WildBox) — handle as
self-citations. **Confirmed: no existing public drone/aerial wildlife dataset provides 3D boxes or
metric heading** (registry + per-paper verification).

### 6.4 The citable chain for the intro
(a) drones are established instruments [Linchant'15, Hodgson'18] → (b) CV automation is
established and entirely 2D [Kellenberger'18 → HerdNet'23; BIRDSAI'20 → BuckTales'24 →
WildLive/MMLA'25; KABR'24] → (c) the geometry-needing tasks exist and are served by per-species
photogrammetry or not at all [whales'19–'24; distance bottleneck '22–'24; metric disturbance
rules '17–'25; flank-seeking flight control '26] → (d) nothing fills the gap: no aerial 3D
detector, no 3D wildlife dataset before WildBox [verified against the dataset registry].

## 7. Experimental comparison plan (the paper's tables, drafted)

**Primary (WildBox val, official evaluator + BEV + label-sane diagnostics + per-species orientation
grading vs train-transferred floors):**
rows = zero-shot OVMono3D (0.00) · fine-tuned OVMono3D-LIFT (13.17/8.68) · fine-tuned DetAny3D
(8.33/1.99) · trivial-floor rows (with stated ranking) · A0 geometric lift (gt2d AND gdino) ·
TA-DETR A1…A6 ablation ladder · (A7 DINOv3 swap).
Parity discipline: every learned row also reported under the frozen GroundingDINO-detection
swap-in protocol; 3 seeds on headline rows; track-clustered CIs on orientation.

**Secondary (transfer / external validity, wildlife intent preserved):**
- KABR-style held-out footage (same pipeline, unseen videos) — generalization within the domain.
- An aerial-vehicle 3D benchmark row (from sweep A's findings, e.g. CDrone/AM3D-class) — NOT to
  chase SOTA there, but as the metric-GT sanity check our pseudo-labeled primary cannot provide:
  does terrain-anchored depth hold where ground truth is real? (Terrain source there: flat plane
  or SfM — documents the method's portability.)
- Deployment-tier ablation: full terrain vs telemetry-plane-only inference (the edge story) —
  quantifies what the mechanism costs when no reconstruction exists.

**Honest failure-mode reporting (the "test until failure" commitment):** per-segment stratification
by reconstruction confidence + camera-rotation envelope (we already know 7 degenerate-confidence
segments — one in val — and that high rotation correlates with terrain failure); grazing-ray
fallback rate; junk-label sensitivity; giraffe (neck breaks the axis prior) reported, not hidden.

## 8. Positioning verdicts — the claims we can defend, exactly as scoped

Each claim below names its nearest priors; the paper must cite them IN the claim sentence.
Phrase every absence as "to our knowledge" — the sweeps are thorough, not exhaustive.

1. **Terrain-anchored depth (the headline method claim).** *To our knowledge, the first detector
   in which depth is obtained solely by DIFFERENTIABLE ray–height-field intersection at a
   predicted per-object ground-contact point — no depth regression anywhere.* Nearest priors,
   each holding one piece: GPENet (predicted contact + tilted PLANE back-projection), CHARM3R
   (analytic ray∩flat-ground, AVERAGED with a regressed branch), BEVHeight (regressed
   height-above-plane), MOSE (learned plane correction), GEDepth (predicted ground slope, depth
   estimation not detection), DEM-raycast geolocation (true height field, post-hoc,
   non-differentiable). The claim is the CONJUNCTION.
2. **The regime argument.** CHARM3R (regression fails across camera height) + AerialMetric
   (metric depth FMs degrade from the air) + our measured zero-shot 0.00-with-GT-2D give a
   three-legged, partly external case that the aerial failure is structural, not a data gap —
   and our zoom analysis (fx unidentifiable ⇒ apparent size uninformative) says WHY. A0's
   training-free NHD-z beating every fine-tuned model is the constructive proof.
3. **Dataset claim, scoped.** *WildBox is the first aerial wildlife benchmark with 3D boxes and a
   detection evaluation* (verified against the community dataset registry + per-paper checks).
   Not claimable: "first aerial animal 3D boxes ever" (WildLIFT's pipeline labels exist — our own
   line; self-citation) or metric GT (pseudo-labels, per-segment gauge — say so).
4. **Telemetry claim, scoped.** *First monocular 3D detector conditioned on drone telemetry
   jointly with a non-flat terrain field under unknown/variable focal length* — citing MGF3D
   (RTK altitude over flat water) as the degenerate-case precedent, Archangel for
   telemetry-shipping precedent, MonoUNI/MonoIA for intrinsics-diversity handling.
5. **Orientation claim.** *Axis(mod π) × sign factorization with independently sourced sign
   supervision, forced by two measured facts: pseudo-label signs are arbitrary AND the standard
   3D metrics are exactly 180°-flip-blind.* Nearest: aerial OBB losses (CSL/GWD/KLD — axis
   without sign), FCOS3D (sin/cos + direction over 2π), von Mises pose heads (Prokudin).
   The flip-blindness proof + the honest per-species floors grading protocol appear novel in
   their own right.
6. **Herd scale claim (conditional on A4 results).** *Per-class latent scale shared across
   con-specific instances with an amortized posterior inside a detector* — nearest: Kar et al.
   ICCV'15 (multi-instance size coupling), ZeroDepth/WorDepth (variational scale latents for
   dense depth), size-prior SLAM/calibration lines. Claim only if the 1/√N shrinkage prediction
   verifies.
7. **What we do NOT claim.** The backbone/head shape (frozen DINOv2 + deformable DETR) is
   assembled from known parts (MonoDINO-DETR is close) — claim the system, not the components.
   Not "first 3D from drone footage" (DSC3D, WildLIFT: offline pipelines). Not metric accuracy on
   WildBox (gauge). Not tracking, not temporal.
8. **The contrast sentence for WildLIFT (the domain neighbor / self-citation):** WildLIFT shows
   aerial wildlife 3D is ANNOTATABLE from video reconstruction; TA-DETR makes it DETECTABLE from
   one image in one forward pass, with the reconstruction demoted to training-time teacher +
   0.5 MB terrain map — and at deployment tiers, to telemetry alone.

### Follow-ups this recce surfaced (not yet scheduled)
- CDrone as the metric-GT transfer arm (§7): oblique-diverse, synthetic — check license/eval kit.
- CHARM3R + AerialMetric: obtain exact numbers before quoting (flagged unverified in sweeps).
- GeoCalib-inside-a-detector is unclaimed — our pitch-fallback (40 gimbal-less videos) could be
  stated as a minor contribution if wired in an ablation.
- Venue-unverified entries (CoBEV, IROAM, BEVHeight++, DPETR, MonoDDE alias, "Small or Far
  Away") — verify before the camera-ready bibliography.
