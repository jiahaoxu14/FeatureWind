# Design Plan — **Feature Wind Map × Feature Compass (Wind Vane)**

**Research question.** *How well do animated feature-gradient vector fields drawn over 2D projections reconstruct the local directions of change induced by feature perturbations? And which metric‑aware dynamic renderers most faithfully encode those directions across DR methods and over time?*

---

## 0) Goal & Contributions

**Goal.** Make nonlinear DR plots (t‑SNE/UMAP/etc.) self‑explanatory by overlaying **dynamic feature-gradient winds** (global) and **feature compass** summaries (local), then measure how faithfully these visuals reflect high‑D behavior and how stably they evolve over time.

**Planned contributions.**
1. **Metric‑aware Feature Wind Map** for DR: aggregate per‑point feature gradients into a smooth vector field; animate with particle advection; color by locally dominant feature. (global view)
2. **Feature Compass (Wind Vane)** for precise point‑wise & aggregated sensitivity with convex‑hull needle filtering + elliptical baseplate + sensitivity meter. (local/detail view)
3. **Geometry‑ and time‑aware rendering**: pushforward of gradients via the DR Jacobian; pullback‑metric normalization of lengths; temporally coherent projections for streaming/model updates.
4. **Evaluation protocol** (algorithmic, not HCI): angular/topological fidelity, renderer comparisons (LIC/IBFV vs. glyphs), temporal coherence, and ablations across DR methods.

---

## 1) System Overview (Pipeline)

**Input.** High‑D dataset \(X\in\mathbb{R}^{n\times d}\); DR map \(g: \mathbb{R}^d\to\mathbb{R}^2\) (t‑SNE/UMAP/PCA/parametric variants).

**Steps.**
1. **Compute feature gradients** per point and feature:
   - \(v_i=g(x_i)\), feature‑wise projected gradients \(g_{i,k} = [\partial f_x/\partial x_{i,k},\; \partial f_y/\partial x_{i,k}]\).
   - Stack into Jacobian \(J\in\mathbb{R}^{(2n)\times d}\). Use autodiff (Torch) so the method is model‑agnostic.
2. **Metric‑aware pushforward** of directions:
   - For any high‑D direction \(u\) (e.g., \(\nabla_x s\) or basis \(e_k\)), draw on map the 2‑D vector \(J_g(x_i)\,u\).
   - Use **pullback metric** (from \(J_g\)) to normalize magnitudes/step sizes so a unit of motion corresponds to comparable high‑D change across the canvas.
3. **Global view — Feature Wind Map.**
   - Interpolate per‑point \(g_{i,k}\) to a grid → per‑feature fields \((U_k,V_k)\); sum top‑\(n\) features to get aggregate field \((U,V)\). Mask empty regions via KD‑tree distance. Color by **locally dominant feature** (Voronoi labeling). Animate with particle advection (spawn/respawn, short trails).
4. **Local view — Feature Compass.**
   - **Needles:** the set \(\{g_{i,k}\}\) as arrows; apply **convex‑hull filter** to keep significant needles.
   - **Baseplate ellipse:** covariance of needle endpoints → orientation (\(\theta\)), anisotropy (\(a,b\)), area ↔ sensitivity. **Sensitivity meter** maps ellipse area to a scalar level.
   - **Aggregated compass:** for a selection \(S\), use \(\hat g_k = |S|^{-1}\sum_{i\in S} g_{i,k}\).
5. **Linking & interaction.** Brushing on the wind map updates (a) an aggregated compass, (b) a sensitivity histogram, and (c) optional per‑feature single‑wind overlays.

---

## 2) Views & Encodings

### 2.1 Feature Wind Map (global)
- **Field construction:** grid resolution \(r\times r\); component‑wise interpolation of \(g_{i,k}\). Select top‑\(n\) features by mean \(\|g_{i,k}\|\).
- **Dominance map:** smooth magnitudes \(\tilde M_k\); label cells by \(\arg\max_k\tilde M_k\); color trails by that label.
- **Animation:** particle advection with step \(\Delta p = \alpha\,\hat s\), where \(\hat s\) is **metric‑normalized** velocity; short fading tails; low random respawn to avoid dead zones.
- **Scalability knobs:** top‑\(n\) features; sparse vs dense seeding; mask by data density.

### 2.2 Feature Compass (local/detail)
- **Compass needles:** color encodes sign (± change); opacity ties to raw feature value (optional). **Convex‑hull filter** removes small/duplicate directions.
- **Baseplate ellipse:** principal axes from covariance of needle endpoints → anisotropy & magnitude; replaces circular markers in the projection for at‑a‑glance orientation.
- **Sensitivity meter & histogram:** area‑based sensitivity with binned distribution for a selection.

### 2.3 Cross‑view linking
- Hover/brush in Wind Map ⇒ update **Aggregated Compass**; click a point ⇒ show **Point Compass**; brushing updates histogram & dominance legends.

---

## 3) Metric‑Aware Rendering Details

- **Direction:** draw \(v_z = J_g(x)\,u\) for \(u\in\{e_k\}\) or \(u=\nabla_x s(x)\).
- **Length normalization:** use local metric \(G = J_g J_g^\top\) (or singular values of \(J_g\)) to scale velocities so identical high‑D perturbations produce comparable on‑screen speeds.
- **Advection step:** integrate with RK2/RK4 using metric‑normalized field to avoid bias in stretched regions of the DR.
- **Renderer variants:**
  - **LIC** (Line Integral Convolution) — static texture snapshots of flow.
  - **IBFV** / **particle trails** — dynamic; our default (as in Wind Map). Compare all three.

---

## 4) Temporal Coherence (Streaming / Evolving Models)

- **Projection alignment:** use Aligned‑UMAP / dynamic t‑SNE or Procrustes alignment of successive projections; reuse particle seeds across frames; interpolate fields between time steps.
- **Metrics:** temporal angular stability of \(v_z\); topological stability (critical point tracks, separatrix persistence); projection incoherence (from dynamic DR literature).

---

## 5) Evaluation (Algorithmic)

**Datasets.** Synthetic two‑rings; Iris; Breast Cancer Wisconsin; optional: MNIST subset for scale. (Two‑rings/Iris/Breast Cancer used in both prior papers.)

**DR methods.** PCA, t‑SNE, UMAP, parametric t‑SNE/UMAP (for \(J_g\)); dynamic/aligned variants for time.

**Measures.**
1. **Angular fidelity (local directions of change).** For each point/feature, compare high‑D gradient direction projected via exact pushforward vs. rendered field direction on the grid. Report mean angular error; distribution over regions.
2. **Vector‑field topology.** Extract critical points (sources/sinks/saddles) & separatrices in 2‑D wind; compare counts/locations to pushforward field; compute overlap scores.
3. **Neighborhood recovery (context).** Correlate wind‑inferred local flow directions with DR quality measures (trustworthiness/continuity deltas) to show where winds “explain” distortions.
4. **Renderer comparison.** LIC vs IBFV vs glyphs: direction correlation, power‑spectrum agreement, streamline hit‑rate to true separatrices.
5. **Temporal stability.** Angular stability and topological persistence across time steps (aligned vs re‑fit projections).

**Ablations.** (i) naive vs **metric‑aware** normalization; (ii) with/without dominance coloring & KD‑mask; (iii) top‑\(n\) features; (iv) compass: with/without convex‑hull filter; (v) aggregated vs point compass.

---

## 6) Implementation Notes

- **Autodiff:** Torch to compute \(g_{i,k}\) & \(J_g\) (where available). Parametric DRs preferred for stable Jacobians.
- **Field grid:** bilinear interpolation; edge masking via KD‑tree distance threshold.
- **Performance:** GPU kernels for gradient eval; WebGL/WebGPU canvas for advection/LIC where possible. Cache per‑feature fields and recompute aggregates on selection.
- **Design defaults:** animation on with pause; top‑5 features; trails length 15–30; respawn ~5%/frame; baseplate scale via dataset span fraction.

---

## 7) Risks & Mitigations

- **DR instability across runs** → use fixed seeds, aligned projections, and report temporal metrics.
- **Overplotting in dense regions** → reduce seeding, show baseplates only; move needles to Compass view.
- **Jacobian availability** → favor parametric DR; finite‑difference fallback for non‑parametric runs (documented costs).

---

## 8) Deliverables & Timeline (sketch)

- **D1.** Prototype (wind + compass, linked; metric‑aware) with Iris/Two‑Rings.
- **D2.** Renderer study (LIC vs IBFV vs glyphs) + angular/topology metrics.
- **D3.** Temporal coherence benchmark (aligned DR) + stability metrics.
- **D4.** Paper package: figures, videos, and open‑source code.

**References to prior work used in this plan:** Feature Wind Map (vector‑field aggregation + particle advection + dominance coloring) and Feature Compass (needle set + convex‑hull filtering + ellipse baseplate + sensitivity meter + aggregated compass).
