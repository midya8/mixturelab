# Method, frozen protocol and limits

## 1. Density, not classification

For a d-dimensional feature vector x, the supplied model is

`p_theta(x) = sum_j pi_j N(x; mu_j, C_j)`

with positive weights summing to one and covariance constraint `C_j >= a I`, a = 0.05 in squared standardized units. The diagonal family additionally requires diagonal C. We maximize the training mean log likelihood; held-out mean **negative** log likelihood (NLL) is smaller-is-better. Natural logarithms, nats per sample. A density has units and may exceed one.

The source first column is a cultivar target. The loader strips it and does not return labels. These latent components are not cultivar probabilities or quality scores. There is no supervised model, label-derived split, classification accuracy or causal interpretation.

## 2. E-step

Compute weighted component log densities with Cholesky triangular solves, then subtract the row maximum before exponentiating:

`b_ij = log pi_j - (d log(2pi) + log det C_j + (x_i-mu_j)^T C_j^-1 (x_i-mu_j))/2`

`ell_i = logsumexp_j b_ij`, `r_ij = exp(b_ij - ell_i)`.

Native NumPy uses Cholesky and `solve`, not an explicit covariance inverse. A separate JavaScript implementation builds its own scalar Cholesky factor and forward substitution. All eight exported models on all 178 source rows are compared against frozen native references, including every responsibility, not only argmax assignments.

## 3. The actual constrained M-step

For fixed r, set `N_j = sum_i r_ij`, `pi_j = N_j/n`, `mu_j = sum_i r_ij x_i/N_j`, and weighted scatter

`S_j = sum_i r_ij (x_i-mu_j)(x_i-mu_j)^T / N_j`.

For full covariance, minimizing `log det C + tr(S C^-1)` subject to `C >= a I` gives

`S = U diag(s) U^T`, `C = U diag(max(s,a)) U^T`.

For diagonal covariance, use `C_kk = max(S_kk,a)`, with zero off-diagonal entries. This is a covariance eigenvalue/variance constraint, not the common additive update `S + a I`.

### Why this is the constrained maximizer

Use precision `K = C^-1`, so the covariance subproblem maximizes the strictly concave function `log det K - tr(SK)` over `0 < K <= I/a`. For the spectral update, `M = C-S` is positive semidefinite, `I/a-K` is positive semidefinite, and `M(I/a-K)=0`. The stationarity equation is `K^-1-S-M=0`. These KKT conditions certify the global conditional subproblem optimum in exact mathematics. They do **not** certify the global mixture optimum.

Tests check these conditions numerically on 100 random PSD scatters in dimensions 1 through 5, compare against 400 random feasible competitors, and separately compare 40 rotated 2D cases against analytic eigenvalues/projectors without calling the implementation's eigensolver. Tests use explicit float tolerances. The report is not an exact-arithmetic proof.

A supplied rotated-scatter control with eigenvalues 0.01 and 2 contrasts spectral clipping at 0.05 with adding 0.05I, under this same fixed-scatter objective. It is not a whole-experiment comparison of regularization schemes. No hyperparameter is selected from that control.

## 4. The EM gain identity

Let `q_i(j)=r_ij(theta_old)`. Per-sample complete-data objective:

`Q(theta;q) = (1/n) sum_i,j q_i(j) log[pi_j N(x_i;mu_j,C_j)]`.

The exact mathematical identity is

`mean ell(theta_new) - mean ell(theta_old)`

`= Q(theta_new;q) - Q(theta_old;q)`

`  + (1/n) sum_i KL(q_i || r_i(theta_new))`.

At the exact constrained M-step the Q gain is nonnegative; KL is nonnegative. The code checks both terms and their sum numerically at every update. It computes log responsibilities from component log densities, not `log(0)` after probability underflow. A Q/KL/gain below -1e-9 or identity discrepancy above 1e-8 raises an error; no rollback, hidden reseed, smoothing or fabricated ascent is substituted. Complete histories record every observed mean objective, Q gain, posterior KL, identity error, minimum component mass and minimum covariance eigenvalue.

Stopping: mean observed-likelihood increment <= 1e-7, at most 500 EM updates. A zero-update budget is valid and labeled budget-exhausted. All 24 saved fits reach this increment tolerance. This is **not** a gradient/KKT stationary-point check, parameter error bound, finite-sample confidence statement or global optimum certificate. Small increments can occur near a saddle or a poor local solution.

If a component has zero numerical mass, the fit fails explicitly rather than reseeding. Numerically singular/nonfinite proposals also fail. No failure occurs in this recorded grid; the implementation does not claim that arbitrary inputs always fit.

## 5. Frozen public-data experiment

UCI Wine, Aeberhard & Forina (1992), 178 rows and 13 numeric chemical features, CC BY 4.0. Original source bytes remain bundled and SHA256-checked before loading:

- `wine.data`: `6be6b1203f3d51df0b553a70e57b8a723cd405683958204f96d23d7cd6aea659`
- `wine.names`: `f1b84f2ef845e0bdebf13e14fa7a213e56de4f1baa40c5974dbd1ee51c5ae710`

The protocol artifact was written before fitting and holds the literal row IDs. NumPy default RNG seed 20261108 makes a label-free permutation: first 106 training, next 36 validation, last 36 test. Every source feature row is unique, so there is no exact-feature-copy leakage. This is a new demonstration split, not an official UCI benchmark split or a stratified split.

1. Compute feature means and population standard deviations from **training only**. Every feature has positive variance. Reuse this transform unchanged for validation and test.
2. Fixed grid: diagonal/full families x k in {1,2,3,5} x seeds {11,23,37}: 24 fits. Same floor 0.05, same stopping rule, same maximum budget. Initial weights are uniform; centers are distinct randomly chosen training rows; covariance is the floored pooled training covariance. Initial means need not be distinct values for arbitrary input, though the source rows here are unique.
3. Fit every candidate using training only. For each family/k pair choose the restart with highest training likelihood, ties by lower seed.
4. Score those eight winners on validation and choose lowest validation NLL, ties by family name then k. There is no BIC tuning, floor search, test-driven seed choice or refit on combined train/validation.
5. Only after the choice, compute test scores. Retain every candidate's train, validation and test metrics. Showing all scores is transparency, not a second selection stage.

The report stores the full protocol plus its hash, source hash, scale transform, every initial/final model, full objective histories, final-transition before/after parameters, all candidate scores and the selected ID. The browser exports the eight train-selected models, all 24 summary rows, full histories for those eight models, public raw features and training PCA geometry.

The selected model is diagonal k=5, seed 11: validation NLL 13.45753, test NLL 15.34782. The one-Gaussian full-covariance control has test NLL 15.72686. The full k=5 training-selected fit has training NLL 10.20168 and test NLL 18.47913. More capacity improves its training fit here while losing on test. No statistical-significance claim is made from these 36 test observations, and no future tuning should treat this already-reported test set as untouched.

To report densities in original feature units, `log p_X(x) = log p_Z((x-mean)/scale) - sum_k log scale_k`. Original-unit NLL therefore adds the training log-scale Jacobian. The independent test transforms the entire fitted Gaussian back to source units and checks this identity.

## 6. Projection, interaction and uncertainty

A training-only SVD supplies two deterministic-sign PCA axes for drawing. It does not affect the 13D fit or posterior. A component's displayed mean and covariance are `P mu` and `P C P^T`. Ellipses use `-2 log(0.1)` as the squared Mahalanobis radius for a 90% probability contour of the **2D Gaussian marginal**. These are not 90% empirical coverage estimates, 13D assignment regions or confidence intervals for parameters. Point colors are argmax responsibilities, not source class labels.

All 13 source features are editable numerically and through sliders. Sliders show observed public-data ranges, not constraints learned from held-out data or an outlier detector. Arbitrary finite numeric edits can extend beyond those ranges. Viewport expansion affects the picture only. Invalid/nonfinite inputs leave an explicit not-updated error and block export; no silent normalization or substitution.

The far-input button sets every standardized feature to +12. In the recorded selected model this yields a very low density with a near-100% component posterior. This is a supplied counterexample to treating posterior concentration as calibrated in-distribution confidence, not a general out-of-distribution detector.

## 7. Ambiguity and collapse controls

- **Label switching:** permutation of components preserves p(x) while permuting responsibilities.
- **Duplicate components:** one Gaussian with weight one is the same density as two identical Gaussians weighted 0.2/0.8. The latter posterior is 0.2/0.8 everywhere. A density can be well specified without unique component assignments.
- **Unbounded unconstrained likelihood:** on supplied 1D points {-2,0,3}, a weight-0.4 component centered at 0 shrinks its sigma toward zero while a weight-0.6 broad component remains fixed. The density at 0 diverges, while the other points retain positive broad-component density. The mean log likelihood is unbounded on this path. This is a parameter path, not a fitted EM trajectory.
- **Fixed-floor bound:** with `C >= a I`, each Gaussian and hence its convex mixture have density <= `(2pi a)^(-d/2)`. The fixed constraint blocks the shown collapse, but not poor local solutions, misspecification or nonidentifiability. Scale and floor are part of the model, not numerical implementation trivia.

## 8. Reproduction and the audit's scope

The standalone audit recomputes source-space splits/scaling, every saved candidate's scores, covariance constraints, history arithmetic and endpoints, the final E/M transition, and both selection stages. It rejects incomplete grids, modified scores, bad hashes, inconsistent statuses and selected IDs. It does not replay every historical parameter state or authenticate a model's author. Fresh-refit tests and installed-wheel runs supply separate numerical reproduction evidence; floating eigensolver bits need not match across platforms. Real-Chrome tests also cover the actual controls, 1,424 numerical reference cases, edits, invalid inputs, extreme input, physical keyboard interaction, JSON copy failure/success and the actual download blob at 320/390/1280px.

## Sources

- Dempster, A. P., Laird, N. M. & Rubin, D. B. (1977), *Maximum Likelihood from Incomplete Data via the EM Algorithm*. Original EM framework, monotonic likelihood and incomplete-data formulation. https://onlinelibrary.wiley.com/doi/full/10.1111/j.2517-6161.1977.tb01600.x
- Ingrassia, S. & Rocci, R. (2007), *Constrained monotone EM algorithms for finite mixture of multivariate Gaussians*. Primary constrained-EM context and eigenvalue-constraint monotonicity. Repository abstract was read; this project does not claim to reproduce that paper's complete algorithm. The specific fixed-floor update is derived above. https://www.iris.unict.it/handle/20.500.11769/31964
- University of Waterloo, *Mixture Models and Expectation-Maximization*. Responsibilities and weighted Gaussian moment updates, Algorithm 14.19 and Example 14.21. https://cs.uwaterloo.ca/~y328yu/teaching/480/480-note-mog.pdf
- Taboga, M. (2021), *Gaussian mixture - Maximum likelihood estimation*. EM, initialization, local maxima and singularity discussion. Its practical recommendation to discard degenerate runs is not used as a substitute for our explicitly constrained M-step. https://statlect.com/fundamentals-of-statistics/Gaussian-mixture-maximum-likelihood
- UCI Wine dataset, provenance, variables and license. https://archive.ics.uci.edu/dataset/109/wine
- NumPy: symmetric eigenvectors/eigenvalues, Cholesky factorization and determinant conventions. https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigh.html ; https://numpy.org/doc/stable/reference/generated/numpy.linalg.cholesky.html ; https://numpy.org/doc/stable/reference/generated/numpy.linalg.slogdet.html

Standard algorithms, original code, explicit numerical limits. No endorsement or claim of algorithmic novelty.

## v0.3 conditional inference

[CONDITIONAL.md](CONDITIONAL.md) derives fixed-model marginal evidence, component Schur complements, centered within/between covariance and original-unit Jacobians. It records six source notes, empty-dimensional conventions, underflow and numerical refusal, direct scalar/precision-block controls and all 5,696 original model/row/mask factorization checks. The current complete suite has 45 Python and eleven Node tests; installed reproduction imports twelve modules and runs nine commands. No original fit, split, scores or geometry references change. Analytical masks on complete source data are not a natural-missingness or interval-coverage validation.
