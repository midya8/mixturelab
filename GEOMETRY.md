# Local density geometry and bounded ascent

This extension differentiates **inputs**, holding every v0.1 model parameter fixed. It neither retrains a mixture nor changes the split, scale, original experiment report or test scores. It implements standard Gaussian-mixture derivatives and a simple bounded sufficient-ascent search, not a new mode-finding algorithm.

[Live geometry inspector](https://midya8.github.io/mixturelab/geometry.html) · [Original experiment](METHOD.md)

## 1. Analytic derivatives

Let z be a standardized input, `r_j(z)` its component posterior, `K_j=C_j^-1`, and component score

`s_j(z) = -K_j (z-mu_j)`.

For `L(z)=log sum_j pi_j N(z;mu_j,C_j)`:

`g = grad L = sum_j r_j s_j`

`grad r_j = r_j (s_j-g)`

`H = Hessian L = -sum_j r_j K_j + sum_j r_j (s_j-g)(s_j-g)^T`.

The covariance term is positive semidefinite; the weighted precision term is negative definite. Their sum need not be negative definite. A mixture of log-concave Gaussian components need not have a concave log density. Covariance is computed with centered component scores, not by subtracting two large second-moment matrices.

Precision matrices are computed by linear solves, then symmetrized for tiny numerical differences. Native NumPy and scalar JavaScript use separate solves. These derivatives are numerical values for the actual represented floating parameters, not exact certificates. Finite inputs outside numerical range fail explicitly.

### Coordinates and units

Original feature vector x is standardized by fixed training means/scales: `z_i=(x_i-mean_i)/scale_i`. The original-unit derivatives satisfy

`g_x,i = g_z,i / scale_i`

`H_x,ij = H_z,ij / (scale_i scale_j)`

`J_x,ji = J_z,ji / scale_i`.

The constant log-density Jacobian has zero input derivatives. The displayed gradient bars use standardized coordinates; original-unit derivatives are labeled/exported separately. Gradient norms depend on coordinates. No causal, scale-invariant feature-importance or uncertainty interpretation is implied.

For a chosen coordinate displacement delta, the page compares the actual finite log-density change against `g_i delta` and `g_i delta + H_ii delta^2/2`. The quadratic is local, not a certified remainder bound or an integrated attribution. Large displacements can disagree sharply with it.

## 2. Every accepted ascent step

The search works in standardized 13D coordinates and holds the density fixed. At point z:

1. Compute g and its Euclidean norm. If norm <= 1e-5, stop with `gradient_norm_criterion`.
2. Direction `d=g/||g||`. Start with Euclidean step length t=1.
3. Accept `z_new=z+t d` only if the computed actual gain satisfies `L(z_new)-L(z) >= 1e-4 t ||g||`.
4. Otherwise halve t, at most 32 halvings. If no trial passes, return `line_search_exhausted`; do not fabricate a stationary result.
5. Stop after the requested number of accepted updates with `budget_exhausted` unless the final norm criterion is met.

The implementation compares the actual subtraction against the required gain, rather than a rounded `L_old+required` threshold that could accept no observable gain. Every accepted point, log density, norm, step length, halving count, actual gain and required gain is retained. Browser controls offer budgets 0, 1, 16 and 128; library/CLI allow 0 through 4096. A zero budget returns the initial state with no accepted updates. No random perturbation or second-order escape is hidden in the routine.

A near-zero gradient is **not** a mode certificate. The search makes no exhaustive-mode, global-optimum, basin-coverage, finite-error or convergence-rate claim. The source paper below studies more sophisticated mode search; this implementation is deliberately the narrower bounded normalized-gradient routine defined here. We do not import that paper's mode-count or exhaustive-search claims. The CMU backtracking rule is adapted to ascent by negating the objective; no convex global convergence theorem is applied to this nonconcave log density.

## 3. Zero-gradient minimum control

Take two equally weighted 1D Gaussians of variance one centered at -2 and +2. At z=0:

- Posteriors are 1/2 and 1/2.
- Component scores are -2 and +2.
- g=0.
- Weighted negative precision is -1 and score variance is 4, giving H=+3.

The routine immediately returns `gradient_norm_criterion`, zero updates, at a strict local minimum. A start at z=0.01 moves away toward a higher-density point under the same rule. No label is changed to "mode found." A supplied displacement of 2 also exposes a large error in the initial quadratic prediction.

A single Gaussian has H=-C^-1 everywhere. Duplicating one Gaussian and splitting its weight preserves its density gradient/Hessian while making the component-posterior Jacobian zero. Label permutation preserves g and H while permuting Jacobian rows. These are mathematical controls, not inferred chemistry or causal model ablations.

## 4. Preserved real-model audit

All original v0.1 models, data, protocol, report, browser reference and browser dataset remain byte-for-byte unchanged. The extension adds 40 geometry reference cases: all eight train-selected models x original rows {0,35,87,142,177}. Each case includes every 13D gradient coordinate, every 13x13 Hessian entry, every kx13 responsibility-Jacobian entry, both Hessian terms and original-unit transforms.

Eight saved traces start at the same public test-example row already displayed by v0.1 and run at a fixed budget of 128. These are an optimization audit of already-public fixed models/inputs, not new test-based model selection or a held-out model-quality benchmark. The diagonal models and one-component full model reach the numerical norm criterion; full models with k=2,3,5 remain `budget_exhausted` at 128. All remain visible. Input ascent can leave empirical ranges and does not guarantee chemically plausible feature combinations.

## 5. Independent checks

- Central likelihood differences check all 13 score coordinates on three independently drawn points for each of eight real fixed models: 312 coordinate probes.
- Responsibility differences check every Jacobian column at those same points.
- Score differences check every Hessian column; 50 independent directional second differences of scalar likelihood separately check curvature, without using the analytic gradient as their reference.
- Translation/rotation, label permutation, normalization of Jacobian columns, PSD of the posterior score covariance, original-unit affine transforms and single-Gaussian closed forms are checked.
- All 40 native/scalar-JavaScript references check all entries; eight trace references compare finite-budget statuses, counts and endpoints, with numerical tolerances rather than bitwise portability promises.
- Every accepted trace step is checked for the actual sufficient-ascent inequality, step length and finite budget. The zero-gradient strict minimum, zero budget, invalid values and duplicate controls are explicit.
- Actual Chrome checks numerical witnesses on the page, model/input changes, trace reset, all budgets, keyboard input, invalid export blocking, copy failure/success, the downloaded JSON blob and expanded mobile/desktop layouts.

The stored reference generator never refits or rewrites the original experiment. `python -m mixturelab.geometry_export` recreates only the two new geometry artifacts.

## 6. CLI

Input JSON uses the same strict `{"samples":[[13 original feature values]]}` format as v0.1, with exactly one row for these commands:

```sh
mixturelab geometry --input input.json
mixturelab ascend --input input.json --budget 128
mixturelab ascend --input input.json --model-id full-5-23 --budget 1
```

Outputs retain the selected fixed model ID, local derivatives and, for ascent, all points and explicit status. These outputs contain no origin authentication or parameter-training claims.

## Sources and scope

- Carreira-Perpinan, M. A. (2000), *Mode-finding for mixtures of Gaussian distributions*. The primary paper's equations (4)-(7) give density and log-density input derivatives; it explicitly distinguishes zero-gradient minima/saddles from maxima. The fixed-model derivative formulas here are equivalent to its formulas, with a centered covariance expression for numerical implementation. Our bounded normalized-gradient search is not a reproduction of its gradient-quadratic/exhaustive strategy. https://faculty.ucmerced.edu/mcarreira-perpinan/papers/pami00.pdf
- Tibshirani, R., CMU convex optimization lecture, *Gradient descent*. Backtracking sufficient-decrease rule and halving; used only for the corresponding sufficient-ascent condition, not a convex convergence theorem for mixtures. https://www.stat.cmu.edu/~ryantibs/convexopt-F16/lectures/grad-descent.pdf
- *Gradients of Softmax and Logsumexp* (2024). The elementwise derivatives and stable max-shift explanation are useful cross-checks. Its displayed matrix formula `I - ss^T` is incorrect for the ordinary softmax Jacobian; it should be `diag(s) - ss^T`. We use the elementwise chain rule above, not that erroneous matrix formula. https://indii.org/blog/gradients-of-softmax-and-logsumexp/
- NumPy official documentation for numerical Euclidean norms and symmetric curvature eigenvalues in native checks. https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html ; https://numpy.org/doc/stable/reference/generated/numpy.linalg.eigvalsh.html
- Original data/EM/constrained-M-step sources remain in [METHOD.md](METHOD.md), including UCI Wine provenance and CC BY 4.0 license. https://archive.ics.uci.edu/dataset/109/wine

Established formulas and search rule, original implementation. No new algorithm, global mode, exact certificate, causal importance, calibrated confidence or chemistry-validity claim.
