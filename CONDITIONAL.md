# Conditioning a frozen mixture

Condition on any subset of observed coordinates in the saved Gaussian mixtures. The model parameters, data, split, restart choices, scores and geometry references stay fixed. The implementation uses standard Gaussian identities written from scratch, not a new estimator or a package implementation.

## Scope and input

`condition(model, observed, values)` takes distinct integer coordinate indices in the user's supplied order and the matching finite numeric values. The free dimensions are the sorted complement. There is no supplied hidden value, class label or target to enter inference. Empty observed and free sets are supported. The model stays fixed. Numeric booleans and strings are refused as observed values.

For the public experiment, standardized inputs use the original training-only center and scale. `condition_raw` converts only observed values into standardized coordinates, then transforms free means/covariances back to original units. Its observed marginal log density subtracts the log-scale Jacobian for the observed dimensions only. Conditional log density subtracts the free-coordinate Jacobian. These are densities in their stated coordinates, not probabilities comparable across dimensions or units.

The source Wine table has no naturally missing feature values. These masks are analytical queries. This is not a real missingness-mechanism model, multiple imputation with parameter uncertainty, causal intervention, physically plausible chemistry, calibrated uncertainty, empirical interval coverage, classification or new test-performance result.

## Component algebra

Partition a component's mean and covariance into observed `o` and free `m` blocks. For component k,

```
log e_k = log pi_k + log N(x_o; mu_ko, Sigma_koo)
log evidence = logsumexp_k(log e_k)
log w_k = log e_k - log evidence
m_k = mu_km + Sigma_kmo solve(Sigma_koo, x_o - mu_ko)
C_k = Sigma_kmm - Sigma_kmo solve(Sigma_koo, Sigma_kom)
```

The conditional is `sum_k w_k N(y_m; m_k, C_k)`. Each component's conditional covariance is independent of the observed values, but its mean and posterior weight need not be. The full mixture covariance can depend on observed values through both.

The implementation uses Cholesky factors and triangular/native linear solves, not an explicit covariance inverse. Symmetry averaging removes roundoff asymmetry only. No ridge, covariance floor, eigenvalue clipping, silent reseeding or fallback is added to the conditional. A nonpositive Cholesky pivot, nonfinite density or nonfinite moment is refused instead of repairing a different distribution into success. The source model's original validation/floor is retained; no conditional `Model` object is created with misleading fitting-floor metadata.

Weights are normalized in log space. Displayed floating weights can underflow to zero; their normalized finite log weights remain in the result and are used to evaluate conditional log density. This is finite-precision numerical computation, not exact arithmetic or a proof about the population.

With no observations, the zero-dimensional observed margin has unit mass and log evidence exactly zero by convention. The conditional is the original mixture, up to floating normalization. With every coordinate observed, the free set is empty: moments are empty arrays and a zero-dimensional target has log mass zero. This is not a continuous density over a nonempty target space.

## The two sources of conditional spread

```
mean = sum_k w_k m_k
within = sum_k w_k C_k
between = sum_k w_k (m_k - mean)(m_k - mean)^T
total covariance = within + between
```

The centered form avoids subtracting large raw second moments in production. Tests also compare with a separately summed raw-second-moment expression. Neither term measures uncertainty in fitted parameters. Zero cross-covariance inside each diagonal component does not imply independence across mixture coordinates: seeing one coordinate can reweight the shared latent component and change the distribution of another.

Two complete synthetic models expose limits:

1. **Mean in a density gap.** Equal components have target means -4/+4 and target variance 0.25; the observed coordinate has the same unit Gaussian in both. The conditional mean is 0. Log target density there is -32.225791..., compared with -0.918939... at a component mean. The mean is a squared-error summary, not a mode finder or likely example.
2. **Conditioning increases mixture variance.** Weights are 0.99/0.01. Observed means are 0/4 with unit variance; target means are 0/10 with variance 0.1. Observing `x=(8+log 99)/4` yields equal conditional weights. Target variance rises from 1.09 to 25.1. The component target variance remains 0.1; the increase is between-component spread. This is not a causal experiment or learned-data result.

## Independent checks and what agreement means

The complete project suite has **45 Python tests and eleven Node/Chrome tests**. Nine Python tests cover conditional inference:

- Direct scalar 2D Gaussian conditional means, variances and log densities, using explicit formulas rather than production helpers.
- One hundred fixed-seed six-dimensional Gaussian controls checked through the separate precision-block conditional identity, instead of the production covariance Schur complement. Native matrix arithmetic is shared, not an independent numerical library.
- All eight original selected-restart models and all 178 source rows, for four deterministic masks: none observed, all observed, alternating coordinates and the supplied-order indices `[12,1,5]`. All **5,696 model/row/mask cases** check joint log density = observed marginal log density + conditional target log density, normalized weights, positive conditional covariance and centered versus raw second-moment decomposition. These are algebra/source checks, not a new missing-feature benchmark.
- Empty-dimensional limits, observed-order permutation, original-unit means/covariances/Jacobians, component-label permutation, duplicate splitting, no model mutation, invalid/numerically extreme input, finite logweights after displayed-weight underflow and strict CLI refusal/duplicate-key exit codes.
- Both fixed synthetic controls above.

Two Node tests compare **160 complete native references**: all eight models, five fixed source rows and four masks. Means, every component and total covariance entry, within/between moments, log evidence/weights, original units and conditional target density are compared. Separate scalar Gaussian, refusal, no-mutation, empty-target and underflow controls are included. JavaScript uses scalar loops and the existing browser Cholesky helper, not the native NumPy implementation. Cross-language agreement is not an independent mathematical oracle.

The actual-Chrome conditional test applies all **160 reference cases through the UI**, checks complete moments and displayed context/card counts, omitted-value poisoning, invalid-input retention including a changed model selector, keyboard checkbox interaction, clipboard denial/success and actual Blob JSON. Responsive overflow and visible 44px numeric/select/button/summary controls are checked at 320/390/1280 widths. UI/module matching is not another independent Gaussian oracle; actual screenshots are inspected separately.

## Browser behavior

[Conditional inspector](https://midya8.github.io/mixturelab/conditional.html). The initial source row is the first original test row and the initial mask observes the first three coordinates. Neither is chosen for quality or confidence. All 178 public rows and eight unchanged fitted models remain available. Input edits are explicit. Invalid input is not applied and the last valid model/source context remains in the exported witness.

The UI accepts observed standardized magnitudes up to 1000 to keep its drawing interface within a stated numerical bound, not to define plausible chemistry. The native API instead refuses values when its calculations cease to be finite. Unchecked input fields are disabled and never sent to conditioning; changing one does not change the conditional.

Choose any free coordinate for a one-dimensional marginal plot. Other free coordinates are integrated out, not set to their means. Solid green is the conditional marginal, dashed gray is the same model's prior marginal, colored thin curves are weighted conditional components and the gold line is the conditional mean. The finite drawing range includes four component standard deviations on both prior and conditional sides. The polyline combines 301 uniform range samples with 81 local samples around each prior/conditional component mean to keep narrow peaks from being skipped by a broad uniform grid. It remains an approximate drawing, not a certified extremum search. It does not include all tails, compute a coverage probability or certify a mode. Exact input numbers and complete numerical moments are available as text/JSON; SVG drawing approximations do not enter inference.

## Reproduce

```sh
python -m mixturelab condition --input examples/conditional.json
python -m mixturelab conditional-controls
```

The condition JSON has exactly `observed` and `values`. Values are original-unit numbers, not full rows with hidden values. Optional `--model-id` selects an unchanged saved model; the default remains the original validation pick. Optional `--report` retains the existing external-report interface. Error paths exit 2 with no JSON result.

```python
from mixturelab.core import Model
from mixturelab.dataset import load
from mixturelab.experiment import saved
from mixturelab.conditional import condition, condition_raw, log_density
report = saved()
model = Model.from_dict(next(c for c in report["candidates"]
                            if c["id"] == report["selected_id"])["model"])
r = condition(model, [2, 0], [0.5, -0.2])
raw = condition_raw(model, [2, 0], [2.43, 14.23], report["mean"], report["scale"])
target = load()[0, raw["free"]].tolist()
ll = log_density(raw, [target])
```

`log_density` consumes a result from these functions. It is not an intended-source verifier, authenticator or security boundary for arbitrary supplied documents. A caller must keep its chosen model and coordinate/input context independently. Browser/native validation details and floating tolerances are recorded in code; exact bitwise portability is not promised.

Installed-wheel reproduction imports **twelve modules** and exercises **nine CLI commands**, preserving all 24 original fresh refits, original holdout/EM audits, derivatives and finite-budget traces. Reference export is separate and does not refit.

## Sources

Read November 8, 2026. Sources establish standard identities, API semantics and dataset provenance. No third-party estimator or source implementation is imported or copied.

1. Kevin P. Murphy, [Multivariate Gaussians](https://www.cs.ubc.ca/~murphyk/Teaching/CS340-Fall07/reading/gauss.pdf), UBC teaching notes. Section 5 and its 2D example: conditional means/covariances, marginals and dependence on observed values.
2. C. E. Rasmussen and C. K. I. Williams, [Gaussian Processes for Machine Learning, Appendix A](https://gaussianprocess.org/gpml/chapters/RWA.pdf), MIT Press 2006. A.1-A.2: density factorization, Bayes rule, covariance and precision Gaussian conditional identities. Not a Gaussian-process implementation here.
3. NumPy, [numpy.linalg.solve](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html): matrix/shape contract, full-rank requirement, singular failure and LAPACK. Current fetched page says NumPy 2.5; this project keeps its original 2.2.6 pin. The documentation's word "exact" does not establish real-arithmetic exactness.
4. proxymix, [Conditional of a Gaussian mixture](https://max578.github.io/proxymix/reference/gmm_conditionalise.html): independent package documentation corroborates component Schur complements and marginal-evidence reweighting. No dependency/code reuse or claim about that package's validation.
5. tgmm, [Noise, Missing Values and Conditioning](https://tgmm.readthedocs.io/en/latest/user-guide/noise-missing-conditioning/), marginals and conditionals section: mixture formulas, dimension ordering and distinction between regression mean and multimodal distribution. Incomplete-data EM in that page is separate from the fixed complete-data models used here.
6. UCI, [Wine dataset](https://archive.ics.uci.edu/dataset/109/wine): 178 rows, 13 features, source features marked as having no missing values. Original data, license and attribution remain in [METHOD.md](METHOD.md) and the [data ledger](mixturelab/data/README.md).
