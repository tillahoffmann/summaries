---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
from operator import sub
import pathlib
import pickle
from scipy import optimize, spatial, stats
import summaries
from summaries import benchmark
import torch as th
from tqdm.notebook import tqdm
mpl.style.use("scrartcl.mplstyle")


WORKSPACE_ROOT = pathlib.Path("workspace")
FIGURE_ROOT = WORKSPACE_ROOT / "figures"
FIGURE_ROOT.mkdir(exist_ok=True)
```

## Composite figure illustrating challenges associated with entropy and mean-squared error minimization.

```{code-cell} ipython3
def get_aspect(ax):
    """
    Get the actual aspect ratio of the figure so we can plot arrows orthogonal to curves.
    """
    # Total figure size
    figW, figH = ax.get_figure().get_size_inches()
    # Axis size on figure
    _, _, w, h = ax.get_position().bounds
    # Ratio of display units
    disp_ratio = (figH * h) / (figW * w)
    # Ratio of data units
    # Negative over negative because of the order of subtraction
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())

    return disp_ratio / data_ratio


def evaluate_entropy(a, b):
    "Evaluate the entropy of a gamma distribution."
    return stats.gamma(a, scale=1 / b).entropy()


def evaluate_posterior_params(a, b, n, second_moment):
    "Evaluate the parameters of a gamma posterior distribution (normal likelihood with known mean)."
    return a + n / 2, b + n * second_moment / 2


def evaluate_posterior_entropy(a, b, n, second_moment):
    "Evaluate the entropy of a gamma posterior distribution."
    return evaluate_entropy(*evaluate_posterior_params(a, b, n, second_moment))


def evaluate_entropy_gain(a, b, n, second_moment):
    "Evaluate the gain in entropy in light of data."
    return evaluate_posterior_entropy(a, b, n, second_moment) - evaluate_entropy(a, b)


n = 4  # Number of observations.
b = 1  # Scale parameter of the gamma distribution.

# Build a grid over the shape parameter and the realized second moment to evaluate the entropy gain.
a = np.linspace(1, 4, 100)
second_moment = np.linspace(.2, .725, 101)
aa, ss = np.meshgrid(a, second_moment)
entropy_gain = evaluate_entropy_gain(aa, b, n, ss) 

fig, axes = plt.subplots(1, 2)

# Plot entropy gain with "centered" colorbar.
ax = axes[0]
vmax = np.abs(entropy_gain).max()
mappable = ax.pcolormesh(a, second_moment, entropy_gain, vmax=vmax, vmin=-vmax, 
                         cmap='coolwarm', rasterized=True)
cs = ax.contour(a, second_moment, entropy_gain, levels=[0], colors='k', linestyles=':')

# Plot the expected second moment.
ax.plot(a, 1 / a, color='k', ls='--', label='Expected sec-\nond moment 'r'$\left\langle t\right\rangle$')

# Consider a particular example.
a0 = 1.5
sm0 = 0.3
pts = ax.scatter(a0, sm0, color='k', marker='o', zorder=9, label='Example point')
pts.set_edgecolor('w')

arrowprops = {
    'arrowstyle': '-|>',
    'connectionstyle': 'arc3,rad=-.3',
    'patchB': pts,
    'facecolor': 'k',
}
bbox = {
    'boxstyle': 'round',
    'edgecolor': '0.8',
    'facecolor': '1.0',
    'alpha': 0.8,
}

handles, labels = ax.get_legend_handles_labels()
handles = [cs.legend_elements()[0][0]] + handles
labels = [r'$\Delta=0$'] + labels
ax.legend(handles, labels, loc='upper right')

ax.set_xlabel(r'Prior mean $a$')
ax.set_ylabel(r'Second moment $t=n^{-1}\sum_{i=1}^n x_i^2$')
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
cb = fig.colorbar(mappable, location='top', ax=ax)
cb.set_label('Entropy gain\n'r'$\Delta=H\{p(\theta\mid x)\} - H\{p(\theta)\}$')
ax.set_xlim(a[0], a[-1])
ax.set_ylim(second_moment[0], second_moment[-1])


# Show the posterior if we use the absolute value of \theta as the precision.
ax = axes[1]
ax.set_ylabel(r'Posterior $p(\theta\mid x)$')
ax.set_xlabel(r'Parameter $\theta$')

ax.axvline(1 / sm0, color='k', ls='--')
a1, b1 = evaluate_posterior_params(a0, b, n, sm0)
posterior = stats.gamma(a1, scale=1 / b1)
xmax = posterior.ppf(0.99)
lin = np.linspace(-xmax, xmax, 101)
# Posterior needs a factor of 0.5 to be normalized because we have the left and right branch.
ax.plot(lin, posterior.pdf(np.abs(lin)) / 2)
ax.ticklabel_format(axis='y', scilimits=(0, 0), useMathText=True)
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))

summaries.label_axes(axes)
fig.tight_layout()

# Draw arrows. This needs to happen after tight layout to get the right aspect.
ax = axes[0]
# First get the curve, then find the index of a vertex close to the reference position x0.
path, = cs.collections[0].get_paths()
x0 = 2.05
i = np.argmax(path.vertices[:, 0] < x0)
# Compute the normals to the curve at the position we've identified.
x, y = path.vertices.T
aspect = get_aspect(ax)
dy = (y[i + 1] - y[i]) * aspect ** 2
dx = x[i + 1] - x[i]
scale = .3 / np.sqrt(dx ** 2 + dy ** 2)

# Draw the arrows pointing to the increasing and decreasing regions for black and white printing.
arrowprops = {'arrowstyle': '<|-', 'facecolor': 'k'}
pt = (x[i] - scale * dy, y[i] + scale * dx)
ax.annotate('', (x[i], y[i]), pt, arrowprops=arrowprops)
ax.text(*pt, r"$\Delta>0$", ha='right', va='center')

pt = (x[i] + scale * dy, y[i] - scale * dx)
ax.annotate('', (x[i], y[i]), pt, arrowprops=arrowprops)
ax.text(*pt, r"$\Delta<0$", ha='left', va='center')

fig.savefig(FIGURE_ROOT / 'tough-nuts.pdf')
```

```{code-cell} ipython3
# Generate summaries for the manuscript.
lines = []

# Sample from the prior predictive distribution to generate samples.
precision_samples = stats.gamma(a0, scale=1 / b).rvs(100_000)
x_samples = np.random.normal(0, 1, (precision_samples.size, n)) / np.sqrt(precision_samples[:, None])
sm_samples = np.mean(x_samples ** 2, axis=1)

# Evaluate the prior and posterior entropy for the example point.
lines.extend([
    f"prior entropy: {evaluate_entropy(a0, b):.2f}",
    f"posterior entropy: {evaluate_posterior_entropy(a0, 1, n, sm0):.2f}",
])

# Estimate the fraction of simulated second moments that are smaller than the example we picked.
cdf = np.mean(sm_samples < sm0)
lines.append(f"cdf(second moment < {sm0}): {cdf:.2f}")

# Estimate the posterior entropies for all samples to get the EPE ...
ap, bp = evaluate_posterior_params(a0, b, n, sm_samples)
entropies = evaluate_posterior_entropy(a0, b, n, sm_samples)
lines.append(f"epe: {entropies.mean():.3f} +- {entropies.std() / np.sqrt(entropies.size - 1):.3f}")

# ... and fraction of entropy increases.
deltas = entropies - evaluate_entropy(a0, b)
p = np.mean(deltas > 0)
lines.append(f'increase: {p:.3f} +- {p * (1 - p) / np.sqrt(entropies.size - 1):.3f}')

print('\n'.join(lines))
```

```{code-cell} ipython3
# Run a posterior p-value calibration check: the cdf of the true values should  be uniform under the 
# posterior.
plt.hist(stats.gamma(ap, scale=1 / bp).cdf(precision_samples), density=True)
pass
```

## Example showing that the right summary statistics depend on the prior.

```{code-cell} ipython3
def generate_data(m: int, n: int, entropy_method: str, scale: float) -> dict:
    """
    Generate synthetic data that exemplifies the sensitivity of mutual information to prior choice.

    Args:
        m: Number of independent samples for estimating the entropy.
        n: Number of observations per sample.
        entropy_method: Nearest neighbor method for estimatingn the mutual information.
        scale: Scale of each prior distribution.
    """
    mus = {'left': -1, 'right': 1}
    results = {}
    for key, mu in mus.items():
        # Sample from the prior and likelihood.
        theta = np.random.normal(mu, scale, m)
        left = np.random.normal(0, np.exp(theta[:, None] / 2), (m, n))
        right = np.random.normal(theta[:, None], 1, (m, n))
        x = np.where(theta[:, None] < 0, left, right)

        # Evaluate the summary statistics.
        mean = x.mean(axis=-1)
        log_var = np.log(x.var(axis=-1))

        # Store the results in a dictionary for later plotting.
        results[key] = {
            'mu': mu,
            'theta': theta,
            'x': x,
            'mean': mean,
            'log_var': log_var,
            'mi_mean': summaries.estimate_mutual_information(theta, mean, method=entropy_method),
            'mi_log_var': summaries.estimate_mutual_information(theta, log_var, method=entropy_method),
        }

    return results


np.random.seed(0)

m = 100_000  # Number of independent samples for estimating the entropy.
n = 100  # Number of observations per sample.
entropy_method = 'singh'  # Nearest neighbor method for estimatingn the mutual information.
scale = 0.25  # Scale of each prior distribution.
num_points = 200  # Number of points in the figure (we sample more for MI estimates).

results = generate_data(m, n, entropy_method, scale)

fig, axes = plt.subplots(2, 2, sharex=True)

# Show the two priors.
ax = axes[1, 0]
for result in results.values():
    mu = result['mu']
    lin = mu + np.linspace(-1, 1, 100) * 3 * scale
    prior = stats.norm(mu, scale)
    label = fr'$\theta\sim\mathrm{{Normal}}\left({mu}, 0.1\right)$'
    line, = ax.plot(lin, prior.pdf(lin), label=label)
    ax.axvline(mu, color=line.get_color(), ls='--')
ax.set_ylabel(r'Prior density $p(\theta)$')

# Show the likelihood parameters as a function of the parameter.
ax = axes[0, 0]
lin = np.linspace(-1, 1, 100) * (1 + 3 * scale)
ax.plot(lin, np.maximum(0, lin), label=r'Location', color='k')
ax.plot(lin, np.minimum(np.exp(lin / 2), 1), label=r'Scale', color='k', ls='--')
ax.set_ylabel('Likelihood parameters')

# Plot the scatter of summaries against parameter value for both priors.
step = m // num_points  # Only plot `num_points` for better visualisation.
for key, result in results.items():
    for ax, s in zip(axes[:, 1], ['mean', 'log_var']):
        mi = result[f"mi_{s}"].mean()
        # Very close to zero, we may end up with negative results. We manually fix that to avoid
        # "-0.00" instead of "0.00" in the figure. This is a hack but consistent with the sklearn
        # implementation for mutual information (see https://bit.ly/3NXRn5r).
        if abs(mi) < 1e-3:
            mi = abs(mi)
        ax.scatter(result['theta'][::step], result[s][::step], marker='.', alpha=.5,
                    label=fr'${key.title()}$ ($\hat{{I}}={mi:.2f}$)')

# Set axes labels and label each panel.
axes[0, 1].set_ylabel(r'$\bar x$')
axes[1, 1].set_ylabel(r'$\log\mathrm{var}\,x$')
[ax.set_xlabel(r'Parameter $\theta$') for ax in axes[1]]

summaries.label_axes(axes[0], loc='bottom right')
summaries.label_axes(axes[1], loc='top left', label_offset=2)

axes[0, 0].legend()
[ax.legend(handletextpad=0, loc=loc)
    for ax, loc in zip(axes[:, 1], ['upper left', 'lower right'])]

fig.tight_layout()
fig.savefig(FIGURE_ROOT / 'mi-prior-dependence.pdf')
```

## Figure for the benchmark problem.

```{code-cell} ipython3
# Load all the data for the experiments on the benchmark problem with the small dataset, i.e. few
# observations per example.
with open(WORKSPACE_ROOT / 'benchmark/small/data/test.pkl', 'rb') as fp:
    test_data = pickle.load(fp)
    test_samples = test_data['samples']
    test_features = benchmark.preprocess_candidate_features(test_samples['x'])
    
methods = ['fearnhead', 'mdn_compressor', 'nunes', 'stan', 'mdn', 'naive', 'regressor']
results_by_methods = {}
for method in methods:
    with open(WORKSPACE_ROOT / f'benchmark/small/samples/{method}.pkl', 'rb') as fp:
        results_by_methods[method] = pickle.load(fp)
        
mdn = th.load(WORKSPACE_ROOT / 'benchmark/small/mdn.pt')
```

```{code-cell} ipython3
# Get information about the particular data point. We choose point "3" here which will always give
# the same results because the generated data are seeded.
idx = 3
xs0 = th.as_tensor(test_samples['x'][idx])
x0 = xs0[:, 0]
theta0 = th.as_tensor(test_samples['theta'][idx, 0])

fig, axes = plt.subplots(2, 2, sharex='col')
axes[1, 0].sharey(axes[1, 1])

# Plot the likelihood.
ax = axes[0, 0]
ax.scatter(x0, np.zeros_like(x0), color='k', marker='|')
dist = benchmark.evaluate_gaussian_mixture_distribution(theta0)
xlin = th.linspace(-3, 3, 249)
ax.plot(xlin, dist.log_prob(xlin).exp())
ax.set_ylabel(r'Likelihood $p(x\mid\theta_0)$')

# Plot various posteriors.
ax = axes[0, 1]
samples = results_by_methods['mdn_compressor']['posterior_samples'][idx]
tmax = np.abs(samples).max()
ax.hist(samples, density=True, bins=19, range=(-tmax, tmax), label='ABC', color='silver')

# Show the full posterior.
tmax += .5
tlin = th.linspace(-tmax, tmax, 251)
lp = benchmark.evaluate_log_joint(xs0, tlin)
ax.plot(tlin, lp.exp(), color='k', label='exact', ls='--')
ax.axvline(theta0, color='k', ls='--')

# Show the mixture density estimate.
dist = mdn(xs0)
with th.no_grad():
    lp = dist.log_prob(tlin[:, None])
ax.plot(tlin, lp.exp(), color='C0', label='MDN')
ax.set_ylabel(r'Posterior $p(\theta\mid x_0)$')
ax.legend(ncol=2, loc='upper right', columnspacing=.5, labelspacing=.1)
ax.set_ylim(top=0.6)

# Show the compression function. We flip the sign (which is irrelevant for the algorithm but nicer 
# for plotting).
ax = axes[1, 0]
with th.no_grad():
    y0 = -mdn.compressor.layers(xs0)
pts = ax.scatter(x0, y0, marker='o', zorder=9)
pts.set_edgecolor('w')
ax.axhline(y0.mean(), ls='--', color='k')
    
xlin_with_zero_noise = th.hstack([xlin[:, None], th.zeros((xlin.shape[0], 2))])
with th.no_grad():
    y = mdn.compressor(xlin_with_zero_noise[:, None]).ravel()
ax.plot(xlin, -y, label='learned summary', zorder=3)
ax.set_xlabel(r'Data $x$')

# Run a polynomial regression weighted by the prior to show that the initial features can generate 
# similar output in principle.
def predict(x, *coefs):
    return np.dot(x[:, None] ** [0, 2, 4, 6, 8], coefs)

result = optimize.curve_fit(predict, xlin.numpy(), y.numpy(), np.random.normal(0, 1, 5),
                            stats.norm(0, 1).pdf(xlin) ** -.5)
coefs, _ = result
ax.plot(xlin, -predict(xlin.numpy(), *coefs), label='polynomial fit', ls='--')
ax.legend(loc='upper right')

# Show the posterior density as a function of summary statistic.
ax = axes[1, 1]

# Fudge the mixture density network to return the summaries we want. I.e. rather than applying the
# compressor to the data to get summaries, we instead "inject" a grid of summaries so we can 
# evaluate the density as a function of the summaries on a grid.
sumlin = th.linspace(-3.9, y.max() + .25, 250)
compressor_forward = mdn.compressor.forward
try:
    mdn.compressor.forward = lambda *args: sumlin[:, None]
    with th.no_grad():
        dist = mdn(sumlin[:, None, None])
        lp = dist.log_prob(tlin[:, None, None])
finally:
    # Make sure to restore the mixture density network to its original state.
    mdn.compressor.forward = compressor_forward

ax.imshow(lp.exp().T.numpy()[::-1], extent=(tlin[0], tlin[-1], -sumlin[-1], -sumlin[0]), 
          aspect='auto')
ax.axhline(y0.mean(), color='w', ls='--')
ax.axvline(theta0, color='w', ls='--')
ax.set_xlabel(r'Parameter $\theta$')
summaries.label_axes(axes.ravel()[:3])
summaries.label_axes(axes[1, 1], label_offset=3, color='w')

[ax.set_ylabel(r'Summary statistic $t(x)$') for ax in axes[1]]
[ax.text(.99, y0.mean() + 0.1, '$t(x_0)$', transform=ax.get_yaxis_transform(), 
         color=color, ha='right') for ax, color in zip(axes[1], 'kw')]
[ax.text(theta0 + 0.1, .025, r'$\theta_0$', transform=ax.get_xaxis_transform(), 
         color=color) for ax, color in zip(axes[:, 1], 'kw')]

fig.tight_layout()
fig.savefig(FIGURE_ROOT / 'benchmark.pdf')

print(fr"\theta_0 = {theta0:.3f}")
```

```{code-cell} ipython3
# List the expected entropies.
entropies_by_method = {}
for method, result in tqdm(results_by_methods.items()):
    entropies = np.asarray([summaries.estimate_entropy(x) for x in result['posterior_samples']])
    entropies_by_method[method] = entropies
    print(f'{method}: {entropies.mean():.2f}+-{entropies.std() / np.sqrt(entropies.size - 1):.2f}')
```

### Because Fearnhead's regression method is dominated by noise, we need to bootstrap the regression; we'll also compare that with random projections.

```{code-cell} ipython3
with open(WORKSPACE_ROOT / 'benchmark/small/fearnhead_random_entropies.pkl', 'rb') as fp:
    entropies = pickle.load(fp)['entropies']
    
for key, values in entropies.items():
    value = values.mean(axis=1)
    print(f'{key}: {value.mean():.2f}+-{value.std() / np.sqrt(value.size - 1):.2f}')

# Also show the prior entropy. This should be larger than the entropies reported above because even
# random projections contain some information.
th.distributions.Normal(0, 1).entropy()
```

## Larger dataset 

We can also learn summaries on a small dataset and then apply them to a dataset with larger number of observations per example.

```{code-cell} ipython3
large_results_by_method = {}
for method in ['stan', 'mdn_compressor', 'mdn_compressor_small']:
    with open(WORKSPACE_ROOT / f'benchmark/large/samples/{method}.pkl', 'rb') as fp:
        large_results_by_method[method] = pickle.load(fp)
        
for method, result in large_results_by_method.items():
    entropies = np.asarray([summaries.estimate_entropy(x) for x in result['posterior_samples']])
    print(f'{method}: {entropies.mean():.3f}+-{entropies.std() / np.sqrt(entropies.size - 1):.3f}')
```

## Coalescent problem in population genetics.

```{code-cell} ipython3
# Load all the posterior samples by method.
methods = ['fearnhead', 'mdn_compressor', 'nunes', 'mdn', 'naive', 'regressor']
results_by_methods = {}
for method in methods:
    with open(WORKSPACE_ROOT / f'coal/samples/{method}.pkl', 'rb') as fp:
        results_by_methods[method] = pickle.load(fp)
```

```{code-cell} ipython3
# List the expected entropies and rmses.
entropies_by_method = {}
rmses_by_method = {}
for method, result in tqdm(results_by_methods.items()):
    entropies = np.asarray([summaries.estimate_entropy(x) for x in result['posterior_samples']])
    entropies_by_method[method] = entropies
    
    rmses = np.sqrt([
        np.square(result['posterior_samples'] - result['theta'][:, None]).sum(axis=-1).mean(axis=-1)])
    rmses_by_method[method] = rmses
    
    print(
        f'{method}: {entropies.mean():.2f}+-{entropies.std() / np.sqrt(entropies.size - 1):.2f}\t'
        f'{rmses.mean():.2f}+-{rmses.std() / np.sqrt(rmses.size - 1):.2f}\t'
    )
    
# Show what we would expect if we sampled from the prior. Factors of two account for the model 
# having two parameters.
prior_entropy = 2 * th.distributions.Uniform(0, 10).entropy()
prior_rmse = summaries.evaluate_rmse_uniform(10) * np.sqrt(2)
print('prior', prior_entropy, prior_rmse)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2)

labels = {
    'fearnhead': 'Linear regression',
    'regressor': 'Nonlinear regression',
    'mdn': 'MDN',
    'mdn_compressor': 'MDN-compressed ABC',
    'nunes': 'Minimum entropy',
    'naive': 'Standard ABC',
}
# Show different markers and different colors for black and white print/colorblind people.
markers = {
    'fearnhead': 'o',
    'regressor': 's',
    'mdn': '^',
    'mdn_compressor': 'v',
    'nunes': 'X',
    'naive': 'D',
}

# Plot the results as a scatter plot.
ax = axes[1]
for method, entropies in entropies_by_method.items():
    rmses = rmses_by_method[method]
    xerr = entropies.std() / np.sqrt(entropies.size - 1)
    yerr = rmses.std() / np.sqrt(rmses.size - 1)
    errorbar = ax.errorbar(entropies.mean(), rmses.mean(), yerr, xerr, marker=markers[method], ls='none', 
                           label=labels[method], zorder=9 if method == 'mdn_compressor' else None)
    pts, *_ = errorbar.get_children()
    pts.set_markeredgecolor('w')
    # Print the results for inspection.
    print(f"method: {method}; entropy: {entropies.mean():.3f} +- {xerr:.3f}; "
          f"RMSE: {rmses.mean():.3f} +- {yerr:.3f}")
    
ax.legend(loc='upper left')
ax.set_xlabel(r'Expected posterior entropy $\mathcal{H}$')
ax.set_ylabel('Root mean squared error')
ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
ax.set_ylim(top=4.175)

# A reasonably hacky approach to generating the illustrative model for the coalescent... Sorry.
ax = axes[0]
# Size of each individual (square or circle).
size = 0.65
# Sex indicator for individuals in each layer, corresponding to generations. We tack on an 
# individual with "square" sex to each layer to illustrate that the population is much larger.
layers = [
    [0, 1, 0, 1, ],
    [1, 0, 0, 1, ],
    [1, 1, 0, 1, ],
]

for i, layer in enumerate(layers):
    for j, sex in enumerate(layer):
        kwargs = {'facecolor': 'none', 'edgecolor': 'black'}
        # Use different shapes by sex.
        if sex:
            patch = mpl.patches.Circle((i, -j), size / 2, **kwargs)
        else:
            patch = mpl.patches.Rectangle((i - size / 2, - j - size / 2), size, size, **kwargs)
        ax.add_patch(patch)
        
ax.set_ylim(- len(layer) - 1, 1)
ax.set_xlim(-.5, len(layers)-.5)
ax.set_aspect('equal')

# Relationships in each layer. Each tuple corresponds to one of the individuals in `layers`. E.g.,
# a tuple (i, j) at the second position indicates that the person with index 1 in the layer has 
# parents i and j in the previous layer.
relationships = [
    [(0, 1), (2, 3), (2, 3), (2, 3)],
    [(0, 1), (0, 1), (3, 4), (3, 4)],
]

# Draw the relationship lines.
for i, layer in enumerate(relationships):
    for j, parents in enumerate(layer):
        for sign in [-1, 1]:
            xy = [
                (i - size / 2 + 1, j), 
                (i + .5, j), 
                (i + .5, np.mean(parents)),
                (i, np.mean(parents)),
                (i, np.mean(parents) + (0.5 - size / 2) * sign),
            ]
            x, y = np.transpose(xy)
            line = mpl.lines.Line2D(x, -y, solid_capstyle='butt', 
                                    color='k', linewidth=1, solid_joinstyle='round')
            ax.add_line(line)
            
# Height and which of each illustrated chromosome.
csize = 0.75 * size
cheight = 0.2 * size
# Color of chromosomes by individual. Individuals are identified by their position in the grid. The
# value comprises two "chromosomes", each having one or more different genes (indicated by different
# colors).
chromosomes_by_individual = {
    (0, 0): [['C0'], ['C1']],
    (0, 1): [['C4'], ['C7']],
    (0, 2): [['C2'], ['C7']],
    (0, 3): [['C3'], ['C7']],
    (1, 0): [['C0', 'C1'], ['C4']],
    (1, 1): [['C2'], ['C3']],
    (2, 0): [['C0', 'C1'], ['C2']]
}

# Draw the chromosome patches.
for (x, y), chromosomes in chromosomes_by_individual.items():
    for i, colors in enumerate(chromosomes):
        for j, color in enumerate(colors):
            patch = mpl.patches.Rectangle(
                (x - csize / 2 + j / len(colors) * csize, - y - i * cheight), 
                csize / len(colors), cheight, 
                facecolor=color, alpha=.75)
            ax.add_patch(patch)
        patch = mpl.patches.Rectangle(
            (x - csize / 2, - y - i * cheight),
            csize, cheight,
            facecolor='none', edgecolor='k'
        )
        ax.add_patch(patch)
        
# Add on a random mutation.
ax.scatter([1, 2], [-1 + cheight / 2, -cheight / 2], marker='.', color='k').set_edgecolor('w')

# Illustrate the direction of time.      
y = - len(layer) - .5
ax.arrow(- size / 2, y, len(layers) - 1 + size, 0, 
         linewidth=1, head_width=.1, length_includes_head=True, facecolor='k')
ax.text(len(layers) / 2 - .5, y - .1, r'Generations', va='top', ha='center')

# Plot the semi-transparent individuals illustrating the larger population (squares are easier to 
# plot). This isn't pretty, but it does the trick.
segments = []
z = []
for i in range(len(layers)):
    left = i - size / 2
    right = i + size / 2
    top = -len(layer) + size / 2
    segments.append([(left, top), (right, top)])
    z.append(0.)
    for x in [left, right]:
        previous = None
        for y in np.linspace(0, 1, 10):
            if previous is not None:
                segments.append([(x, top - y * size / 2), (x, top - previous * size / 2)])
                z.append(y)
            previous = y
    
collection = mpl.collections.LineCollection(segments, array=z, cmap='binary_r', lw=1)
ax.add_collection(collection)
ax.yaxis.set_ticks([])
ax.xaxis.set_ticks([])
ax.set_axis_off()

summaries.label_axes(axes[0])
summaries.label_axes(axes[1], loc='bottom right')

fig.tight_layout()
fig.savefig(FIGURE_ROOT / 'coalescent.pdf')
```

## Comparison of different compression methods.

```{code-cell} ipython3
# Load the test set to compare the compression methods.
with open(WORKSPACE_ROOT / 'coal/data/test.pkl', 'rb') as fp:
    test_data = pickle.load(fp)
```

```{code-cell} ipython3
# Apply two different models to the test set to get embeddings.
x = test_data['samples']['x']
x = th.as_tensor(x)
model1 = th.load(WORKSPACE_ROOT / 'coal/mdn_compressor.pt')
model2 = th.load(WORKSPACE_ROOT / 'coal/regressor.pt')
with th.no_grad():
    y1 = model1(x).numpy()
    y2 = model2(x).numpy()
```

```{code-cell} ipython3
# Align the embeddings using a procrustes transform.
y1p, y2p, _ = spatial.procrustes(y1, y2)
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
for i, ax in enumerate(axes):
    a = y1p[:, i]
    b = y2p[:, i]
    ax.scatter(a, b, c=test_data['samples']['theta'][:, i])
    print(f"correlation of dimension {i}: {stats.pearsonr(a, b)[0]:.3f}")
```

```{code-cell} ipython3
# Note that these "look too good to be true" because `spatial.procrustes` also applies scalings. It
# is reassuring to see that the two embeddings prior to procrustes transform have different 
# covariance eigenvalues.
print(f"evals of method 1: {np.linalg.eigvalsh(np.cov(y1.T))}")
print(f"evals of method 2: {np.linalg.eigvalsh(np.cov(y2.T))}")
```
