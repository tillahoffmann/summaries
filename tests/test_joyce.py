import numpy as np
from scipy import stats
from summaries import joyce


def test_joyce_bin_sensitivity():
    """
    This test verifies that Joyce and Marjoram's proposed method is sensitive to the choice of the
    number of bins because there is no multiple-hypothesis correction.
    """
    num_runs = 100
    num_samples = 10000
    scores = []
    for num_bins in [5, 20]:
        probas = np.random.dirichlet(np.ones(num_bins), num_runs)
        xs = np.asarray([np.random.multinomial(num_samples, proba) for proba in probas])
        ys = np.asarray([np.random.multinomial(num_samples, proba) for proba in probas])
        scores.append(joyce.evaluate_joyce_score(xs, ys))

    scores1, scores2 = scores
    assert np.median(scores2) > np.median(scores1)


def test_joyce_likelihood_ratio_calibration():
    """
    This test ensures that the likelihood ratio test for rejecting the null-hypothesis that the bin
    counts are the same is properly calibrated.
    """
    num_runs = 100
    num_samples = 10000
    num_bins = 10
    alpha = .3
    probas = np.random.dirichlet(np.ones(num_bins), num_runs)
    xs = np.asarray([np.random.multinomial(num_samples, proba) for proba in probas])
    ys = np.asarray([np.random.multinomial(num_samples, proba) for proba in probas])
    scores = joyce.evaluate_log_likelihood_ratio(xs, ys)
    cdfs = stats.chi2(num_bins - 1).cdf(scores)
    binom = stats.binomtest((cdfs < alpha).sum(), num_runs, alpha)
    assert binom.pvalue > 0.001
