functions {
    real softplus(real x) {
        return log1p(exp(-fabs(x))) + fmax(x, 0);
    }
}

data {
    int n;
    vector[2] x[n];
    real<lower=0> epsilon;
}

parameters {
    real<lower=0> radius;
    real<lower=-1, upper=1> corr;
}

transformed parameters {
    // Container for likelihood contributions.
    vector[n] parts;
    real loc = 2.0 + radius;
    real<lower=0> scale = sqrt(softplus(25 - loc ^ 2));

    cov_matrix[2] covp;
    cov_matrix[2] covm;
    vector[2] locp = rep_vector(loc, 2);
    vector[2] locm = -locp;

    // Construct the positive and negative correlation matrices.
    for (i in 1:2) {
        for (j in 1:2) {
            covp[i, j] = scale ^ 2 * (i == j ? 1 : corr) + (i == j ? epsilon : 0);
            covm[i, j] = scale ^ 2 * (i == j ? 1 : -corr) + (i == j ? epsilon : 0);
        }
    }

    // Likelihood contributions.
    for (i in 1:n) {
        parts[i] = log_sum_exp(
            multi_normal_lpdf(x[i] | locp, covp),
            multi_normal_lpdf(x[i] | locm, covm)
        ) - log(2);
    }
}

model {
    // This implies that the square of the radius is a chi_square(2) distribution.
    target += log(radius) - radius ^ 2 / 2;
    // This implies we sample a angle uniformly at random and then evaluate the sine. To avoid the
    // nastiness of circular boundaries, we instead deal with the sine directly.
    target += -log1p(-corr ^ 2);
    target += sum(parts);
}

generated quantities {
    // Evaluate the parameters of interest in generated quantities so we don't accidentally use them
    // in the likelihood.
    vector[2] theta;
    theta[1] = radius * corr;
    // The second parameter can be either positive or negative because the likelihood does not
    // depend on the sign. We only sample one mode to make things easier. Here, we "recreate" the
    // second mode.
    theta[2] = radius * sqrt(1 - corr ^ 2) * (2 * bernoulli_rng(0.5) - 1);
}
