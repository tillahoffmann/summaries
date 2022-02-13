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
    vector[2] theta;
}

transformed parameters {
    // Container for likelihood contributions.
    vector[n] parts;
    real t1 = theta[1];
    real t2 = theta[2];
    real<lower=0> radius = sqrt(t1 ^ 2 + t2 ^ 2);
    real loc = 2.0 + radius;
    real<lower=0> scale = sqrt(softplus(25 - loc ^ 2));
    real corr = t1 / radius;
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
    theta ~ normal(0, 1);
    target += sum(parts);
}
