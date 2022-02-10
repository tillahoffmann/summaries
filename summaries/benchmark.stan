data {
    int n;  // Number of observations per likelihood.
    int p;  // Number of different likelihoods.
    int x[p, n];  // Count data from negative binomial model.
}

parameters {
    real<lower=0, upper=1> t1;
    real<lower=0, upper=1> t2;
}

transformed parameters {
    real<lower=0> trials[p];
    real<lower=0, upper=1> proba[p];
    real parts[p];

    // Evaluate all the parameters.
    trials[1] = 1 + t1;
    proba[1] = 0.1 + 0.8 * t2;

    trials[2] = 1 + t1;
    proba[2] = 0.1 + 0.8 * sqrt(t1 * t2);

    trials[3] = 1 + sqrt(t1 * t2);
    proba[3] = 0.1 + 0.8 * t1;

    trials[4] = 1 / (1 + t1);
    proba[4] = 0.1 + 0.8 * t2;

    // Evaluate likelihood contributions so we can compare with the python implementation.
    for (i in 1:p) {
        parts[i] = neg_binomial_lpmf(x[i] | trials[i], proba[i] / (1 - proba[i]));
    }
}

model {
    // Specify a prior, but these are "no-op"s.
    t1 ~ beta(1, 1);
    t2 ~ beta(2, 2);
    // Evaluate the likelihood.
    target += sum(parts);
}
