data {
    int num_obs;
    vector[num_obs] x;
    real<lower=0> variance_offset;
}

parameters {
    // We focus on the positive mode to avoid poor mixing and generate the actual samples in
    // `generated quantities`.
    real<lower=0> theta_;
}

transformed parameters {
    real loc = tanh(theta_);
    real<lower=0> scale = sqrt(variance_offset - loc ^ 2);
    vector[num_obs] target_parts;
    for (i in 1:num_obs) {
        target_parts[i] = log_sum_exp(
            normal_lpdf(x[i] | loc, scale),
            normal_lpdf(x[i] | -loc, scale)
        ) - log(2);
    }
}

model {
    theta_ ~ normal(0, 1);
    target += sum(target_parts);
}

generated quantities {
    // Randomly assign the sample to either mode.
    real theta = theta_ * (2.0 * bernoulli_rng(0.5) - 1.0);
}
