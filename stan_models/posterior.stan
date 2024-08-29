/* Gaussian posterior model */
data {
    real mu;
    real<lower=0> sigma;
    real<lower=0,upper=1> reduction_factor;
    real z;
}
parameters {
    real theta;
}
model {
    theta ~ normal(mu,sigma) T[mu-2*sigma,mu+2*sigma];
    z ~ normal(theta,reduction_factor*sigma);
}