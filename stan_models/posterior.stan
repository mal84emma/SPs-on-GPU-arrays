/* Gaussian posterior model for mean load given sample measurement */
data {
    real mu;
    real<lower=0> sigma;
    real error;
    real z;
}
parameters {
    real theta;
}
model {
    theta ~ normal(mu,sigma) T[mu-2*sigma,mu+2*sigma];
    z ~ normal(theta,error*theta);
}