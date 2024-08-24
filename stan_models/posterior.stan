/* Gaussian posterior model for mean load given sample measurement */
data {
    real mu;
    real sigma;
    real error;
    real z;
}
parameters {
    real theta;
}
model {
    theta ~ normal(mu,sigma);
    z ~ normal(theta,error*theta);
}