% Testing what statistics can be useful to predict marignal likelihoods

%% Load paths

addpath(genpath('../source/gpml'));
addpath(genpath('./util'));

%% PRNG

seed=1;   % fixing the seed of the random generators
randn('state',seed); %#ok<RAND>
rand('state',seed); %#ok<RAND>

%% Generate data from GP

cov_fn = {@covSum, {{@covMask, {[1, 0, 0], @covSEiso}}, ...
                    {@covMask, {[0, 1, 0], @covSEiso}}, ...
                    {@covNoise} }};
hyp.cov = [0, 0, -1, 0, -1];

x = randn(500, 3);
K = feval(cov_fn{:}, hyp.cov, x);
y = chol(K)' * randn(size(x,1), 1);

%% Compute correlations

rho = corr(x, y);
display(rho);

%% Compute RDC

rho = [rdc(x(:,1), y);
       rdc(x(:,2), y);
       rdc(x(:,3), y)];
display(rho);

%% Compute some sort of generalisation of autocorrelation