%% Clear and load paths

close all;
clear all;
addpath(genpath('../gpml/'));
addpath(genpath('../util/'));

%% Set params

max_n = 500;
step_size = 50;
repeats = 50;

%% Select a kernel - white noise

cov_fn = {@covNoise};
hyp.cov = 10;

hyp.lik = [];
hyp.mean = [];

%% Select a kernel - const + noise

cov_fn = {@covSum, {@covConst, @covNoise}};
hyp.cov = [0, 0];

hyp.lik = [];
hyp.mean = [];

SnR = exp(hyp.cov(1)) / exp(hyp.cov(2));

%% Select a kernel - SE + noise

cov_fn = {@covSum, {@covSEiso, @covNoise}};
hyp.cov = [-3, 1, 0];

hyp.lik = [];
hyp.mean = [];

SnR = exp(hyp.cov(2)) / exp(hyp.cov(3));

%% Copy params

other_cov_fn = cov_fn;
other_hyp = hyp;

%% Generate data and calculate scaled ML

lmls = zeros(floor(max_n / step_size), 1);
lmls_var = zeros(floor(max_n / step_size), 1);

for n = step_size:step_size:max_n
    x = rand(n,1);
    K = feval(cov_fn{:}, hyp.cov, x);
    for repeat = 1:repeats
        y = chol(K)' * randn(n, 1);
        lml = -gp(hyp, @infDelta, @meanZero, cov_fn, @likDelta, x, y);
        % lmls(n/step_size) = lmls(n/step_size) + lml;
        lmls(n/step_size) = lmls(n/step_size) + (lml / n);
        lmls_var(n/step_size) = lmls_var(n/step_size) + (lml / n)^2;
    end
    lmls(n/step_size) = lmls(n/step_size) / repeats;
    lmls_var(n/step_size) = lmls_var(n/step_size) / repeats - ...
                            lmls(n/step_size).^2;
end

h = figure();
errorbar(step_size:step_size:max_n, lmls, 1*sqrt(lmls_var));
title(['SnR = ' num2str(SnR)]);
ylabel('log ML per data point');
xlabel('Number of data points');
save2pdf('temp.pdf', h, 600, true);
close all;

%% Bayes factor version

lmls = zeros(floor(max_n / step_size), 1);
lmls_var = zeros(floor(max_n / step_size), 1);

for n = step_size:step_size:max_n
    x = rand(n,1);
    K = feval(cov_fn{:}, hyp.cov, x);
    for repeat = 1:repeats
        y = chol(K)' * randn(n, 1);
        lml = -gp(hyp, @infDelta, @meanZero, cov_fn, @likDelta, x, y) ...
              +gp(other_hyp, @infDelta, @meanZero, other_cov_fn, ...
                  @likDelta, x, y);
        % lmls(n/step_size) = lmls(n/step_size) + lml;
        lmls(n/step_size) = lmls(n/step_size) + (lml / n);
        lmls_var(n/step_size) = lmls_var(n/step_size) + (lml / n)^2;
    end
    lmls(n/step_size) = lmls(n/step_size) / repeats;
    lmls_var(n/step_size) = lmls_var(n/step_size) / repeats - ...
                            lmls(n/step_size).^2;
end

h = figure();
errorbar(step_size:step_size:max_n, lmls, 1*sqrt(lmls_var));
title(['SnR = ' num2str(SnR)]);
ylabel('log Bayes factor per data point');
xlabel('Number of data points');
save2pdf('temp.pdf', h, 600, true);
% close all;