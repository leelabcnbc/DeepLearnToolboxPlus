function test_example_sparse_RBM_Lee_vanHateren_layer2()
close all;
rng(0,'twister'); %reproducible...
load vanHaterenPatch_20140316T130207;
train_x = train_x(1:100000,:);
train_x = preprocessing.contrast_normalization(train_x,0.001); % preprocessing the same
dbn_old = load('dbn_vanHateren_lee_layer1.mat');

train_x = rbmup(dbn_old.dbn.rbm{1},train_x);

dbn.sizes = [200]; %again 200 hidden units.
dbn.types = {'gaussian','binary'};

opts.numepochs =  600;
opts.batchsize =  200; 
opts.momentum  =  0;
opts.alpha     =  0.01;
opts.alphaFinal   =  0.005;
% opts.weightPenaltyL2 = 3e-3; % L2 penalty
opts.momentumFinal  = 0; % momentum in the later stages
opts.epochFinal = 500; % when to change the alpha for momentum
opts.batchOrderFixed = false;
opts.sparsityTarget = 0.1; % the target p in Honglak Lee's paper.
opts.nonSparsityPenalty = 2; % the constant (or lambda) in Lee's paper.
opts.sigma = 0.5; % default 1.
opts.sigmaDecay = 0.99; % default 1.
opts.sigmaMin = 0.05; % default 1.
opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x);


reference = load('dbn_vanHateren_lee_layer2.mat','dbn','opts');

assert(isequal(reference.dbn,dbn));
assert(isequal(reference.opts,opts));


end