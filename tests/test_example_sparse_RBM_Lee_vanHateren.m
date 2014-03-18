function test_example_sparse_RBM_Lee_vanHateren()
close all;
rng(0,'twister'); %reproducible...
load vanHaterenPatch_20140316T130207;

train_x = train_x(1:100000,:);

penalty = 4;

dbn.sizes = [200];
dbn.types = {'gaussian','binary'};

train_x = preprocessing.contrast_normalization(train_x,0.001);

opts.numepochs =  500;
opts.batchsize =  200; 
opts.momentum  =  0;
opts.alpha     =  0.01;

opts.alphaFinal   =  0.01;
% opts.weightPenaltyL2 = 3e-3; % L2 penalty
opts.momentumFinal  = 0; % momentum in the later stages
opts.epochFinal = 500; % when to change the alpha for momentum
opts.batchOrderFixed = false;

opts.sparsityTarget = 0.02; % the target p in Honglak Lee's paper.
opts.nonSparsityPenalty = penalty; % the constant (or lambda) in Lee's paper.

opts.sigma = 0.5; % default 1.
opts.sigmaDecay = 0.99; % default 1.
opts.sigmaMin = 0.05; % default 1.

opts = expandOpts(opts,numel(dbn.sizes));

% fileName = ['image' datestr(now,30) '.png'];

% saveas(gcf,fileName);

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x);

dW = dbn.rbm{1}.W;
visualize(dW');

reference = load('dbn_vanHateren_lee_layer1.mat');

assert(isequal(reference.dbn,dbn));
assert(isequal(reference.opts,opts));

end