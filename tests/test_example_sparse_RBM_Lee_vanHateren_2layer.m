function test_example_sparse_RBM_Lee_vanHateren_2layer()
% TEST_EXAMPLE_SPARSE_RBM_LEE_VANHATEREN_2LAYER ... 
%  
%   debug script to be compared with new implementation in RSA research. 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 02-Aug-2015 12:41:40 $ 
%% DEVELOPED : 8.3.0.532 (R2014a) 
%% FILENAME  : test_example_sparse_RBM_Lee_vanHateren_2layer.m 




load vanHaterenPatch_20140316T130207;

%% train V1
train_x = train_x(1:100000,:);
penalty = 4;
dbn.sizes = [200];
dbn.types = {'gaussian','binary'};
train_x = preprocessing.contrast_normalization(train_x,0.001);
opts.numepochs =  10;
opts.batchsize =  200; 
opts.momentum  =  0;
opts.alpha     =  0.01;

opts.alphaFinal   =  0.01;
% opts.weightPenaltyL2 = 3e-3; % L2 penalty
opts.momentumFinal  = 0; % momentum in the later stages
opts.epochFinal = 500; % when to change the alpha for momentum
opts.batchOrderFixed = true;

opts.sparsityTarget = 0.02; % the target p in Honglak Lee's paper.
opts.nonSparsityPenalty = penalty; % the constant (or lambda) in Lee's paper.

opts.sigma = 0.5; % default 1.
opts.sigmaDecay = 0.99; % default 1.
opts.sigmaMin = 0.05; % default 1.

opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);

rng(0,'twister');
WinitV1 = 0.1*randn(size(dbn.rbm{1}.W,2),size(dbn.rbm{1}.W,1))';
dbn.rbm{1}.W = WinitV1;
dbn.rbm{1}.initialized = true;

rng(0,'twister'); %reproducible...
dbnV1 = dbntrain(dbn, train_x);

%% train V2
train_x = rbmup(dbnV1.rbm{1},train_x);
dbn = struct();
opts = struct();
dbn.sizes = [200]; %again 200 hidden units.
dbn.types = {'gaussian','binary'};

opts.numepochs =  10;
opts.batchsize =  200; 
opts.momentum  =  0;
opts.alpha     =  0.01;
opts.alphaFinal   =  0.01;
% opts.weightPenaltyL2 = 3e-3; % L2 penalty
opts.momentumFinal  = 0; % momentum in the later stages
opts.epochFinal = 500; % when to change the alpha for momentum
opts.batchOrderFixed = true;
opts.sparsityTarget = 0.1; % the target p in Honglak Lee's paper.
opts.nonSparsityPenalty = 2; % the constant (or lambda) in Lee's paper.
opts.sigma = 0.5; % default 1.
opts.sigmaDecay = 0.99; % default 1.
opts.sigmaMin = 0.05; % default 1.
opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);

rng(0,'twister');
WinitV2 = 0.1*randn(size(dbn.rbm{1}.W,2),size(dbn.rbm{1}.W,1))';
dbn.rbm{1}.W = WinitV2;
dbn.rbm{1}.initialized = true;

rng(0,'twister');
dbnV2 = dbntrain(dbn, train_x);

save('sparseRBM_reference.mat',...
    'dbnV1','dbnV2','WinitV1','WinitV2');

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [test_example_sparse_RBM_Lee_vanHateren_2layer.m] ======  
