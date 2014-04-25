function test_example_RBM_hinton()
% load mnist_uint8;
% 
% train_x = double(train_x) / 255;
% test_x  = double(test_x)  / 255;
% train_y = double(train_y);
% test_y  = double(test_y);

load hinton_2006_mnist_train;
%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0,'twister'); % reset random state to get reproducible results.
% dbn.sizes = [1000 500 250];
% dbn.types = {'binary','binary','binary','binary'};

dbn.sizes = [1000 500 250 30];
dbn.types = {'binary','binary','binary','binary','gaussian'};

opts.numepochs =   5;
opts.batchsize =  100; 
opts.momentum  =   0.5;
% opts.alpha     =   0.1;
opts.alpha     =   [0.1 0.1 0.1 0.001];

opts.weightPenaltyL2 = 0.0002; % L2 penalty
opts.momentumFinal  = 0.9; % momentum in the later stages
opts.epochFinal = 5; % when to change the alpha for momentum
opts.batchOrderFixed = true;

opts.lateralVisible = true;
opts.lateralVisibleMask = {'none'};

% opts.CDIter = 1; % you can uncomment this, but it's the same.

opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, batchdata2, opts);
dbn = dbntrain(dbn, batchdata2, true);


load mnistvh 
load mnisthp
load mnisthp2
load mnistpo
assert(isequal(dbn.rbm{1}.W,vishid'));
assert(isequal(dbn.rbm{1}.b,visbiases'));
assert(isequal(dbn.rbm{1}.c,hidrecbiases'));
assert(isequal(dbn.rbm{2}.W,hidpen'));
assert(isequal(dbn.rbm{2}.b,hidgenbiases'));
assert(isequal(dbn.rbm{2}.c,penrecbiases'));
assert(isequal(dbn.rbm{3}.W,hidpen2'));
assert(isequal(dbn.rbm{3}.b,hidgenbiases2'));
assert(isequal(dbn.rbm{3}.c,penrecbiases2'));
assert(isequal(dbn.rbm{4}.W,hidtop'));
assert(isequal(dbn.rbm{4}.b,topgenbiases'));
assert(isequal(dbn.rbm{4}.c,toprecbiases'));

fprintf('all pass!\n');

end