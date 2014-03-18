function test_example_sparse_RBM_Lee_MNIST()
load mnist_real_stanford;

rng(0,'twister'); %reproducible...

train_x = trainData;
test_x  = testData;
train_y = trainLabelsFull;
test_y  = testLabelsFull;

% train_x = rand(60000,784); 
% [train_x_pca, unwhitenMatrix, M] = preprocessing.whiten_PCA(train_x,epsilon);
% K = 400;
% train_x_pca = train_x_pca(:,1:K);
% unwhitenMatrix = unwhitenMatrix(1:K,:);
% train_x_pca_reconstruct = bsxfun(@plus,train_x_pca*unwhitenMatrix, M);

dbn.sizes = [200];
dbn.types = {'gaussian','binary'};
opts.numepochs =   50;
opts.batchsize =  500; 
opts.momentum  =   0.5;
opts.alpha     =   0.01;

opts.weightPenaltyL2 = 3e-3; % L2 penalty
opts.momentumFinal  = 0.9; % momentum in the later stages
opts.epochFinal = 10; % when to change the alpha for momentum
opts.batchOrderFixed = false;

opts.sparsityTarget = 0.1; % the target p in Honglak Lee's paper.
opts.nonSparsityPenalty = 3; % the constant (or lambda) in Lee's paper.

opts.sigma = 0.5; % default 1.
opts.sigmaDecay = 0.99; % default 1.
opts.sigmaMin = 0.05; % default 1.

opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x);

reference = load('dbn_mnist_lee.mat','dbn','opts');

assert(isequal(reference.dbn,dbn));
assert(isequal(reference.opts,opts));

end