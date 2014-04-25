function test_example_RBM_hinton_layer1_full()
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

dbn.sizes = [500]; % parameter in Hinton's 2006 neural computation paper.
dbn.types = {'binary','binary'};

opts.numepochs =   50;
opts.batchsize =  100; 
opts.momentum  =   0.5;
% opts.alpha     =   0.1;
opts.alpha     =   [0.1];

opts.weightPenaltyL2 = 0.0002; % L2 penalty
opts.momentumFinal  = 0.9; % momentum in the later stages
opts.epochFinal = 5; % when to change the alpha for momentum
opts.batchOrderFixed = true;

opts.lateralVisible = false;
opts.visualize = true;

% opts.CDIter = 1; % you can uncomment this, but it's the same.

opts = expandOpts(opts,numel(dbn.sizes));

fileNameArray = {['RBM_hinton_layer1_full_' datestr(now,30) '.mat']};

dbn = dbnsetup(dbn, batchdata2, opts);
dbn = dbntrain(dbn, batchdata2, true, true, fileNameArray);

end