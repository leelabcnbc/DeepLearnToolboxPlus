function test_example_DBN_MNIST_Stanford
load mnist_real_stanford;

train_x = trainData;
test_x  = testData;
train_y = trainLabelsFull;
test_y  = testLabelsFull;

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rng(0,'twister');
dbn.sizes = [100];
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0.5;
opts.alpha     =   1;

opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN

dbn = [];

rng(0,'twister');
%train dbn
dbn.sizes = [100 100];
% dbn.types = {'binary','binary','binary'};
opts.numepochs =   10;
opts.batchsize = 100;
opts.momentum  =   0.5;
opts.alpha     =   [0.1];

opts = expandOpts(opts,numel(dbn.sizes));

dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  10;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

disp(er);

assert(er < 0.10, 'Too big error');
