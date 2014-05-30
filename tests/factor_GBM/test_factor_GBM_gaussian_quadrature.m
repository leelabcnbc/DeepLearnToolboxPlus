% use Roland's data for testing.

load('modelInit_gaussian.mat');
load('after_one_batch_gaussian.mat','batch_x','batch_y');
% pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
%     'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',true,...
%     'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
%     'everySave',1,'numbatches',1,'numepoch',1,'visType','binary','nummap',100);

pars = factor_GBM_train(batch_x',batch_y','numfactors',256,'stepsize',0.01,'meanfield_output',true,...
    'momentum',0.9,'batchsize',200,'batchOrderFixed',true,'weightPenaltyL2',0.001,...
    'everySave',1,'numepoch',1,'visType','gaussian','nummap',128,'saveFile',false,'seed',0,...
    'zeromask','quadrature','zeromaskAdditional',true);


% load('after_one_epoch.mat');

load('after_one_batch_gaussian.mat');

max(abs(pars.wh(:)-wh(:)))
max(abs(pars.wy(:)-wy(:)))
max(abs(pars.whf(:)-whf(:)))
max(abs(pars.wyf(:)-wyf(:)))
max(abs(pars.wxf(:)-wxf(:)))



