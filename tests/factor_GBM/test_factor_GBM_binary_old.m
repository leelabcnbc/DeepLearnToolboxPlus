% use Roland's data for testing.

load('modelInitAndData.mat');

% pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
%     'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',true,...
%     'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
%     'everySave',1,'numbatches',1,'numepoch',1,'visType','binary','nummap',100);

pars = factor_GBM_train_old(inputimages,outputimages,'numfactors',200,...
    'stepsize',0.01,'meanfield_output',false,...
    'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
    'everySave',200,'numepoch',200,'visType','binary','nummap',100,'saveFile',false,...
    'seed',0);

pars_new = pars;
load('test_factor_GBM_binary_demo_pars_reference.mat');

pars_new = rmfield(pars_new,{'datestring','saveFile'});
pars = rmfield(pars,{'datestring','saveFile'});

assert(isequaln(pars_new,pars));

