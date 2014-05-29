% use Roland's data for testing.

load('modelInitAndData.mat');

% pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
%     'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',true,...
%     'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
%     'everySave',1,'numbatches',1,'numepoch',1,'visType','binary','nummap',100);

pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
    'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',false,...
    'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
    'everySave',10,'numepoch',200,'visType','binary','nummap',100,'saveFile',false);

figure;
visualize(pars.wxf);
figure;
visualize(pars.wyf);


