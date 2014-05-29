% use Roland's data for testing.

load('modelInitAndData.mat');

% pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
%     'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',true,...
%     'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
%     'everySave',1,'numbatches',1,'numepoch',1,'visType','binary','nummap',100);

pars = factor_GBM_train(inputimages,outputimages,'wxf',wxf,'wyf',wyf,'whf',whf,...
    'wy',wy,'wh',wh,'numfactors',200,'stepsize',0.01,'meanfield_output',true,...
    'momentum',0.9,'batchsize',100,'batchOrderFixed',true,'weightPenaltyL2',0,...
    'everySave',1,'numepoch',1,'visType','binary','nummap',100,'saveFile',false);


% load('after_one_epoch.mat');

load('after_one_epoch.mat');

max(abs(pars.wh(:)-wh(:)))
max(abs(pars.wy(:)-wy(:)))

max(abs(pars.whf(:)-whf(:)))
hist(abs(pars.whf(:)-whf(:)),100);
pause;

max(abs(pars.wyf(:)-wyf(:)))
hist(abs(pars.wyf(:)-wyf(:)),100);
pause;


max(abs(pars.wxf(:)-wxf(:)))
hist(abs(pars.wxf(:)-wxf(:)),100);
pause;



