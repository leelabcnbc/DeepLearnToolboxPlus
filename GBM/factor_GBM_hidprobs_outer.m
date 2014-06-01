function hidprobs = factor_GBM_hidprobs_outer(outputs,inputs,pars)
% FACTOR_GBM_HIDPROBS_OUTER ...
%
%   ...
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 01-Jun-2014 11:43:21 $
%% DEVELOPED : 8.3.0.532 (R2014a)
%% FILENAME  : factor_GBM_hidprobs_outer.m
factors_x = inputs*pars.wxf;
factors_y = outputs*pars.wyf;
hidprobs = sigm(bsxfun(@plus, (factors_x.*factors_y)*(pars.whf'), pars.wh'));  
%  [N x numfactor] x [numfactor x nummap]
end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [factor_GBM_hidprobs_outer.m] ======
