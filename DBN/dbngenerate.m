function topInput = dbngenerate(dbn, pars)
% DBNGENERATE generate samples from a given DBN. 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 28-Apr-2014 20:44:37 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : dbngenerate.m 

% from top layer to bottom layer

if isfield(pars,'numberOfSamples')
    numberOfSamples = pars.numberOfSamples;
else
    numberOfSamples = 100; % 100 samples by default.
end

% initialize data

rbmNumber = numel(dbn.rbm);

topInputSize = size(dbn.rbm{rbmNumber}.W,1); % hidden size of top RBM.

if isequal(dbn.rbm{rbmNumber}.types,'binary') % generate random binary units
    topInput = rand(numberOfSamples,topInputSize) > 0.5;
else
    assert(isequal(dbn.rbm{rbmNumber}.types,'gaussian'));
    topInput = randn(numberOfSamples,topInputSize);
end


for iRBM = numel(dbn.rbm)-1:-1:1
    topInput = rbmsample(dbn.rbm{iRBM},topInput,pars.rbmPars{iRBM}); % one set of parameters for each RBM.
end


end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [dbngenerate.m] ======  
