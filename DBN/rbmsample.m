function [v2] = rbmsample(rbm, h2Sampled, pars)
% RBMSAMPLE ...
%
%   pars contains 4 parameters. numberIterMF, lateralUpdateMode, and
%   stochasticVisible, numberIter.
%   if numberIter is 1, then it's just rbm down.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 28-Apr-2014 21:00:35 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : rbmsample.m

numberIter = pars.numberIter;

if isfield(rbm,'lateralVisible') && rbm.lateralVisible
    rbm.lateralVisibleMFIter = pars.numberIterMF; % put the parameter into rbm so that rbm_meanfield can use that directly.
    lateralUpdateMode = pars.lateralUpdateMode;
    assert(all(rbm.LV(~rbm.lateralVisibleMask)==0)); % for sanity check.
end

sigma = rbm.sigmaFinal; % no compatibility
rbm.batchsize = size(h2Sampled,1); % hack.. don't want to change code much...
for iIter = 1:numberIter
    if isequal(rbm.types{2},'binary')
        v2 = sigm(  (1/(sigma^2))*    (h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1))         );
        
        if pars.stochasticVisible
            v2 = (v2 > rand(size(v2)));
        end
        
    else
        v2 = h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1);
        
        if pars.stochasticVisible
            v2 = v2 +  sigma*randn(size(v2));
        end
        
    end
    
    % update lateral visible layers
    if isfield(rbm,'lateralVisible') && rbm.lateralVisible
        
        if isequal(lateralUpdateMode,'MF')
            % update v2 using mean-field so that they somewhat converge
            [v2] = rbm_meanfield(v2, h2Sampled, rbm, sigma);
            %                 disp(MFiter);
        else
            % normal Gibbs sampling
            assert(isequal(lateralUpdateMode,'gibbs'));
            [v2] = rbm_gibbs(v2, h2Sampled, rbm, sigma);
        end
    end
    
    % get hidden layer.
    if isequal(rbm.types{1},'binary') % h2 is probability, and h2Sampled is stochastic version.
        h2 = sigm(   (1/(sigma^2)) *(v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1))     );
    else
        h2 = v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
    end
    
    if isequal(rbm.types{1},'binary') %stochastic sample based on h2.
        h2Sampled = h2 > rand(size(h2));
        %             h1Sampled = h1;
    else
        h2Sampled = h2 + sigma*randn(size(h2)); % sigma is the std deviation
        % why my current implementation fail to match old one on
        % vanHateren_Lee
    end
    disp(iIter);
end




end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [rbmsample.m] ======
