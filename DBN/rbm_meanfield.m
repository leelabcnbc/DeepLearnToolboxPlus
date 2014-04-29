function [v, iIter] = rbm_meanfield(v, hSampled, rbm, sigma)
% RBM_MEANFIELD update visible units v using mean-field rule. 
%  
%   v is visible units, h is current sample values of hidden units, rbm is
%   the struct saving all parameters.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 23-Apr-2014 21:48:30 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : rbm_meanfield.m 

[N] = size(v,1);
% sizeH = size(hSampled,2);
assert(size(hSampled,1)==N);

% should consider both binary case, and gaussian case. % let's only
% consider binary now.

% related things
% rbm.W is [sizeH x sizeV]
% rbm.b is [sizeV x 1]
% rbm.LV is [sizeV x sizeV]
% rbm.lateralVisibleMask is [sizeV x sizeV]

% hSampled is [N x sizeH]
% v is [N x sizeV]

assert(isequal(rbm.types{2},'binary')); % only support binary mean-field...

% maskedLV = rbm.LV; % need to be optimized later...
% % maskedLV(~rbm.lateralVisibleMask) = 0;

% rbm.LV is already masked.

assert(isequal(rbm.LV,rbm.LV'));

for iIter = 1:rbm.lateralVisibleMFIter
                   
    inputForAllV = hSampled * rbm.W +  ...  % term from clamped hidden units
                    v  *  rbm.LV;   % term from lateral v
    inputForAllV = bsxfun(@plus, inputForAllV, rbm.b'); % add bias.
    v_new = sigm( (1/(sigma^2))*  inputForAllV  );
    v_new = rbm.lateralVisibleMFDamp * v + (1-rbm.lateralVisibleMFDamp) * v_new; % if Damp parameter is 0, then no dampening.
    v_diff = mean(abs(v_new(:) - v(:)));
    if v_diff < 1e-7 % value used by Ruslan in DBM
%         v_new = v; % just for hinton's sake.
        break;
    end
    v = v_new;
end

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [rbm_meanfield.m] ======  
