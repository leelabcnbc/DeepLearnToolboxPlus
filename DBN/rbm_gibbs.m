function [v, iIter] = rbm_gibbs(v, hSampled, rbm, sigma, parallelFlag)
% RBM_GIBBS update visible units v using gibbs rule.
%
%   ...
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 28-Apr-2014 21:22:20 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : rbm_gibbs.m

[N] = size(v,1);
% sizeH = size(hSampled,2);
assert(size(hSampled,1)==N);

% related things
% rbm.W is [sizeH x sizeV]
% rbm.b is [sizeV x 1]
% rbm.LV is [sizeV x sizeV]
% rbm.lateralVisibleMask is [sizeV x sizeV]

% hSampled is [N x sizeH]
% v is [N x sizeV]

assert(isequal(rbm.types{2},'binary')); % only support binary gibbs ...

assert(isequal(rbm.LV,rbm.LV'));

if nargin < 5
    parallelFlag = false; % just for debug
end

if parallelFlag
    v2 = zeros(size(v));
end

for iIter = 1:rbm.lateralVisibleMFIter
    sequenceThis = randperm(size(v,2));
    for iUnitIdx = 1:size(v,2)
        visibleUnitNumber = sequenceThis(iUnitIdx); % update #visibleUnitNumber unit.
        inputForThisUnit = hSampled*rbm.W(:,visibleUnitNumber) + v  *  rbm.LV(:,visibleUnitNumber);
        inputForThisUnit = inputForThisUnit + rbm.b(visibleUnitNumber);
        v_new = sigm( (1/(sigma^2))*  inputForThisUnit  );
        if parallelFlag
            v2(:,visibleUnitNumber) = v_new; % just a mean field example for testing.
        else
            v(:,visibleUnitNumber) = (v_new > rand(size(v_new)));
        end
    end
    
    if parallelFlag
        v_diff = mean(abs(v2(:) - v(:)));
        if v_diff < 1e-7 % value used by Ruslan in DBM
            %         v_new = v; % just for hinton's sake.
            break;
        end
        v = v2;
    end
    
    disp(iIter);
    
end
end










% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [rbm_gibbs.m] ======
