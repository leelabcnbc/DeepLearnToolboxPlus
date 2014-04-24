function [v,iIter]  = rbm_meanfield_naive(v, hSampled, rbm, sigma)
% RBM_MEANFIELD_NAIVE update visible units v using mean-field rule. 
%  
%   v is visible units, h is current sample values of hidden units, rbm is
%   the struct saving all parameters.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 23-Apr-2014 21:48:30 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : rbm_meanfield_naive.m 

[N,sizeV] = size(v);
% sizeH = size(hSampled,2);
assert(size(hSampled,1)==N);

% should consider both binary case, and gaussian case.

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
                
    inputForAllV = zeros(N,sizeV);
    
    
    for iSample = 1:N
        h2This = hSampled(iSample,:); % [1 x sizeH]
        BiasFromHidden = zeros(1,sizeV); % [1 x sizeV]. every entry is \sum_i h_j w_ij, or \sum_i v_i h_j w_ij.
        % so is v_i one, or is v_i some mean value? I think it should be
        % one. Basically, mean field means, given other nodes (may be 0-1,
        % or real-valued mean), what's the probability that's this node is
        % on? (and this probability is just the mean). In computing this
        % probability, we still think that v_i can take 0 or 1 (binary
        % values).

        vThis = v(iSample,:);
        
        BiasFromBias = rbm.b';
        BiasFromVisible = zeros(1,sizeV);
        
        for iVisible = 1:sizeV
            BiasFromHidden(iVisible) = h2This*rbm.W(:,iVisible); % bias from hidden
            weight1 = rbm.LV(:,iVisible);
            weight2 = rbm.LV(iVisible,:);
            
            weight1 = weight1(:);
            weight2 = weight2(:);
            assert(isequal(weight1,weight2));
            
            maskThis1 = rbm.lateralVisibleMask(:,iVisible);
            maskThis2 = rbm.lateralVisibleMask(iVisible,:);
            
            maskThis1 = maskThis1(:);
            maskThis2 = maskThis2(:);
            assert(isequal(maskThis1,maskThis2));
            
            
            assert(all(weight1(~maskThis2)==0));
            assert(all(weight2(~maskThis2)==0));
            
            assert(weight1(iVisible)==0);
            
            vTemp = vThis;
            vTemp(iVisible) = 0;
            BiasFromVisible(iVisible) = vTemp*weight1; % bias from visible
        end
        
        inputForAllV(iSample,:) = BiasFromHidden+BiasFromVisible+BiasFromBias;
    end

        
%         BiasFromHidden = h2This*rbm.W; % 
        
        
    
%     
%     inputForAllV = h2Sampled * rbm.W +  ...  % term from clamped hidden units
%                     v  *  rbm.LV;   % term from lateral v
%     inputForAllV = bsxfun(@plus, inputForAllV, rbm.b'); % add bias.
    v_new = sigm( (1/(sigma^2))*  inputForAllV  );
    v_new = rbm.lateralVisibleMFDamp * v + (1-rbm.lateralVisibleMFDamp) * v_new; % if Damp parameter is 0, then no dampening.
    v_diff = mean(abs(v_new(:) - v(:)));
    if v_diff < 1e-7 % value used by Ruslan in DBM
        break;
    end
    v = v_new;
end

v = v_new;

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [rbm_meanfield_naive.m] ======  
