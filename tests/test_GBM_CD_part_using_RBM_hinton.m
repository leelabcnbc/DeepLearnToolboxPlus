% how to use this:

% add breakpoint just after the loop for CD in rbmtrain.m, and run this.


params = struct();
params.xSize = 1;
params.ySize = 784;
params.hSize = 1000;

assert(isequal(size(rbm.W'),[ params.ySize params.hSize ]));

params.W_xyh = reshape(rbm.W',[1,params.ySize,params.hSize ])/2;
params.W_yh = squeeze(params.W_xyh);

params.W_xy = rbm.b'/2;
params.W_xh = rbm.c'/2;

params.W_y = rbm.b/2;
params.W_h = rbm.c/2;
params.sigma = sigma;
params.sampleV = false; % use probabilistic version of visible units.
% params.CDIter = rbm.CDIter;
% check returned h1 match what's computed by the reference RBM.

% effectiveWeightOverride = struct();
% 
% effectiveWeightOverride.W_yh_effective = rbm.W';
% effectiveWeightOverride.W_h_effective = rbm.c;
% effectiveWeightOverride.W_y_effective = rbm.b;


for iPoint = 1:100
    
%         randomly select CD iterations.
    params.CDIter = randi([1,5]);
%     disp( params.CDIter);
%     params.CDIter = 1;
    rng(0,'twister');
    
    xPoint = 1;
    
    batchNew = batch(iPoint,:);
    
    
%     [~,~,h1_GBM,h2_GBM,v2_GBM] = bm.gbm_statistics_single(1,batchNew,params,'CD',false,effectiveWeightOverride);
    [~,~,h1_GBM,h2_GBM,v2_GBM] = bm.gbm_statistics_single(1,batchNew,params,'CD',false);
    
    clear h1;
    clear v2;
    clear h2;
    
    rng(0,'twister'); % here, we repeat the code in rbmtrain.
    
    
    
    v1 = batchNew; % this is N x D.
    
    if isequal(rbm.types{1},'binary')
        h1 = sigm(  (1/(sigma^2))  *  (v1 * rbm.W' +repmat(rbm.c', 1, 1) )     );
%         h1 = sigm(  (1/(sigma^2))  *  (rbm.W*v1' +rbm.c )     )';
        %             h1 = sigmrnd(v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1));
    else
        h1 = v1 * rbm.W' + repmat(rbm.c', 1, 1);
    end
    
    if hintonFlag && rbm.lateralVisible && all(rbm.lateralVisibleMask(:) == false) % save last states, to match hinton
        rbm.xlast{l} = h1;
    end
    
%     c1 = (v1'*h1)'; % positive statistics.
    
    %             c1_alter = h1'*v1;
    %             assert(isequal(c1_alter,c1));
    
    poshidact   = sum(h1);
    posvisact = sum(batchNew);
    
    
    if isfield(rbm,'lateralVisible') && rbm.lateralVisible
        posvisvis = (v1')*v1;
        % this thing should not be halved later, since each entry in LV
        % is the full weight, not halved version.
        % should write something naive to check my above statement.
    end
    
    
    h2 = h1;
    
    for iCDIter = 1:params.CDIter % do CDIter-CD.
        % h2 is the hidden layer probability from last step of CD, or
        % the probability from positive phase (for first step in CD).
        % following Hinton's recommendation, we should always
        % reconstruct based on a 0-1 sampled hidden layer, but when
        % infer hidden from visible, the visible can be real-valued.
        
        
        if isequal(rbm.types{1},'binary') %stochastic sample based on h2.
            h2Sampled = h2 > rand(size(h1));
            %             h1Sampled = h1;
        else
            h2Sampled = h2 + sigma*randn(size(h1)); % sigma is the std deviation
            % why my current implementation fail to match old one on
            % vanHateren_Lee
        end
        
        
        % reconstruction
        if isequal(rbm.types{2},'binary')
            v2 = sigm(  (1/(sigma^2))*    (h2Sampled * rbm.W+repmat(rbm.b', 1, 1))         );
            %             v2 = sigmrnd(h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1));
        else
            v2 = h2Sampled * rbm.W+repmat(rbm.b', 1, 1);
            % let's try sampled version...
            %             v2 = h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1)...
            %                  + (sigma^2)*randn(rbm.batchsize, size(rbm.W,2)) ; ZYM:
            %                  why my current result on vanHateren_Lee failed to match
            %                  old one.
        end
        
        
        % the input v2 is the initial mean of v2.
        if isfield(rbm,'lateralVisible') && rbm.lateralVisible
            % update v2 using mean-field so that they somewhat converge
            [v2] = rbm_meanfield(v2, h2Sampled, rbm, sigma);
            %                 disp(MFiter);
        end
        
        
        if isequal(rbm.types{1},'binary') % h2 is probability, and h2Sampled is stochastic version.
            h2 = sigm(   (1/(sigma^2)) *(v2 * rbm.W' + repmat(rbm.c', 1, 1))   );
        else
            h2 = v2 * rbm.W' + repmat(rbm.c', 1, 1);
        end
        
    end
    
    
    
    
    
    assert(max(abs(h1_GBM' - h1))<1e-10); % this should be very small.
    
    assert(max(abs(v2_GBM' - v2))<1e-10); % this should be very small.
    assert(max(abs(h2_GBM' - h2))<1e-10); % this should be very small.
    
%     disp(max(abs(h1_GBM' - h1))); % this should be very small.
%     
%     disp(max(abs(v2_GBM' - v2))); % this should be very small.
%     disp(max(abs(h2_GBM' - h2))); % this should be very small.
    

end

% how to check h2?....



