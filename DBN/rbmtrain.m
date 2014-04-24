function rbm = rbmtrain(rbm, x, hintonFlag, saveFlag, fileName)
assert(isfloat(x), 'x must be a float');
% assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');

if nargin < 4
    saveFlag = false;
end

m = size(x, 1);

numbatches = m / rbm.batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% can be commented out

if nargin < 3
    hintonFlag = false;
end


% add an initialization phase.
rbm.W = 0.1*randn(size(rbm.W,2),size(rbm.W,1))'; % follow the initialization of hinton


if hintonFlag
    rbm.xlast = cell(numbatches,1);
end

gaussianLayerCount = 0;
for i = 1:2
    assert(isequal(rbm.types{i},'binary') || ...
        isequal(rbm.types{i},'gaussian'));
    if isequal(rbm.types{i},'gaussian')
        gaussianLayerCount = gaussianLayerCount+1;
    end
end
assert(gaussianLayerCount<=1);

sigma = rbm.sigma;

for i = 1 : rbm.numepochs
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        sparsity = 0;
    end
    
    if ~rbm.batchOrderFixed
        kk = randperm(m);
    else
        kk = 1:m;
    end
    err = 0;
    
    for l = 1 : numbatches
        batch = x(kk((l - 1) * rbm.batchsize + 1 : l * rbm.batchsize), :);
        
        v1 = batch;
        
        if isequal(rbm.types{1},'binary')
            h1 = sigm(  (1/(sigma^2))  *  (v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1) )     );
            %             h1 = sigmrnd(v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1));
        else
            h1 = v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
        end
        
        if hintonFlag % save last states, to match hinton
            rbm.xlast{l} = h1;
        end
        
        c1 = (v1'*h1)'; % positive statistics.
        
        %             c1_alter = h1'*v1;
        %             assert(isequal(c1_alter,c1));
        
        poshidact   = sum(h1);
        posvisact = sum(batch);
        
        h2 = h1;
        
        for iCDIter = 1:rbm.CDIter % do CDIter-CD.
            % h2 is the hidden layer probability from last step of CD, or
            % the probability from positive phase (for first step in CD).
            % following Hinton's recommendation, we should always
            % reconstruct based on a 0-1 sampled hidden layer, but when
            % infer hidden from visible, the visible can be real-valued.
            
            if isequal(rbm.types{1},'binary')
                h2Sampled = h2 > rand(size(h1));
                %             h1Sampled = h1;
            else
                h2Sampled = h2 + sigma*randn(size(h1)); % sigma is the std deviation
            end
            
            
            % reconstruction
            if isequal(rbm.types{2},'binary')
                v2 = sigm(  (1/(sigma^2))*    (h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1))         );
                %             v2 = sigmrnd(h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1));
            else
                v2 = h2Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1);
                % let's try sampled version...
                %             v2 = h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1)...
                % + (sigma)*randn(rbm.batchsize, size(rbm.W,2)) ;
            end
            
            
            if isequal(rbm.types{1},'binary') % h2 is probability, and h2Sampled is stochastic version.
                h2 = sigm(   (1/(sigma^2)) *(v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1))     );
            else
                h2 = v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
            end
            
        end
        
        c2 = (v2'*h2)'; % this can affect result?!!! order of computation...
        
        %             c2_alter = h2'*v2;
        %             assert(isequal(c2_alter,c2)); this can indeed affect
        %             computation...
        
        if i > rbm.epochFinal
            momentum=rbm.momentumFinal;
            alpha = rbm.alphaFinal;
        else
            momentum=rbm.momentum;
            alpha = rbm.alpha;
        end
        
        neghidact = sum(h2);
        negvisact = sum(v2);
        
        rbm.vW = momentum * rbm.vW + alpha * ((c1 - c2)/rbm.batchsize - rbm.weightPenaltyL2*rbm.W);
        rbm.vb = momentum * rbm.vb + alpha/rbm.batchsize * ((posvisact-negvisact)');
        % put division first, so that we may have better accuracy?
        rbm.vc = momentum * rbm.vc + alpha/rbm.batchsize * ((poshidact-neghidact)');
        
        
        assert(all(~isnan(rbm.vW(:))));
        assert(all(~isnan(rbm.vb(:))));
        assert(all(~isnan(rbm.vc(:))));
        
        rbm.W = rbm.W + rbm.vW;
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;
        
        
        
        
        err = err + sum(sum((v1 - v2) .^ 2)) / rbm.batchsize;
        
    end
    
    if i > rbm.epochFinal
        fprintf('finally!, with alpha %f, momentum %f\n', alpha, momentum);
    else
        fprintf('initial!, with alpha %f, momentum %f\n', alpha, momentum);
    end
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        assert(isfield(rbm,'sparsityTarget'));
        assert(isequal(rbm.types{1},'binary')); % only works for binary hidden layer
        v1All = x;
        
        if isequal(rbm.types{1},'binary')
            h1All = sigm( (1/(sigma^2))  *  (v1All * rbm.W' + repmat(rbm.c', m, 1) )     );
        else
            h1All = v1All * rbm.W' + repmat(rbm.c', m, 1);
        end
        
        %         sparsityGradientSecondTerm = sum(h1All.*(1-h1All),1);
        sparsityGradientFirstTerm = (rbm.sparsityTarget - mean(h1All,1));
        sparsity = mean(h1All(:));
        if isnan(sparsity)
            fprintf('what the fuck!\n');
        end
    end
    
    
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        %         rbm.c = rbm.c + (rbm.nonSparsityPenalty/m) *...
        %             (sparsityGradientFirstTerm.*sparsityGradientSecondTerm)';
        fprintf('simple form!\n');
        rbm.c = rbm.c + (rbm.nonSparsityPenalty) *... %no 1/m
            (sparsityGradientFirstTerm)';
    end
    
    fprintf('sigma %f\n',sigma);
    
    rbm.sigmaFinal = sigma; % save the current sigma...
    
    if sigma > rbm.sigmaMin % sigma decay in Honglak
        sigma = sigma*rbm.sigmaDecay; %sigmaDecay is 1 by default.
    end
    
    disp(['epoch ' num2str(i) '/' num2str(rbm.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
        fprintf('sparsity is %f\n', sparsity);
    end
    
    
    if saveFlag
        errAvg = err / numbatches;
        save(fileName,'rbm','sparsity','i','errAvg');
    end
    
end



end
