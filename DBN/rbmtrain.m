function rbm = rbmtrain(rbm, x)
assert(isfloat(x), 'x must be a float');
%assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
m = size(x, 1);

numbatches = m / rbm.batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches not integer');

% can be commented out

% add an initialization phase.
rbm.W = 0.1*randn(size(rbm.W,2),size(rbm.W,1))'; % follow the initialization of hinton
rbm.xlast = cell(numbatches,1);


gaussianLayerCount = 0;
for i = 1:2
    assert(isequal(rbm.types{i},'binary') || ...
        isequal(rbm.types{i},'gaussian'));
    if isequal(rbm.types{i},'gaussian')
        gaussianLayerCount = gaussianLayerCount+1;
    end
end
assert(gaussianLayerCount<=1);


for i = 1 : rbm.numepochs
    
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
            h1 = sigm(v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1));
        else
            h1 = v1 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
        end
        
        rbm.xlast{l} = h1;
        
        c1 = (v1'*h1)'; % positive statistics.
        
        %             c1_alter = h1'*v1;
        %             assert(isequal(c1_alter,c1));
        
        poshidact   = sum(h1);
        posvisact = sum(batch);
        
        if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
            assert(isfield(rbm,'sparsityTarget'));
            assert(isequal(rbm.types{1},'binary')); % only works for binary hidden layer
            sparsityGradientSecondTerm = sum(h1.*(1-h1),1);
            sparsityGradientFirstTerm = (rbm.sparsityTarget - mean(h1,1));
            
            % debug
            %             h11 = h1(:,1);
            %             sparsityGradientFirstTerm1 = (rbm.sparsityTarget - mean(h11));
            %
            %             sparsityGradientSecondTerm1 = sum(h11.*(1-h11));
            %             disp(abs(sparsityGradientSecondTerm1-sparsityGradientSecondTerm(1)));
            %             disp(abs(sparsityGradientFirstTerm1-sparsityGradientFirstTerm(1)));
        end
        
        
        
        if isequal(rbm.types{1},'binary')
            h1Sampled = h1 > rand(size(h1));
        else
            h1Sampled = h1 + randn(size(h1));
        end
        
        
        % reconstruction
        if isequal(rbm.types{2},'binary')
            v2 = sigm(h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1));
        else
            v2 = h1Sampled * rbm.W+repmat(rbm.b', rbm.batchsize, 1);
        end
        
        
        if isequal(rbm.types{1},'binary')
            h2 = sigm(v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1));
        else
            h2 = v2 * rbm.W' + repmat(rbm.c', rbm.batchsize, 1);
        end
        
        c2 = (v2'*h2)'; % this can affect result?!!! order of computation...
        
        %             c2_alter = h2'*v2;
        %             assert(isequal(c2_alter,c2)); this can indeed affect
        %             computation...
        
        if i > rbm.epochFinal
            momentum=rbm.momentumFinal;
        else
            momentum=rbm.momentum;
        end
        
        neghidact = sum(h2);
        negvisact = sum(v2);
        
        rbm.vW = momentum * rbm.vW + rbm.alpha * ((c1 - c2)/rbm.batchsize - rbm.weightPenaltyL2*rbm.W);
        rbm.vb = momentum * rbm.vb + rbm.alpha/rbm.batchsize * ((posvisact-negvisact)');
        % put division first, so that we may have better accuracy?
        rbm.vc = momentum * rbm.vc + rbm.alpha/rbm.batchsize * ((poshidact-neghidact)');
        
        rbm.W = rbm.W + rbm.vW;
        rbm.b = rbm.b + rbm.vb;
        rbm.c = rbm.c + rbm.vc;
        
        if isfield(rbm, 'nonSparsityPenalty') && rbm.nonSparsityPenalty~=0
            rbm.c = rbm.c + (rbm.nonSparsityPenalty/rbm.batchsize) *...
                (sparsityGradientFirstTerm.*sparsityGradientSecondTerm)';
        end
        
        err = err + sum(sum((v1 - v2) .^ 2)) / rbm.batchsize;
    end
    
    disp(['epoch ' num2str(i) '/' num2str(rbm.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
    
end
end
