function rbm = rbmtrain(rbm, x)
    assert(isfloat(x), 'x must be a float');
    %assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);
    numbatches = m / rbm.batchsize;
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');

    % add an initialization phase.
    rbm.W = 0.1*randn(size(rbm.W,2),size(rbm.W,1))'; % follow the initialization of hinton

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
            h1 = sigm(repmat(rbm.c', rbm.batchsize, 1) + v1 * rbm.W');
            
            c1 = h1' * v1; % positive statistics.
            
            poshidact   = sum(h1);
            posvisact = sum(batch);

            h1Sampled = h1 > rand(size(h1));
            
            % reconstruction
            v2 = sigm(repmat(rbm.b', rbm.batchsize, 1) + h1Sampled * rbm.W);
            h2 = sigm(repmat(rbm.c', rbm.batchsize, 1) + v2 * rbm.W');

            c2 = h2' * v2;

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

            err = err + sum(sum((v1 - v2) .^ 2)) / rbm.batchsize;
        end
        
        disp(['epoch ' num2str(i) '/' num2str(rbm.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end
