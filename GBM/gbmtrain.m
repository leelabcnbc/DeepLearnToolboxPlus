function gbm = gbmtrain(gbm, xData, yData)
% GBMTRAIN train a gated boltzmann machine.
%
%   ...
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 16-May-2014 16:02:33 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : gbmtrain.m

assert(isfloat(xData), 'xData must be a float');
assert(isfloat(yData), 'yData must be a float');

m = size(xData, 1);

xSize = gbm.xSize;
ySize = gbm.ySize;
hSize = gbm.hSize;

% this params is the parameter struct for the inner function to collect
% statistics.

params = struct();
params.xSize = xSize;
params.ySize = ySize;
params.hSize = hSize;
params.CDIter = gbm.CDIter;
params.sigma = gbm.sigma;

if isfield(gbm,'nonZeroMask')
    params.nonZeroMask = gbm.nonZeroMask;
end

% check dimension match
assert(isequal(size(xData),[m,xSize]));
assert(isequal(size(yData),[m,ySize]));

numbatches = m / gbm.batchsize;

assert(rem(numbatches, 1) == 0, 'numbatches not integer');

display(gbm);
display(params);

initMultiplierW = gbm.initMultiplierW; % most of the time this is 0.01

if ~isfield(gbm,'initialized') || ~gbm.initialized
    % add an initialization phase.
    %     gbm.W = initMultiplierW*randn(size(rbm.W,2),size(rbm.W,1))'; % follow the initialization of hinton
    % 0.1 can be changed. In Ruslan's DBM code, this value can be sometimes
    % 0.01, sometimes 0.001.
    
    % initialize 6 sets of weights.
    
    gbm.W_xyh = initMultiplierW*randn([xSize,ySize,hSize]);
    gbm.W_yh = initMultiplierW*randn([ySize,hSize]);
    gbm.W_xy = initMultiplierW*randn([xSize,ySize]);
    gbm.W_xh = initMultiplierW*randn([xSize,hSize]);
    gbm.W_y = initMultiplierW*randn([ySize,1]);
    gbm.W_h = initMultiplierW*randn([hSize,1]);
    
    %
end





% pack things together.

W_all = [gbm.W_xyh(:);gbm.W_yh(:);gbm.W_xy(:);gbm.W_xh(:);gbm.W_y(:);gbm.W_h(:)];

if isfield(gbm,'nonZeroMask')
    W_all(~gbm.nonZeroMask) = 0; % special value...
end

vW = zeros(size(W_all));
for i = 1 : gbm.numepochs
    tic;
    
    if ~gbm.batchOrderFixed
        kk = randperm(m);
    else
        kk = 1:m;
    end
    
    h1_mean = 0;
    err_mean = 0;
    
    for lBatch = 1 : numbatches
        batchX = xData(kk((lBatch - 1) * gbm.batchsize + 1 : lBatch * gbm.batchsize), :);
        batchY = yData(kk((lBatch - 1) * gbm.batchsize + 1 : lBatch * gbm.batchsize), :);
        
        % collect statistics
        
        [~, dWRaw, h1,recon_error] = bm.L_dL_gbm_naive(W_all,batchX, batchY, params,'CD');
        
        h1_mean = h1_mean + (h1);
        err_mean = err_mean + recon_error;
        
        % update parameters
        dWRaw = -dWRaw;
        
        
        if i > gbm.epochFinal
            momentum=gbm.momentumFinal;
            alpha = gbm.alphaFinal;
        else
            momentum=gbm.momentum;
            alpha = gbm.alpha;
        end
        
        if mod(lBatch,50) == 0
            disp(['dW norm: ' num2str( mean((vW(:)).^2) ) ]);
            disp(['h1 sparsity: ' num2str( h1_mean'/50) ]);
            disp(['recon_error: ' num2str(err_mean/50) ]);
            h1_mean = 0;
            err_mean = 0;
        end
        
        vW = momentum * vW + alpha * (dWRaw - gbm.weightPenaltyL2*W_all);
        
        if isfield(gbm,'nonZeroMask')
            vW(~gbm.nonZeroMask) = 0; % special value...
        end

        assert(all(~isnan(vW(:))));
        
        W_all = W_all + vW;
        
        
        %         fprintf('batch %d/%d for epoch %d\n',lBatch,numbatches,i);
    end
    if i > gbm.epochFinal
        fprintf('finally!, with alpha %f, momentum %f\n', alpha, momentum);
    else
        fprintf('initial!, with alpha %f, momentum %f\n', alpha, momentum);
    end
    fprintf('sigma %f\n',params.sigma);
    toc;
end

% pack back...

gbm.W_xyh = W_all(1:xSize*ySize*hSize);
gbm.W_yh = W_all(xSize*ySize*hSize+1:xSize*ySize*hSize+ySize*hSize);
gbm.W_xy = W_all(xSize*ySize*hSize+ySize*hSize+1: xSize*ySize*hSize+ySize*hSize + xSize*ySize);
gbm.W_xh = W_all(xSize*ySize*hSize+ySize*hSize+xSize*ySize+1:xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize);
gbm.W_y =  W_all(xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize+1:xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize+ySize);
gbm.W_h =  W_all(xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize+ySize+1:xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize+ySize+hSize);



% reshape everything...
gbm.W_xyh = reshape(gbm.W_xyh,[xSize,ySize,hSize]);
gbm.W_yh = reshape(gbm.W_yh,[ySize,hSize]);
gbm.W_xy = reshape(gbm.W_xy,[xSize,ySize]);
gbm.W_xh = reshape(gbm.W_xh,[xSize,hSize]);
gbm.W_y = reshape(gbm.W_y,[ySize,1]);
gbm.W_h = reshape(gbm.W_h,[hSize,1]);

% for safety check.
gbm.W_all = W_all;

end







% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [gbmtrain.m] ======
