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
vW = zeros(size(W_all));
for i = 1 : rbm.numepochs
    tic;

    if ~gbm.batchOrderFixed
        kk = randperm(m);
    else
        kk = 1:m;
    end
    
    for lBatch = 1 : numbatches
        batchX = xData(kk((lBatch - 1) * gbm.batchsize + 1 : lBatch * gbm.batchsize), :);
        batchY = yData(kk((lBatch - 1) * gbm.batchsize + 1 : lBatch * gbm.batchsize), :);
        
        % collect statistics 
        
        [~, dWRaw] = bm.L_dL_gbm_naive(W_all,batchX, batchY, params,'CD');

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
        end
        
        vW = momentum * vW + alpha * dWRaw;
        
        assert(all(~isnan(vW(:))));
    
        W = W + vW;
    end
    if i > gbm.epochFinal
        fprintf('finally!, with alpha %f, momentum %f\n', alpha, momentum);
    else
        fprintf('initial!, with alpha %f, momentum %f\n', alpha, momentum);
    end
    fprintf('sigma %f\n',params.sigma);
    toc;
end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [gbmtrain.m] ======
