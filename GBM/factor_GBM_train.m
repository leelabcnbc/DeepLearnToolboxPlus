function pars = factor_GBM_train(varargin)
% FACTOR_GBM_TRAIN ...
%
%   a MATLAB implementation of Roland's CPU GBM.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

% FACTOR_GBM_TRAIN(train_x, train_y, ...)

%% DATE      : 28-May-2014 16:17:01 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : factor_GBM_train.m

% format for varargin:
% numin, numout, nummap, numfactors, sparsitygain, targethidprobs,
% cditerations, meanfield_output, momentum, stepsize, verbose
% zeromask (if there's 1, then corresponding weights are always zero).

% you should first input train_x, then input train_y, and afterwards, we
% have named arguments.

% not supported currently: sigma^2. (I can't figure out the energy
% function).

train_x = varargin{1};
train_y = varargin{2};

assert(ismatrix(train_x));
assert(ismatrix(train_y));

[N,numin] = size(train_x);
numout = size(train_y,2);

assert(N == size(train_y,1));

pars = factor_GBM_default_pars();
% fill in other values, along with numin and numout, inferred from data.
pars = parseArgs([varargin(3:end),{'numin'},{numin},{'numout'},{numout} ]  ,pars);
assert(isempty(pars.NumericArguments));
pars = rmfield(pars,'NumericArguments');

datestring = datestr(now,30);
pars.datestring = datestring;

assert(rem(N,pars.batchsize)==0);
numbatches = N/pars.batchsize;
everySave = pars.everySave;

% intialize seed.
if isempty(pars.seed)
    rng('shuffle');
else
    rng(pars.seed,'twister');
end


if ~isempty(pars.numbatches) % hack for early quitting.
    numbatches = pars.numbatches;
end

pars.numbatches = numbatches;

% initialize weights.
if isempty(pars.wxf) % in case we can override this.
    pars.wxf = pars.initMultiplierW*randn(pars.numin,pars.numfactors);
end

if isempty(pars.wyf)
    pars.wyf = pars.initMultiplierW*randn(pars.numout,pars.numfactors);
end

if isempty(pars.whf)
    pars.whf = pars.initMultiplierW*randn(pars.nummap,pars.numfactors);
end

if isempty(pars.wy)
    pars.wy = zeros(pars.numout,1);
end

if isempty(pars.wh)
    pars.wh = zeros(pars.nummap,1);
end

pars.inc = zeros(  (pars.numin+pars.numout+pars.nummap)*pars.numfactors + pars.numout + pars.nummap,1 );

% initialize zeromask
if isequal(pars.zeromask,'none')
    pars.zeromask = false(size(pars.inc));
end

for epoch = 1:pars.numepoch
    fprintf('epoch %d\n',epoch);
    if ~pars.batchOrderFixed
        kk = randperm(N);
    else
        kk = 1:N;
    end
    
    for batch = 1:numbatches
        batch_x = train_x(kk((batch - 1) * pars.batchsize + 1 : batch * pars.batchsize), :);
        batch_y = train_y(kk((batch - 1) * pars.batchsize + 1 : batch * pars.batchsize), :);
        factor_GBM_train_inner(); % here, I use a nested function for efficieny issue.
    end
    
    if rem(epoch,everySave) == 0 || epoch == pars.numepoch
        if pars.saveFile
            fileName = [datestring '_' int2str(epoch) '.mat'];
            pars = rmfield(pars,'hids'); % save space...
            save(fileName,'pars','epoch'); % remember delete pars.hids before!
        end
    end
    
end

    function factor_GBM_train_inner()
        gradThis = factor_GBM_train_inner_grad();
        pars.inc = pars.momentum*pars.inc - pars.stepsize*gradThis;
        ninc =norm(pars.inc);
        fprintf('norm of inc: %f\n',ninc);
        
        % stablize the things...
        if norm(pars.inc) > pars.incMax
            pars.inc = pars.inc/ninc * pars.incMax;
            fprintf('norm of inc again: %f\n',norm(pars.inc));
        end
        
        assert(all(~isnan(pars.inc)));
        
        % here, I should add them separately.
        pars.wxf(:) = pars.wxf(:) + pars.inc( 1:pars.numin*pars.numfactors );
        pars.wyf(:) = pars.wyf(:) + pars.inc( pars.numin*pars.numfactors+1 : (pars.numin+pars.numout)*pars.numfactors );
        pars.whf(:) = pars.whf(:) + pars.inc( (pars.numin+pars.numout)*pars.numfactors+1 : (pars.numin+pars.numout+pars.nummap)*pars.numfactors );
        
        pars.wy(:) = pars.wy(:) + pars.inc( (pars.numin+pars.numout+pars.nummap)*pars.numfactors+1 : ...
            (pars.numin+pars.numout+pars.nummap)*pars.numfactors + pars.numout);
        
        pars.wh(:) = pars.wh(:) + pars.inc( (pars.numin+pars.numout+pars.nummap)*pars.numfactors + pars.numout + 1 : end );
        
    end

    function grad = factor_GBM_train_inner_grad()
        positiveGrad = factor_GBM_energy_grad(factor_GBM_posdata());
        
        if pars.sparsitygain > 0
            sparsityGrad = factor_GBM_sparsityGrad();
        else
            sparsityGrad = 0;
        end
        
        negativeGrad = factor_GBM_energy_grad(factor_GBM_negdata());
        
        grad = -positiveGrad + negativeGrad;
        grad = grad + sparsityGrad;
        
        weightcostgrad_x = pars.weightPenaltyL2 * pars.wxf(:);
        weightcostgrad_y = pars.weightPenaltyL2 * pars.wyf(:);
        weightcostgrad_h = pars.weightPenaltyL2 * pars.whf(:);
        
        weightcostgrad = [weightcostgrad_x; weightcostgrad_y; weightcostgrad_h];
        
        grad(1:(pars.numin+pars.numout+pars.nummap)*pars.numfactors) = ...
            grad(1:(pars.numin+pars.numout+pars.nummap)*pars.numfactors) + weightcostgrad;
    end

    function data = factor_GBM_posdata()
        data = struct();
        data.inputs = batch_x;
        data.outputs = batch_y;
        data.hidprobs = factor_GBM_hidprobs(batch_y,batch_x);
        pars.hids = data.hidprobs; % this is just used for further use of negative phase.
        return;
    end

    function hidprobs = factor_GBM_hidprobs(outputs,inputs) % this is used in CD as well.
        % we just assume binary hidden unit.
        factors_x = inputs*pars.wxf;
        factors_y = outputs*pars.wyf;
        hidprobs = sigm(bsxfun(@plus, (factors_x.*factors_y)*(pars.whf'), pars.wh'));  %  [N x numfactor] x [numfactor x nummap]
    end

    function data = factor_GBM_negdata()
        data = struct();
        data.inputs = batch_x;
        
        for iCD = 1:pars.cditerations
            if ~pars.meanfield_output
                hidstates = double(pars.hids > rand(size(pars.hids )));
            else
                hidstates = pars.hids;
            end
            negoutput = factor_GBM_outprobs(hidstates,batch_x);
            datastates = factor_GBM_sample_obs(negoutput);
            pars.hids = factor_GBM_hidprobs(datastates,batch_x);
        end
        
        %         data.outputs = negoutput;
        data.outputs = datastates; % seems this is crucial!!!
        data.hidprobs = pars.hids;
        
        if pars.verbose % output recon and norm
            fprintf('mean square error: %f\n', sum( (datastates(:)-batch_y(:)).^2) / pars.batchsize   );
            fprintf('mean norm of w: %f\n', norm([pars.wxf(:); pars.wyf(:); pars.whf(:); pars.wh(:); pars.wy(:)]));
        end
        
    end

    function datastates = factor_GBM_sample_obs(negoutput)
        if ~pars.meanfield_output
            if isequal(pars.visType, 'binary')
                datastates = double( negoutput > rand(size(negoutput)));
            else
                assert(isequal(pars.visType, 'gaussian'));
                datastates = negoutput + randn(size(negoutput)); % normal random variable with sigma^2 = 1.
            end
        else
            datastates = negoutput;
        end
    end

    function negoutput = factor_GBM_outprobs(hidstates,inputs)
        % negoutput should be a N x numout matrix.
        factors_x = inputs*pars.wxf;
        factors_h = hidstates*pars.whf;
        
        negoutput = (bsxfun(@plus, (factors_x.*factors_h)*(pars.wyf'), pars.wy'));  %  [N x numfactor] x [numfactor x numout]
        
        if isequal(pars.visType, 'binary')
            negoutput = sigm(negoutput);
        end
    end

    function grad =  factor_GBM_energy_grad(data)
        % all three are of size N x blablabla.
        inputs = data.inputs;
        hidprobs = data.hidprobs;
        outputs = data.outputs;
        
        % they have shape N x numfactor, each element being the dot product
        % between filter and the input (this input may be sampled).
        
        factors_x = inputs*pars.wxf;
        factors_y = outputs*pars.wyf;
        factors_h = hidprobs*pars.whf;
        
        grady = mean(outputs,1)';
        gradh = mean(hidprobs,1)';
        
        grad_wxf = (inputs')*(factors_y.*factors_h); %  it's  [numin x N] * [N x numfactor]
        grad_wyf = (outputs')*(factors_x.*factors_h);
        grad_whf = (hidprobs')*(factors_x.*factors_y);
        
        grad_wxf = grad_wxf/pars.batchsize;
        grad_wyf = grad_wyf/pars.batchsize;
        grad_whf = grad_whf/pars.batchsize;
        
        grad = [grad_wxf(:);grad_wyf(:);grad_whf(:);grady(:);gradh(:)];
    end

end

function defaultPars = factor_GBM_default_pars()
% returns default value for arguments.
defaultPars = struct('numin',[],'numout',[],'nummap',256,'numfactors',1024, ...
    'sparsitygain',0.0, ...
    'targethidprobs', 0.1, 'cditerations', 1, ...
    'meanfield_output', false, 'momentum', 0.9, ...
    'stepsize', 0.01, 'verbose', true, ...
    'zeromask', 'none','batchsize',500,'numepoch',100,'seed',[],...
    'initMultiplierW',0.05,'batchOrderFixed',false,'weightPenaltyL2',0.001,...
    'everySave',1,'wxf',[],'wyf',[],'whf',[],'wy',[],'wh',[],...
    'incMax',inf,'numbatches',[],'visType','binary','saveFile',true);
% seed is random seed. (twister).
end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [factor_GBM_train.m] ======
