function dbn = dbnsetup(dbn, x, opts)
n = size(x, 2);
dbn.sizes = [n, dbn.sizes];

if isfield(dbn,'types') % assign each layer a type
    assert(numel(dbn.types) == numel(dbn.sizes),'no type or all specified');
else % we create it.
    dbn.types = repmat({'binary'},1,numel(dbn.sizes));
end

% alpha, momentum, final momentum, etc should be converted to
% vectors...

for u = 1 : numel(dbn.sizes) - 1
    dbn.rbm{u}.alpha    = opts.alpha(u); % required parameter.
    dbn.rbm{u}.momentum = opts.momentum(u);
    dbn.rbm{u}.batchsize = opts.batchsize(u);
    dbn.rbm{u}.numepochs = opts.numepochs(u);
    
    
    if isfield(opts,'initMultiplierW')
        dbn.rbm{u}.initMultiplierW = opts.initMultiplierW(u);
    else
        dbn.rbm{u}.initMultiplierW = 0.1; %0.1*rand(sizeW,sizeV);
    end
    
    
    
    % I think I should better first do an opts thing to fill in default
    % values first, but MATLAB just sucks...
    if isfield(opts,'weightPenaltyL2') % add weight penalty, vectorized version possible in the future
        dbn.rbm{u}.weightPenaltyL2 = opts.weightPenaltyL2(u);
    else
        dbn.rbm{u}.weightPenaltyL2 = 0;
    end
    
    if isfield(opts,'momentumFinal')
        dbn.rbm{u}.momentumFinal = opts.momentumFinal(u);
    else
        dbn.rbm{u}.momentumFinal = dbn.rbm{u}.momentum; %keep the old momentum
    end
    
    if isfield(opts,'alphaFinal')
        dbn.rbm{u}.alphaFinal = opts.alphaFinal(u);
    else
        dbn.rbm{u}.alphaFinal = dbn.rbm{u}.alpha; %keep the old alpha
    end
    
    if isfield(opts,'epochFinal')
        dbn.rbm{u}.epochFinal = opts.epochFinal(u);
    else
        dbn.rbm{u}.epochFinal = inf; %never that
    end
    
    if isfield(opts,'batchOrderFixed')
        dbn.rbm{u}.batchOrderFixed = opts.batchOrderFixed(u);
    else
        dbn.rbm{u}.batchOrderFixed = false; %never that
    end
    
    if isfield(opts,'sparsityTarget')
        dbn.rbm{u}.sparsityTarget = opts.sparsityTarget(u);
    else
        dbn.rbm{u}.sparsityTarget = NaN; %no meaning
    end
    
    
    if isfield(opts,'nonSparsityPenalty')
        dbn.rbm{u}.nonSparsityPenalty = opts.nonSparsityPenalty(u);
    else
        dbn.rbm{u}.nonSparsityPenalty = 0; %no sparsity penalty
    end
    
    if isfield(opts,'sigma')
        dbn.rbm{u}.sigma = opts.sigma(u);
    else
        dbn.rbm{u}.sigma = 1; %no scaling
    end
    
    if isfield(opts,'sigmaDecay')
        dbn.rbm{u}.sigmaDecay = opts.sigmaDecay(u);
    else
        dbn.rbm{u}.sigmaDecay = 1; %no sigma decay
    end
    
    if isfield(opts,'sigmaMin')
        dbn.rbm{u}.sigmaMin = opts.sigmaMin(u);
    else
        dbn.rbm{u}.sigmaMin = inf; %never do decay
    end
    
    
    if isfield(opts,'CDIter')  % in default, use CD-1.
        dbn.rbm{u}.CDIter = opts.CDIter(u);
    else
        dbn.rbm{u}.CDIter = 1;
    end
    
    if isfield(opts,'lateralVisible')  % in default, use CD-1.
        dbn.rbm{u}.lateralVisible = opts.lateralVisible(u);
        
        
        if isfield(opts,'lateralVisibleMFIter')
            dbn.rbm{u}.lateralVisibleMFIter = opts.lateralVisibleMFIter(u);
        else
            dbn.rbm{u}.lateralVisibleMFIter = 10; % value in Ruslan's MF code
        end
        
        
        if isfield(opts,'lateralVisibleMFDamp')
            dbn.rbm{u}.lateralVisibleMFDamp = opts.lateralVisibleMFDamp(u);
        else
            dbn.rbm{u}.lateralVisibleMFDamp = 0.2; % value in Hinton's semi RBM paper.
        end
        
        if isfield(opts,'alphaLateral')
            dbn.rbm{u}.alphaLateral = opts.alphaLateral(u);
        else
            dbn.rbm{u}.alphaLateral = 0.5*dbn.rbm{u}.alpha; % half of normal alpha.
        end
        
        if isfield(opts,'lateralVisibleMask')
            dbn.rbm{u}.lateralVisibleMask = generate_lateral_mask(dbn.sizes(u),opts.lateralVisibleMask{u});
        else
            dbn.rbm{u}.lateralVisibleMask = generate_lateral_mask(dbn.sizes(u),'all');
        end
        
        
        if isfield(opts,'initMultiplierLV')
            dbn.rbm{u}.initMultiplierLV = opts.initMultiplierLV(u);
        else
            dbn.rbm{u}.initMultiplierLV = 0.1; %0.1*rand(sizeV,sizeV);
        end
        
        
    else
        dbn.rbm{u}.lateralVisible = false; % by default, no lateral.
    end
    
    
    if isfield(opts,'visualize')
        dbn.rbm{u}.visualize = opts.visualize(u);
    else
        dbn.rbm{u}.visualize = false; %no visualization
    end
    
    
    dbn.rbm{u}.types = {dbn.types{u + 1},  dbn.types{u}};
    
    dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));
    
    dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
    dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);
    
    dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
    dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    
    if isfield(dbn.rbm{u},'lateralVisible') && dbn.rbm{u}.lateralVisible
        dbn.rbm{u}.LV = zeros(dbn.sizes(u),dbn.sizes(u));
        dbn.rbm{u}.vLV = zeros(dbn.sizes(u),dbn.sizes(u));
        
        % TODO: write small routines to generate mask matrix.
    end
    
end

end
