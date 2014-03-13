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
        dbn.rbm{u}.alpha    = opts.alpha(u);
        dbn.rbm{u}.momentum = opts.momentum(u);
        dbn.rbm{u}.batchsize = opts.batchsize(u);
        dbn.rbm{u}.numepochs = opts.numepochs(u);
        
        
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
            dbn.rbm{u}.momentumFinal = NaN; %never that
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
        
        
        dbn.rbm{u}.types = {dbn.types{u + 1},  dbn.types{u}};

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
