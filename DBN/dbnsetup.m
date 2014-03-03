function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];

    if isfield(dbn,'types') % assign each layer a type
        assert(numel(dbn.types) == numel(dbn.sizes),'no type or all specified');
    else % we create it.
        dbn.types = repmat({'binary'},1,numel(dbn.sizes));
    end
    
    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;
        
        if isfield(opts,'weightPenaltyL2') % add weight penalty, vectorized version possible in the future
            dbn.rbm{u}.weightPenaltyL2 = opts.weightPenaltyL2;
        end

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
