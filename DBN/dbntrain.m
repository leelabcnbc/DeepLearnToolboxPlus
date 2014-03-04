function dbn = dbntrain(dbn, x)
    n = numel(dbn.rbm);

    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x);
    for i = 2 : n
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x);
    end

end
