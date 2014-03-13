function dbn = dbntrain(dbn, x, hintonFlag)

if nargin < 3
    hintonFlag = false;
end

n = numel(dbn.rbm);

if hintonFlag
    rng(0,'twister'); % reset random state to get reproducible results.
end

dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x);
for i = 2 : n
    if ~hintonFlag
        x = rbmup(dbn.rbm{i - 1}, x);
    else
        x = cell2mat(dbn.rbm{i - 1}.xlast); % for testing hinton's code
        rng(0,'twister'); % reset random state to get reproducible results.
    end
    dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x);
end

end
