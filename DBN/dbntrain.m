function dbn = dbntrain(dbn, x, hintonFlag, saveFlag, fileName)

n = numel(dbn.rbm);

if nargin < 4
    saveFlag = false;
end

if nargin < 5
    fileName = repmat({[]},n,1);
end

if nargin < 3
    hintonFlag = false;
end

if iscellstr(fileName)
    assert(length(fileName) == n || length(fileName) == 1);
    if length(fileName) == 1
        fileName = repmat(fileName,n,1);
    end
else
    fileName = repmat({fileName}, n,1);
end

if hintonFlag
    rng(0,'twister'); % reset random state to get reproducible results.
end

dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, hintonFlag, saveFlag, fileName{1}); % this is not good... just ad hoc
for i = 2 : n
    if ~hintonFlag
        x = rbmup(dbn.rbm{i - 1}, x);
    else
        x = cell2mat(dbn.rbm{i - 1}.xlast); % for testing hinton's code
        rng(0,'twister'); % reset random state to get reproducible results.
    end
    dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, hintonFlag, saveFlag, fileName{i});
end

end
