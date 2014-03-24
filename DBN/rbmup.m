function x = rbmup(rbm, x, sigmaNew)

if isfield(rbm,'sigmaFinal')
    sigma = rbm.sigmaFinal;
else
    sigma = 1;
end

if nargin >= 3 && ~isempty(sigmaNew)
    sigma = sigmaNew;
end

m = size(x, 1);

if isequal(rbm.types{1},'binary')
    x = sigm(  (1/(sigma^2))  *  (x * rbm.W' + repmat(rbm.c', m, 1) ));
else
    x = x * rbm.W' + repmat(rbm.c', m, 1);
end

end
