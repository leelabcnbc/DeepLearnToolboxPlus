function [ Wpos ] = removeSmallWeights( Wpos,k )
%REMOVESMALLWEIGHTS Summary of this function goes here
%   Detailed explanation goes here

if nargin < 2
    k = 5;
end

[~,I] = sort(Wpos,2,'descend');

for iRow = 1:size(Wpos,1) % process each row
    Ithis = I(iRow,:);
    Wpos(iRow, Ithis(k+1:end) ) = 0;
end

end

