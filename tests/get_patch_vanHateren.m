function get_patch_vanHateren(fileLabels, patchesPerImage, windowSize)
% GET_PATCH_VANHATEREN ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 15-Mar-2014 12:04:30 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : get_patch_vanHateren.m 

train_x = cell(length(fileLabels),1);


if nargin < 2
    patchesPerImage = 30;
end

if nargin < 3
    windowSize = 14;
end

thisLabel = fileLabels(1);
fileName = sprintf('vanHaterenIMC_%02d_pre.mat',thisLabel);
vanHaterenPre = load(fileName);

numberOfImagesToAdd = size(vanHaterenPre.images,3);

train_x{1} = preprocessing.getdata_imagearray(vanHaterenPre.images,...
    windowSize,numberOfImagesToAdd*patchesPerImage);
disp(1);
for iLabel = 2:length(fileLabels)
    thisLabel = fileLabels(iLabel);
    fileName = sprintf('vanHaterenIMC_%02d_pre.mat',thisLabel);
    vanHaterenPreNext = load(fileName);
    
    assert(vanHaterenPre.epsilon == vanHaterenPreNext.epsilon);
    assert(vanHaterenPre.avgVar == vanHaterenPreNext.avgVar);
    assert(vanHaterenPre.gammaCorrection == vanHaterenPreNext.gammaCorrection);
    
    numberOfImagesToAdd = size(vanHaterenPreNext.images,3);
    train_x{iLabel} = preprocessing.getdata_imagearray(vanHaterenPreNext.images,...
        windowSize,numberOfImagesToAdd*patchesPerImage);
    disp(iLabel);
end

epsilon = vanHaterenPre.epsilon;
avgVar = vanHaterenPre.avgVar;
gammaCorrection = vanHaterenPre.gammaCorrection;

train_x = cell2mat(train_x);

fileNameMat = ['vanHaterenPatch_' datestr(now,30) '.mat'];

save(fileNameMat, 'train_x', 'windowSize', 'patchesPerImage', 'fileLabels',...
    'epsilon','avgVar','gammaCorrection');

end







% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [get_patch_vanHateren.m] ======  
