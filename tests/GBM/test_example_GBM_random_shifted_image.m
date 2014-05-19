function test_example_GBM_random_shifted_image()
% TEST_EXAMPLE_GBM_RANDOM_SHIFTED_IMAGE ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 16-May-2014 18:01:00 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : test_example_GBM_random_shifted_image.m 

% better fix random seed.

rng(0,'twister');

load('gbm_image_data.mat');
% load('gbm_image_data_prob05.mat');


gbm = struct();
gbm.xSize = sideSize*sideSize;
gbm.ySize = sideSize*sideSize;
gbm.hSize = 10; % or 20?

gbm.CDIter = 1;
gbm.batchsize = 100;
gbm.initMultiplierW = 0.01;
gbm.sigma = 0.75;

gbm.initialized = false;

gbm.numepochs = 1;
gbm.momentumFinal = 0.5;
gbm.alphaFinal = 0.5;
gbm.momentum = 0.5;
gbm.alpha = 0.5;
gbm.epochFinal = 100;
gbm.batchOrderFixed = true;
gbm.weightPenaltyL2 = 1e-2;

% well, let's add zeromask so that not that many weights are interacting
% with x.

xSize = sideSize*sideSize;
ySize = sideSize*sideSize;
hSize = gbm.hSize;


gbm.nonZeroMask = true(xSize*ySize*hSize+ySize*hSize + xSize*ySize+xSize*hSize+ySize+hSize,1);

% W_xy and W_xh.
gbm.nonZeroMask(xSize*ySize*hSize+ySize*hSize+1: xSize*ySize*hSize+ySize*hSize + xSize*ySize) = false;
gbm.nonZeroMask(xSize*ySize*hSize+ySize*hSize+xSize*ySize+1:xSize*ySize*hSize+ySize*hSize+xSize*ySize+xSize*hSize) = false;
gbm.nonZeroMask(xSize*ySize*hSize+1:xSize*ySize*hSize+ySize*hSize) =false;
% W_yh



% 
% 
% load('gbm_random_shifted_image_20140516T203626.mat','gbm');


% xData = xData(1:300000,:);
% yData = yData(1:300000,:);

gbm = gbmtrain(gbm, xData, yData);




timestampDataSet = timestamp;

timestamp = datestr(now,30);

save(['gbm_random_shifted_image_' timestamp '.mat'], 'gbm','timestamp','timestampDataSet');


end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [test_example_GBM_random_shifted_image.m] ======  
