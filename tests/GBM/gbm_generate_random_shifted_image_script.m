clear all;
close all;

rng(0,'twister');

N = 1000000;
sideSize = 10;
probOn = 0.1;
debug = false;

[xData, yData] = gbm_generate_random_shifted_image(N,sideSize,probOn,debug);

timestamp = datestr(now,30);

save('gbm_image_data.mat','xData','yData','N','sideSize','probOn','debug','timestamp');


% N = 500000;
% sideSize = 10;
% probOn = 0.5;
% debug = true;
% 
% [xData, yData] = gbm_generate_random_shifted_image(N,sideSize,probOn,debug);
% 
% timestamp = datestr(now,30);
% 
% save('gbm_image_data_prob05.mat','xData','yData','N','sideSize','probOn','debug','timestamp');