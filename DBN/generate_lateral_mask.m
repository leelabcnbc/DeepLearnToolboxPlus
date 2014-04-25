function lateralMask = generate_lateral_mask(sizeV, specifier)
% GENERATE_LATERAL_MASK ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 24-Apr-2014 20:52:53 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : generate_lateral_mask.m 

LVdiagIndex = logical(eye(sizeV));

if isequal(specifier,'all');
    lateralMask = true(sizeV,sizeV);
elseif isequal(specifier,'none');
    lateralMask = false(sizeV,sizeV);
else
    error('not implemented yet!');
end
   
lateralMask(LVdiagIndex) = false;
    

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [generate_lateral_mask.m] ======  
