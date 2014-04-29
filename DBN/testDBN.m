classdef testDBN < matlab.unittest.TestCase
    %TESTDBN Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Test)
        function testRBMMeanField(testCase)
            numberOfCases = 200;
            
            for iCase = 1:numberOfCases
                N = randi([1 200]);
                sizeH = randi([1 20]);
                sizeV = randi([1 20]);
                
                % related things
                % rbm.W is [sizeH x sizeV]
                % rbm.b is [sizeV x 1]
                % rbm.LV is [sizeV x sizeV]
                % rbm.lateralVisibleMask is [sizeV x sizeV]
                
                rbm = struct();
                rbm.W = randn(sizeH,sizeV);
                rbm.b = randn(sizeV,1);
                rbm.LV = randn(sizeV,sizeV);
                rbm.types{2} = 'binary';
                rbm.lateralVisibleMFIter = randi([1,10]);
                rbm.lateralVisibleMFDamp = rand();
                rbm.LV = (rbm.LV + rbm.LV')/2;
                LVdiagIndex = logical(eye(size(rbm.LV)));
                rbm.LV(LVdiagIndex) = 0;
                rbm.lateralVisibleMask = rand(sizeV,sizeV);
                rbm.lateralVisibleMask = rbm.lateralVisibleMask > 0.5;
                rbm.lateralVisibleMask = rbm.lateralVisibleMask & rbm.lateralVisibleMask';
                rbm.lateralVisibleMask(LVdiagIndex) = 0;
                sigma = 0.5 + rand();
                
                assert(isequal(rbm.lateralVisibleMask,rbm.lateralVisibleMask'));
                
                rbm.LV(~rbm.lateralVisibleMask) = 0;
                
                v = rand(N,sizeV);
                hSampled = rand(N,sizeH) > 0.5;
                [v1,nIter1] = rbm_meanfield_naive(v, hSampled, rbm,sigma);
                [v2,nIter2] = rbm_meanfield(v, hSampled, rbm,sigma);
                testCase.assertEqual(v1,v2,'AbsTol',1e-7);
                testCase.assertEqual(nIter1,nIter2);
                disp(iCase);
            end
        end
        
        
        function testRBMMeanFieldTrivial(testCase)
            numberOfCases = 200;
            
            for iCase = 1:numberOfCases
                N = randi([1 200]);
                sizeH = randi([1 20]);
                sizeV = randi([1 20]);
                
                % related things
                % rbm.W is [sizeH x sizeV]
                % rbm.b is [sizeV x 1]
                % rbm.LV is [sizeV x sizeV]
                % rbm.lateralVisibleMask is [sizeV x sizeV]
                
                rbm = struct();
                rbm.W = randn(sizeH,sizeV);
                rbm.b = randn(sizeV,1);
                rbm.LV = randn(sizeV,sizeV);
                rbm.types{2} = 'binary';
                rbm.lateralVisibleMFIter = randi([1,10]);
                rbm.lateralVisibleMFDamp = rand();
                rbm.LV = (rbm.LV + rbm.LV')/2;
                LVdiagIndex = logical(eye(size(rbm.LV)));
                rbm.LV(LVdiagIndex) = 0;
                rbm.lateralVisibleMask = false(sizeV,sizeV);
                sigma = 0.5 + rand();
                
                assert(isequal(rbm.lateralVisibleMask,rbm.lateralVisibleMask'));
                
                rbm.LV(~rbm.lateralVisibleMask) = 0;
                hSampled = rand(N,sizeH) > 0.5;
                
                v = sigm( (1/(sigma^2))*    (hSampled * rbm.W+repmat(rbm.b', N, 1))  );
                
                [v1,nIter1] = rbm_meanfield_naive(v, hSampled, rbm,sigma);
                [v2,nIter2] = rbm_meanfield(v, hSampled, rbm,sigma);
                testCase.assertEqual(v1,v2,'AbsTol',1e-7);
                testCase.assertEqual(v1,v,'AbsTol',1e-7);
                testCase.assertEqual(v2,v,'AbsTol',1e-7);
                testCase.assertEqual(nIter1,nIter2);
                testCase.assertEqual(nIter1,1);
                disp(iCase);
            end
        end
        
        function testRBMMeanFieldAndGibbs(testCase)
            numberOfCases = 200;
            
            for iCase = 1:numberOfCases
                N = randi([1 200]);
                sizeH = randi([1 20]);
                sizeV = randi([1 20]);
                
                % related things
                % rbm.W is [sizeH x sizeV]
                % rbm.b is [sizeV x 1]
                % rbm.LV is [sizeV x sizeV]
                % rbm.lateralVisibleMask is [sizeV x sizeV]
                
                rbm = struct();
                rbm.W = randn(sizeH,sizeV);
                rbm.b = randn(sizeV,1);
                rbm.LV = randn(sizeV,sizeV);
                rbm.types{2} = 'binary';
                rbm.lateralVisibleMFIter = randi([1,10]);
                rbm.lateralVisibleMFDamp = 0; % must be 0, because Gibbs sampling has no such things...
                rbm.LV = (rbm.LV + rbm.LV')/2;
                LVdiagIndex = logical(eye(size(rbm.LV)));
                rbm.LV(LVdiagIndex) = 0;
                rbm.lateralVisibleMask = rand(sizeV,sizeV);
                rbm.lateralVisibleMask = rbm.lateralVisibleMask > 0.5;
                rbm.lateralVisibleMask = rbm.lateralVisibleMask & rbm.lateralVisibleMask';
                rbm.lateralVisibleMask(LVdiagIndex) = 0;
                sigma = 0.5 + rand();
                
                assert(isequal(rbm.lateralVisibleMask,rbm.lateralVisibleMask'));
                
                rbm.LV(~rbm.lateralVisibleMask) = 0;
                
                v = rand(N,sizeV);
                hSampled = rand(N,sizeH) > 0.5;
                [v1,nIter1] = rbm_meanfield(v, hSampled, rbm,sigma);
                [v2,nIter2] = rbm_gibbs(v, hSampled, rbm,sigma,true);
                testCase.assertEqual(v1,v2,'AbsTol',1e-7);
                testCase.assertEqual(nIter1,nIter2);
                disp(iCase);
            end
        end
        
    end
    
end

