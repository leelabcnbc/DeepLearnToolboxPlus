classdef TestPreprocessing < matlab.unittest.TestCase
    %TESTPREPROCESSING Test preprocessing functions.
    %   contrast_normalization, contrast_normalization_3D, whiten_PCA,
    %   whiten_ZCA, and whiten_olsh.
    
    % 2014-03-14
    % Yimeng Zhang
    % Computer Science Department, Carnegie Mellon University
    % zym1010@gmail.com
    
    properties
    end
    
    methods (Test)
        function testWhitenZCA(testCase)
            import preprocessing.*
            numberOfCases = 50;
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = rand(1);
                [Xnew, unwhitenMatrix, M, ~] = whiten_ZCA(X, epsilon);
                % in this case, I don't know a closed form solution of the
                % resultant covariance matrix, since there's epsilon.
                testCase.assertEqual(bsxfun(@plus, Xnew*unwhitenMatrix, M),X,'AbsTol',1e-6);
            end
            % test the returned covariance, with epsilon = 0, var should be
            % 1
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = 0;
                [Xnew, unwhitenMatrix, M, D] = whiten_ZCA(X, epsilon);
                covNew = cov(Xnew);
                testCase.assertEqual(diag(covNew), ones(size(D,1),1)  , 'AbsTol',1e-6);
                testCase.assertEqual(bsxfun(@plus, Xnew*unwhitenMatrix, M),X,'AbsTol',1e-6);
            end
        end
        
        function testWhitenPCA(testCase)
            import preprocessing.*
            numberOfCases = 50;
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = rand(1);
                [Xnew, unwhitenMatrix, M, D] = whiten_PCA(X, epsilon);
                covNew = cov(Xnew);
                testCase.assertEqual(diag(covNew), diag(D)./(diag(D) + epsilon), 'AbsTol',1e-6);
                testCase.assertEqual(bsxfun(@plus, Xnew*unwhitenMatrix, M),X,'AbsTol',1e-6);
            end
            
            % test the returned covariance, with epsilon = 0, var should be
            % 1
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = 0;
                [Xnew, unwhitenMatrix, M, D] = whiten_PCA(X, epsilon);
                covNew = cov(Xnew);
                testCase.assertEqual(diag(covNew), ones(size(D,1),1), 'AbsTol',1e-6);
                testCase.assertEqual(bsxfun(@plus, Xnew*unwhitenMatrix, M),X,'AbsTol',1e-6);
            end
        end
        
        function testWhitenOlsh(testCase)
            import preprocessing.*
            numberOfCases = 50;
            for iCase = 1:numberOfCases
                X = rand(100,100,100);
                avg_var = rand(1);
                [Xnew] = whiten_olsh(X, avg_var);
                Xnew = reshape(Xnew,10000,100);
                testCase.assertEqual(abs(avg_var),mean(var(Xnew)),'AbsTol',1e-6);
            end
        end
        
        function testContrastNormalization(testCase)
            import preprocessing.*
            numberOfCases = 50;
            
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = rand(1);
                Xnew = contrast_normalization(X,epsilon);
                varX = var(X,[],2);
                testCase.assertEqual(mean(Xnew,2), zeros(10000,1), 'AbsTol',1e-6);
                testCase.assertEqual(var(Xnew,[],2), varX./(varX + epsilon), 'AbsTol',1e-6);
            end
            
            % trivial case when epsilon = 0, var should be 1.
            for iCase = 1:numberOfCases
                X = rand(10000,100);
                epsilon = 0;
                Xnew = contrast_normalization(X,epsilon);
                testCase.assertEqual(mean(Xnew,2), zeros(10000,1), 'AbsTol',1e-6);
                testCase.assertEqual(var(Xnew,[],2), ones(10000,1), 'AbsTol',1e-6);
            end
            
        end
        
        function testContrastNormalization3D(testCase)
            import preprocessing.*
            numberOfCases = 50;
            
            for iCase = 1:numberOfCases
                X = rand(100,100,100);
                epsilon = rand(1);
                Xnew = contrast_normalization_3D(X,epsilon);
                Xnew = reshape(Xnew,10000,100)';
                varX = var(reshape(X,10000,100)',[],2);
                testCase.assertEqual(mean(Xnew,2), zeros(100,1), 'AbsTol',1e-6);
                testCase.assertEqual(var(Xnew,[],2), varX./(varX + epsilon), 'AbsTol',1e-6);
            end
            
            % trivial case when epsilon = 0, var should be 1.
            for iCase = 1:numberOfCases
                X = rand(100,100,100);
                epsilon = 0;
                Xnew = contrast_normalization_3D(X,epsilon);
                Xnew = reshape(Xnew,10000,100)';
                testCase.assertEqual(mean(Xnew,2), zeros(100,1), 'AbsTol',1e-6);
                testCase.assertEqual(var(Xnew,[],2), ones(100,1), 'AbsTol',1e-6);
            end
        end
        
    end
    
end

