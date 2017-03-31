function doCuttingPlaneExperiment(datasetList, perfMeasure)
    splitOptions = load('splitOptions.mat');
    splitOptions = splitOptions.splitOptions;
    
    baseLearner = @CuttingPlane;    
    
    datasets = splitOptions.datasets;
    numSplits = splitOptions.numSplits;
    
    betaValues = 1;
    numBetaValues = splitOptions.numBetaValues;
    
	for datasetCounter =  datasetList
        X = load(['datasets/' datasets{datasetCounter} '.X.mat']);
        X = X.X;
        y = load(['datasets/' datasets{datasetCounter} '.y.mat']);
        y = y.y;
        splits = load(['datasets/' datasets{datasetCounter} '.splits.mat']);
        splits = splits.splits;
                
        PerfValues = cell(numBetaValues, numSplits);        
        CValues = zeros(numBetaValues, numSplits);
        wVectors = cell(numBetaValues, numSplits);
        tValues = cell(numBetaValues, numSplits);
        tValValues = zeros(numBetaValues, numSplits);
        
        for splitCounter = 1 : numSplits
            % Normalize data
            %%%%%%%%%%(Added check for isNormalized?)%%%%%%%%%%%%
            if(~splits.isNormalized)
                XThisSplit = bsxfun(@plus,full(X),-splits.means(:,splitCounter));
                XThisSplit = bsxfun(@rdivide,XThisSplit,splits.stds(:,splitCounter));
            else
                XThisSplit = X;                
            end
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            % Apply split permutation
            XThisSplit = XThisSplit(:,splits.IDs(:,splitCounter));
            yThisSplit = full(y(splits.IDs(:,splitCounter)));

            % Get the training and test splits
            XTrainingThisSplit = XThisSplit(:,1:splits.numTraining);
            yTrainingThisSplit = yThisSplit(1:splits.numTraining);			
            XTestThisSplit = XThisSplit(:,splits.numTraining+1:end);
            yTestThisSplit = yThisSplit(splits.numTraining+1:end);
            
            %%%%%%%%%%%
            %Added by Hari - 20/1/14; modified 21/1/14
%             numValidating = 25000;
%             if(splits.numTraining > 2*numValidating)
%                 splits.numValidating = numValidating;
%             end
            splits.numValidating = min(25000, floor(splits.numTraining/2));
            %%%            
            
            % Vary values of beta
            for betaCounter = 1:numBetaValues
                %%%%%%%%%%%%%%%%%
                options.epsilon = 0.1;%splitOptions.epsilon;
                %%%%%%%%%%%%%%%%%
                
                %%%%%
                options.perfMeasure = perfMeasure;
                %%%%%
                
                %%%%%%%%%%%% Added by Harikrishna %%%%%%%%%%
                options.ticSpacing = Inf;
                % set tick spacing to Inf to avoid storing per-iteration stats during validation
                
                fprintf('Dataset %s Split number %d Beta counter %d of %d Beta value %d\n',datasets{datasetCounter},splitCounter,betaCounter,numBetaValues,betaValues(betaCounter));
                
                % Get a good value of C by cross Validation
                     
                %%%%%%%%%%%% Modified by Harikrishna %%%%%%%%%%
                if(splitCounter == 1) % cross-validate only for first split
                    [tValValues(betaCounter,splitCounter) CThisSplit] = doCVForC(XThisSplit(:,1:2*splits.numValidating),yThisSplit(1:2*splits.numValidating),options,splits.numValidating,baseLearner);                    
                    CValues(betaCounter,splitCounter) = CThisSplit;
                else
                    CValues(betaCounter,splitCounter) = CValues(betaCounter,1); 
                    % pick C from first split
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                options.C = CValues(betaCounter,splitCounter);        
                
                %%%%%
                options.epsilon = 0.1;
                %%%%%
                                
                % Learn!
                options.ticSpacing = splitOptions.ticSpacing; % set tick spacing to preset value
                [tValues{betaCounter,splitCounter} wVectors{betaCounter,splitCounter} PerfValues{betaCounter,splitCounter}] = baseLearner(XTrainingThisSplit, yTrainingThisSplit, XTestThisSplit, yTestThisSplit, options);
                                
                perf = PerfValues{splitCounter};
                fprintf('(%s) Dataset %s Split %d: %f\n',perfMeasure,datasets{datasetCounter},splitCounter, perf(end));
                
                save(sprintf('results/CP/%s/PerfValues_%s.mat',perfMeasure,datasets{datasetCounter}),'PerfValues');
                save(sprintf('results/CP/%s/wVectors_%s.mat',perfMeasure,datasets{datasetCounter}),'wVectors');
                save(sprintf('results/CP/%s/tValues%s.mat',perfMeasure,datasets{datasetCounter}),'tValues');
                save(sprintf('results/CP/%s/tValValues%s.mat',perfMeasure,datasets{datasetCounter}),'tValValues');
                save(sprintf('results/CP/%s/CValues_%s.mat',perfMeasure,datasets{datasetCounter}),'CValues');
            end
        end
	end
end