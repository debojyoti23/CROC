function [timingStats, wBar, PerfStats] = CuttingPlane(XTrain, yTrain, XTest, yTest, options)

    C = options.C;%/options.lambda; % regularization parameter
    epsilon = options.epsilon; % cutting plane tolerance parameter
       
    verbosity = 0; % set verbosity level for cutting plane solver to 0 (off)
    if(strcmp(options.perfMeasure,'fmeasure'))
        cpdir = 'batch-solvers/cutting-plane-fmeasure/'; % CP solver directory
    elseif(strcmp(options.perfMeasure, 'qmean'))
        cpdir = 'batch-solvers/cutting-plane-qmean/'; % CP solver directory
    elseif(strcmp(options.perfMeasure, 'mintprtnr'))
        cpdir = 'batch-solvers/cutting-plane-min-tpr-tnr/'; % CP solver directory
    elseif(strcmp(options.perfMeasure, 'jac'))
        cpdir = 'batch-solvers/cutting-plane-jac/'; % CP solver directory
    end
    
    % Create temporary folder to store input/output files
    mkdir([cpdir 'paucsolvertemp']);
	
    % Write training data to a file in the temporary folder
    write_in_svm_light_format(yTrain, XTrain, [cpdir 'paucsolvertemp/data.txt']);
% 	fip = fopen([cpdir 'paucsolvertemp/data.txt'], 'w');
% 	for i = 1 : length(yTrain)
% 		fprintf(fip, '%d ', yTrain(i));
%         for j = 1 : size(XTrain, 1)
%             if(XTrain(j, i) ~= 0)
%                 fprintf(fip, '%d:%f ', j, XTrain(j, i));
%             end
%         end
%         fprintf(fip, '\n');
% 	end
% 	fclose(fip);
    
    % Files to dump timing statistics
    tstatsfile = [cpdir 'paucsolvertemp/timingstats.txt'];
    % mvcstatsfile = [cpdir 'paucsolvertemp/mvcstats.txt'];
    
    % Start the timer
    % tic    
	
%     C = C / 100; % in svmperf, both loss and psi multiplied by 100
    
    % System call to cutting plane solver:
    system(['./' cpdir 'cuttingplanesolver -c ' num2str(C) ' -l 1 -v ' num2str(verbosity)...
                    ' -w 3 -e ' num2str(epsilon) ' --m ' tstatsfile ' --b 1 ' ...
                   ' ' cpdir 'paucsolvertemp/data.txt ' cpdir 'paucsolvertemp/wout.txt']);               
        %	  Input parameters:	        
        %     	-c  regularization parameter
        %       -l 1 fmeasure
        %       --p beta (we have K = beta x fraction of positives)
        %       -e epsilon (tolerance parameter)
        %       -w 3 1 slack algo dual (not default in svmperf)
        %     	-v verbosity (0/1: on or off; default = 1)
        %       --m tstatsfile (was --t for pAUC)
        %       --b 1 (bias term in model)
        %
        %     Input file:
        %       data.txt (label, feature vector)
        %	  Output file:
        %	  	wout.txt (d x 1 model learned)
    
    % How much time did I take to compute wBar
    % time = toc;
    
    % Read timing stats
    perIterationStats = dlmread(tstatsfile);
        
    % Read wBar from output file; do not read from output file as this
    % holds a different format
 	wBar = perIterationStats(end, 2:1+size(XTrain,1));
    
    % Remove temporary folder
    rmdir([cpdir 'paucsolvertemp'], 's');  % remove folder with subfolders / files
    	
    % Check if we are in validation phase - in which case just return the
    % total run time and last pAUC
    if(options.ticSpacing == Inf)
        perIterationStats = perIterationStats(end, :);
    end
    
    % pAUC per iteration    
    numPoints = size(XTest, 2);
    yPredicted = XTest'*perIterationStats(:, 2:1+size(XTest,1))' + repmat(perIterationStats(:, end)', [numPoints, 1]); % ( + bias)
    % first column is the run time; 2:1+numfeatures is wt vector (two more
    % entries ignored) --- add bias stored in last component
    
%     PerfStats = calculateFmeasure(sign(yPredicted),yTest);
    PerfStats = calculateTPRTNR(sign(yPredicted),yTest,options.perfMeasure); %% sign -- predictions must be in {-1,1}
    timingStats = perIterationStats(:, 1);
end

%%% Write data to file in svmlight format %%%
%%% Adapted from snipplr.com/view/22106/svmlightwrapper.m/ %%%
function write_in_svm_light_format(y, X, fname)
    M = [y, X']; % append labels and features (labels first)
    
    dlmwrite([fname '.temp'], M,'delimiter',' '); % wrie temp file with blank space as delimiter
        
    % write new file from temp file in svm light format
    system(['awk -F" " ''{printf $1" "; for (i=2;i<=NF;i++) {if($i!=0) printf i-1":"$i " "}; print ""}'' ' fname '.temp > ' fname]);
end