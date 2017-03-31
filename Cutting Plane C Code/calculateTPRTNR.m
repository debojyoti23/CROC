function perf = calculateTPRTNR(yPredicted, yTest, perfMeasure)        
    % Odds Ratio
    oddsRatio = sum(yTest ~= 1)/sum(yTest == 1);

    % TPR:
    tpr = sum(yPredicted(yTest == 1, :) == 1) / sum(yTest == 1);
    % Check for 0's in denominator
    if(sum(yTest == 1) == 0)
        tpr = 1;
    end
    
    % TNR:
    tnr = sum(yPredicted(yTest ~= 1, :) ~= 1) / sum(yTest ~= 1);    
    % Check for 0's in denominator
    if(sum(yTest ~= 1) == 0)
        tnr = 1;
    end
    
    % Precision:
    prec = sum(yPredicted(yTest == 1, :) == 1) ./ sum(yPredicted == 1);    
    % Check for 0's in denominator
    prec(sum(yPredicted == 1) == 0) = 1;   
        
    if(strcmp(perfMeasure,'hmean'))
        % Compute hmean
        perf = 2 * tpr .* tnr ./ (tpr + tnr);
        % Check for 0's in denominator    
        perf(tpr+tnr == 0) = 0;
    elseif(strcmp(perfMeasure,'qmean'))
        % Compute qmean
        perf =  1 - sqrt(((1-tpr).^2 + (1-tnr).^2)/2);    
    elseif(strcmp(perfMeasure,'mintprtnr'))
        % Compute mintprtnr
        perf =  min([tpr; tnr]);
    elseif(strcmp(perfMeasure,'fmeasure'))
        % Compute fmeasure
        perf =  2*prec.*tpr./(prec+tpr);
        % Check for 0's in denominator    
        perf(prec+tpr == 0) = 0; 
    elseif(strcmp(perfMeasure,'jac'))
        % Compute jac
        if(isinf(oddsRatio))
            perf = 0;
        else
            perf = tpr./(1+oddsRatio*(1-tnr));
        end
    end
end