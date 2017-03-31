function w = svmpauc_tight(X, y, alpha_val, beta_val, C, eps, verbosity)
        % SVMpAUC-tight - optimize pAUC in the FPR interval [alpha_val, beta_val]
        %               - learns a linear score function parameterized by 'w'
        % INPUT:
        %  <X, y> - training set <features, label>,
        %        where X is of size 'num_of_training_examples x dimension'
        %        and y is of size 'num_of_training_examples x 1'
        %  C > 0 - regularization paramter
        %  eps > 0 - tolerance parameter 'epsilon' (in the cutting plane method)
        %            (default: 0.0001)
        %  verbosity - output display (0/1 - yes/no) 
        %              (default: 1)
        % OUTPUT:
        %  w - model learnt; w is of size 'dimension x 1'

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % The structure of this code has been adapted from Andrea Vedaldi's MATLAB API for structural SVMs: 
            % http://www.robots.ox.ac.uk/~vedaldi/code/svm-struct-matlab.html
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
        % Global variables visible to other functions  
        global m n neg_alpha_ceil neg_alpha_ceil_next salpha alpha neg_beta_floor neg_beta_ceil sbeta beta verbosity_level;
        
        % Added by DD
        global gamma;

        %--Error checking--
        % Check the number of inputs
        if nargin < 5
          error('myApp:argChk', 'Wrong number of input arguments: Please give at least 5 input arguments')
        elseif nargin == 5
          eps = 0.0001;
          verbosity = 1;
        elseif nargin == 6
          verbosity = 1;
        end  

        % Check if 'X' and 'y' have same number of points
        if(size(X, 1) ~= size(y, 1))
          error('Mismatch between sizes of X and y');
        end   

        % Check if 'y' is a column vector
        if(size(y, 2) ~= 1)
          error('y must be a column vector');
        end   

        % Check if 'y' is binary && 'y' has only -1 and +1
        if(size(unique(y), 1) > 2 || min(y) ~= - 1 || max(y) ~= 1)
          error('y (set of training labels) can contain only two values: -1 or 1; Also, the training set should contain both positive and negative examples');
        end   

        % Check if 0 <= alpha_val <= 1
        if(alpha_val < 0 || alpha_val > 1)
          error('beta_val should be a value in [0, 1]');
        end   

        % Check if 0 <= beta_val <= 1
        if(beta_val < 0 || beta_val > 1)
          error('beta_val should be a value in [0, 1]');
        end   

        % Check if alpha_val < beta_val
        if(alpha_val >= beta_val)
          error('alpha_val should be lesser than beta_val');
        end

        % Check if C > 0
        if(C <= 0)
          error('Regularization parameter C must be greater than 0');
        end  

        % Check if epsilon > 0
        if(eps <= 0)
          error('Error tolerance parameter eps must be greater than 0');
        end 

        if(verbosity ~= 0 && verbosity ~= 1)
          error('Verbosity level must be 0 or 1');
        end
        %------------------  

        d = size(X,2); % number of features

        pos = find(y==1); % indices of positive objects
        neg = find(y==-1); % indices of positive objects

        m = size(pos,1); % number of positive objects
        n = size(neg,1); % number of negative objects

        %----Another error check---%
        % Check if alpha_val & beta_val are sufficiently separated
        if(n*(beta_val - alpha_val) < 1)
          error('alpha_val and beta_val are too close!\n We require num-of-negatives * (beta_val - alpha_val) >= 1');
        end
        %--------------------------%

        alpha = alpha_val;
        neg_alpha_ceil = ceil(n * alpha); %number of negatives of interest ceil(n x alpha)
        neg_alpha_ceil_next =  neg_alpha_ceil + 1; %number of negatives of interest ceil(n x alpha) + 1
        salpha = neg_alpha_ceil - n*alpha; %additional factor for left out portion of the curve on the left

        beta = beta_val;
        neg_beta_floor = floor(n * beta); %number of negatives of interest floor(n x beta)
        neg_beta_ceil =  ceil(n * beta); %number of negatives of interest ceil(n x beta)
        sbeta = n*beta - neg_beta_floor; %additional factor for left out portion of the curve on the right

        verbosity_level = verbosity; % output on or off
        
        % Added by DD
        mag_factor = 20;
        magfn = magExp(mag_factor,n);
        gamma = magfn(2:end) - magfn(1:end-1);

        % ----------------------------------------------------------------------
        %                                               Run Cutting-Plane Solver
        % ----------------------------------------------------------------------

        %ideal structured output: 'ordering' with positives above negatives
        %                        ('m' positives above 'neg_beta_ceil' negatives)
        ideal_labels = [ones(m, 1); -ones(neg_beta_ceil, 1)];  
        ideal_output = convert2output(ideal_labels, 1:m, 1:neg_beta_ceil);
                        % convert to ideal ordering to output representation      
        % Data set details
        dataset.features = [X(pos,:); X(neg,:)];  % Input data - arrange positive examples first (for convenience)
        dataset.label = ideal_output; % Output (ideal ordering)
        dataset.dimension = d; % Number of features

        %Learn model 'w'
        w = cutting_plane(dataset, C, eps);
        % StructSVM-style optimization problem with margin rescaling (kernel K)
        % 1-slack algorithm (dual), dataset (input features & label),
        % regularization constant C, and tolerance parameter - eps

end

% ------------------------------------------------------------------
%                                               StructSVM callbacks
% ------------------------------------------------------------------

function ret = magExp(alpha,n)
% function to compute exponential magnification over an array [1,2,...,n]
% f(x)=(1-exp(-alpha*x))/(1-exp(-alpha))
% returns an array
xlist = 0:n;
xlist = xlist/n;
ret = (1-exp(-alpha*xlist))/(1-exp(-alpha));
end

function delta = loss_pAUC(a,neg_subset)
        % LOSS FUNCTION: delta(aideal, a) = pAUC(aideal) - pAUC(a) = 1 - pAUC(a)
        % Input:
        %   a - given ordering (the ideal ordering 'aideal' need not be explicitly given as input)
        % Output:
        %   delta - value of 'delta(aideal, a)'    
    
        % Global variables visible to other functions
        global m n neg_alpha_ceil neg_alpha_ceil_next salpha alpha neg_beta_ceil neg_beta_floor beta sbeta;
        global gamma;   % added by DD

        aminus = a(m+1: m+neg_beta_ceil); % a = [aplus; aminus]  

        % 1 - pAUC(0, beta), where
        %     pAUC(0, beta) = 
        %             (fraction of area between neg_alpha_ceil and neg_alpha_ceil_next) 
        %                         + (AUC between neg_alpha_ceil_next and neg_beta_floor) + (fraction of area between neg_beta_floor + neg_beta_ceil)
        if(neg_alpha_ceil == 0)
            delta = sum((m - aminus(1:neg_beta_floor)).*gamma(neg_subset)'); % changed by DD
            if(neg_beta_floor < neg_beta_ceil)
                delta = delta + gamma(neg_beta_ceil)*(sbeta*m - aminus(neg_beta_ceil)); % changed by DD
            end
        else
            delta = salpha*(m - aminus(neg_alpha_ceil))*gamma(neg_alpha_ceil) + sum((m - aminus(neg_alpha_ceil_next:neg_beta_floor)).*gamma(neg_subset)'); %changed by DD
            if(neg_beta_floor < neg_beta_ceil)
                delta = delta +  gamma(neg_beta_ceil)*(sbeta*m - aminus(neg_beta_ceil)); % changed by DD
            end
        end

        % added by DD
        c_alpha_beta = sum(gamma(neg_alpha_ceil_next:neg_beta_ceil));
        
        delta = delta / m / c_alpha_beta; % Normalize % TO DO % changed by DD
end

function psi = joint_feature_map(x, a, neg_subset)
        % JOINT FEATURE MAP: psi(input, output) -> R^d
        % Input:
        %   x - input features
        %   <a, neg_subset> - output: <ordering, subset of negatives>
        % Output:
        %   psi - joint-feature map in R^d

        % Global variables visible to other functions         
        global m n alpha beta neg_beta_ceil;
        global neg_alpha_ceil_next gamma; % added by DD

        if(isempty(neg_subset))
          neg_subset = 1:neg_beta_ceil;
        end

        Xplus = x(1:m, :); % Positive instances
        Xminus = x(m+neg_subset(1:neg_beta_ceil), :); % Negative instances

        aplus = a(1: m); %  a = [aplus; aminus]
        aminus = a(m+1:m+neg_beta_ceil); 

        % Joint feature representation: psi = \sum_{i = 1}^m aplus(i)*Xplus(i) -
        %                                      \sum_{j = 1}^n aminus(j)*Xminus(j)
        psi = aplus'*Xplus - aminus'*Xminus;   
        
        % added by DD
        c_alpha_beta = sum(gamma(neg_alpha_ceil_next:neg_beta_ceil));

        psi = psi' / m / c_alpha_beta; % Normalize % changed by DD
end

function [ahat, neg_subset] = find_most_violated_constraint(w, x)
        % FIND MOST VIOLATED CONSTRAINT
        %   margin rescaling: argmax_{a, negsub} loss(a, negsub) + <psi(x,a, negsub), w>
        % Input: 
        %     w - model
        %     x - input features
        % Output:
        %     <ahat, neg_subset> - most-violated constraint: <ordering, subset of negatives>        
    
        % Global variables visible to other functions
        global neg_alpha_ceil;
        
        if(neg_alpha_ceil == 0)
            [ahat, neg_subset] = find_most_violated_constraint_special(w, x);
            %Call special case for [0, beta]
        else
            [ahat, neg_subset] = find_most_violated_constraint_general(w, x);
            %Call general case for [alpha, beta]
        end
end

function [ahat, neg_ind] = find_most_violated_constraint_special(w, x)
        % FIND MOST VIOLATED CONSTRAINT for the special case of [0, beta] FPR interval
        %   margin rescaling: argmax_{a, negsub} loss(a, negsub) + <psi(x,a, negsub), w>
        % Input: 
        %     w - model
        %     x - input features
        % Output:
        %     <ahat, neg_subset> - most-violated constraint: <ordering, subset of negatives>
 
        % Global variables visible to other functions  
        global m n neg_beta_ceil;
        global gamma;

        wx = x*w; %Score function w.x

        [~, pos_ind] = sort(wx(1:m),'descend'); % Sort positives based on w.x
        
        % Subtracted by DD
%         [~, neg_ind] = sort(wx(m+1:m+n),'descend'); % Sort negatives based on w.x
%         neg_ind = neg_ind(1:neg_beta_ceil);

        % Score function for positive instances: fplus(i) = w^T x^+_i
        fplus = wx(1:m);

        % Score function for negative instances: 
        %         fminus(j) = w^T x^-_j + 1        if (j)_a \in {1, ..., j_\beta}
        %                   = w^T x^-_j + s_\beta  if (j)_a = j_\beta + 1
        %                   = w^T x^-_j             o/w
        
%         % Added by DD
        [~, neg_ind] = sort(wx(m+1:m+n)+gamma','descend'); % Sort negatives based on w.x
        neg_ind = neg_ind(1:neg_beta_ceil);
        fminus = wx(m+neg_ind)+gamma(neg_ind)';
        
        % Subtracted by DD
%         fminus = wx(m+neg_ind);
%         fminus = fminus + 1;

        %Sort instances according to f = [fminus; fplus]; 
        %    (negatives first, to ensure ties are penalized)
        [~, fsort_indices] = sort([fminus; fplus], 'descend');

        %Labels of sorted instances
        labels = [-ones(neg_beta_ceil, 1); ones(m, 1)]; 
        slabels = labels(fsort_indices);

        %Convert to output representation
        ahat = convert2output(slabels, pos_ind, 1:neg_beta_ceil);
end

function [ahat, neg_ind] = find_most_violated_constraint_general(w, x)
        % FIND MOST VIOLATED CONSTRAINT for the general case of [alpha, beta] FPR interval
        %   margin rescaling: argmax_{a, negsub} loss(a, negsub) + <psi(x,a, negsub), w>
        % Input: 
        %     w - model
        %     x - input features
        % Output:
        %     <ahat, neg_subset> - most-violated constraint: <ordering, subset of negatives>

        % Global variables visible to other functions
        global m n neg_alpha_ceil neg_alpha_ceil_next neg_beta_ceil;

        wx = x*w; %Score function w.x

        [wpos, pos_ind] = sort(wx(1:m),'descend'); % Sort positives based on w.x
        [wneg, neg_ind] = sort(wx(m+1:m+n),'descend'); % Sort negatives based on w.x
        neg_ind = neg_ind(1:neg_beta_ceil);
        wneg = wneg(1:neg_beta_ceil);

        % Score function for positive instances: fplus(i) = w^T x^+_i
        fplus = wx(1:m);

        %%%%%%%%%Optimize over r_i \in {1, ..., j_\alpha - 1}%%%%%%%%%

        if(neg_alpha_ceil > 0)
            % Score function for negative instances in range [(1)_a, ..., (j_\alpha - 1)_a];
            %         fminus(j) = w^T x^-_j
            fminus1 = wx(m+1: m+n);
            fminus1 = fminus1(neg_ind(1:neg_alpha_ceil-1));
            
            %Sort instances according to f = [fminus; fplus];
            %   (negatives first, to ensure ties are penalized)
            [~, fsort_indices1] = sort([fminus1; fplus], 'descend');
            
            %Labels corresponding to ordering zbar^1
            labels1 = [-ones(neg_alpha_ceil - 1, 1); ones(m, 1)];
            slabels1 = [labels1(fsort_indices1); -ones(neg_beta_ceil - neg_alpha_ceil + 1, 1)];
            
            %Compute objective values
            Q1 = compute_objective(slabels1, wpos, wneg);
        else
            Q1 = -ones(m, 1); %Make objective values -1 (low value) if j_\alpha = 0
        end
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%Optimize over r_i \in {j_\alpha}%%%%%%%%%
        %Labels corresponding to ordering zbar^2
        slabels2 = [-ones(neg_alpha_ceil, 1); ones(m, 1); -ones(neg_beta_ceil - neg_alpha_ceil, 1)];

        %Compute objective values
        Q2 = compute_objective(slabels2, wpos, wneg);  
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%%%%%Optimize over r_i \in {j_\beta + 2, ..., n}%%%%%%%%%%

        % Score function for negative instances in range [(j_\alpha + 2)_a, ..., (n)_a];
        %         fminus(j) = w^T x^-_j + 1        if (j)_a \in {j_\alpha + 2, ..., j_\beta}
        %                   = w^T x^-_j + s_\beta  if (j)_a = j_\beta + 1
        %                   = w^T x^-_j            if (j)_a \in {j_\beta + 2, ..., n} 
        fminus3 = wx(m+1: m+n);  
        fminus3 = fminus3(neg_ind(neg_alpha_ceil_next+1:neg_beta_ceil)); 
        fminus3 = fminus3 + 1;   
        %Sort instances according to f = [fminus3; fplus];
        %   (negatives first, to ensure ties are penalized)
        [~, fsort_indices3] = sort([fminus3; fplus], 'descend');

        %Labels corresponding to ordering zbar^3
        labels3 = [-ones(neg_beta_ceil - neg_alpha_ceil_next, 1); ones(m, 1)];
        slabels3 = [-ones(neg_alpha_ceil_next, 1); labels3(fsort_indices3)];

        %Compute objective values
        Q3 = compute_objective(slabels3, wpos, wneg);

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%Construct optimal ordering from Q1, Q2 and Q3%%%

        % Divide set of positive into three part:
        %     (based on whether they have maximum value of Q1, Q2, and Q3 respectively)
        [~, max_ind] = max([Q1, Q2, Q3], [], 2);  

        % Calculate number of positives lying in each of the three parts
        num_pos_in_part1 = sum(max_ind == 1);
        num_pos_in_part2 = sum(max_ind == 2);
        num_pos_in_part3 = sum(max_ind == 3);

        % Construct optimal ordering: (each positive example is placed in that 
        % part of the list for which it has maximum objective value)
        [~, fsort_indices3] = sort([fminus3; wpos(num_pos_in_part1+num_pos_in_part2+1 : m)], 'descend');
        labels3 = [-ones(neg_beta_ceil - neg_alpha_ceil_next, 1); ones(num_pos_in_part3, 1)];

        if(neg_alpha_ceil > 0)
            [~, fsort_indices1] = sort([fminus1; wpos(1:num_pos_in_part1)], 'descend');
            labels1 = [-ones(neg_alpha_ceil - 1, 1); ones(num_pos_in_part1, 1)];
            
            slabels = [labels1(fsort_indices1); -1; ones(num_pos_in_part2, 1); -1; labels3(fsort_indices3)];
        else
            slabels = [ones(num_pos_in_part2, 1); -1; labels3(fsort_indices3)];
        end

        %Convert to optimal 'ordering' to output representation
        ahat = convert2output(slabels, pos_ind, 1:neg_beta_ceil);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      
end

function Q = compute_objective(labels, wpos, wneg)
        % COMPUTE OBJECTIVE VALUE FOR GIVEN OUTPUT:
        %     i.e., value of loss(a, negsub) + <psi(x,a, negsub), w> for given output 
        %         (the ordering and subset of negatives are implicitly specified in the input)
        % Input:
        %     labels - labels (1/-1) of examples in given ordering
        %     wpos - scores of positive examples in given ordering
        %     wneg - scores of negative examples in given ordering
        % Output:
        %     Q - objective value of given output
    
        % Global variables visible to other functions
        global m neg_alpha_ceil neg_alpha_ceil_next salpha neg_beta_floor neg_beta_ceil sbeta;

        i = 0;
        j = 0;

        neg1 = 0; % No. of negatives encountered till now
        neg2 = 0; % No. of negatives encountered till now in the range [(j_alpha + 1): (j_beta)]
        flag_alpha = 0; % Has (j_alpha)^th negative been encountered?
        flag_beta = 0; % Has (j_beta + 1)^th negative?
        s = 0; % Sum of scores of negatives encountered till now    
        %%%%%%%%%%%%%%%     
        b = zeros(m, 1); % b[i] = no. of negatives in the range [(j_alpha + 1): (j_beta)] above the (i)^th positive
        c = zeros(m, 1); % c[i] = no. of negatives above the (i)^th positive
        d = zeros(m, 1); % d[i] = sum of scores of negatives above the (i)^th positive
        rho_alpha = zeros(m, 1); % rho_alpha(i) = is (j_alpha)^th negative above (i)^th positive?
        rho_beta = zeros(m, 1); % rho_beta(i) = is (j_beta + 1)^th negative above (i)^th positive?

        % Scan once through given ordered list and compute certain statistics
        % based on labels encountered
        for l = labels'
            if(l == 1) % Positive label
                i = i + 1;
                b((i)) = neg1;
                c((i)) = neg2;
                d((i)) = s;
                rho_alpha((i)) = flag_alpha;            
                rho_beta((i)) = flag_beta;
            else % Negative label
                j = j + 1;
                if(j < neg_alpha_ceil)
                    neg2 = neg2 + 1;    
                    s = s + wneg(j); 
                elseif(neg_alpha_ceil > 0 && j == neg_alpha_ceil)
                    flag_alpha = 1;
                    neg2 = neg2 + 1;
                    s = s + wneg(j); 
                elseif(sum(j == neg_alpha_ceil_next:neg_beta_floor))
                    neg1 = neg1 + 1;
                    neg2 = neg2 + 1;
                    s = s + wneg(j); 
                elseif(j == neg_beta_ceil)
                    flag_beta = 1;
                    neg2 = neg2 + sbeta;
                    s = s + sbeta*wneg(j); 
                end             
            end
        end

        %Compute objective
        Q = (salpha*rho_alpha + b + sbeta*rho_beta) - c.*wpos + d;
end

function a = convert2output(labels, pos_ind, neg_ind)    
        % CONVERT GIVEN ORDERING TO THE (num-of-pos + num-of-neg) SIZED REPRESENTATION
        % Input:
        %    labels - labels of examples in given ordering
        %    pos_ind - indices of positive examples in given ordering
        %    neg_ind - indices of negative examples in given ordering
        % Output:
        %    a - output representaion of size (num-of-pos + num-of-neg)

        % Global variables visible to other functions
        global m neg_beta_floor neg_beta_ceil sbeta;

        % Maintain number of positive and negative examples seen so far
        i = 0; % Number of positive and negative examples seen so far
        j = 0; % Number of positive and negative examples seen so far

        % a = [aplus; aminus]
        aplus = zeros(m, 1);
        aminus = zeros(neg_beta_ceil, 1);

        pos = 0; % No. of positives encountered
        neg = 0; % No. of negatives encountered

        % max. number of negatives ranked below a positive in output 
        % represenation is (neg_beta_floor + sbeta)
        maxnegcount = neg_beta_floor + sbeta;

        % Scan once through given ordered list and convert to output
        % representation based on labels encountered
        for l = labels'
            if(l == 1) % Positive label
                i = i + 1;
                pos = pos + 1;
                aplus(pos_ind(i)) = maxnegcount - neg;
            else % Negative label
                j = j + 1;
                if(j>neg_beta_floor)
                    neg = neg + sbeta;
                    aminus(neg_ind(j)) = sbeta*pos;    % doubt
                else
                    neg = neg + 1;
                    aminus(neg_ind(j)) = pos;
                end
            end
        end

        % output representation for given ordering:
        a = [aplus; aminus];
end

function w = cutting_plane(dataset, C, epsilon)
        % MAIN CUTTING-PLANE OPTIMIZATION PROCEDURE
        % Input:
        %    dataset - input features & labels
        %    C - regularization parameter
        %    epsilon - tolerance paramter (> 0) that determines convergence
        % Output:
        %    a - output representaion of size (num-of-pos + num-of-neg)   

        % Global variable visible to other functions
        global verbosity_level;

        % Joint-feature maps for 'current set of constraints'
        psi_all_outputs = {}; 
        % joint-feature-map for all outputs <ordering-matrix, subset> in 'current set of constraints'
        psi_ideal_output = {}; 
        % joint-feature-map for outputs <ideal-ordering-matrix, subset>  
        %      for all subsets of negatives in 'current set of constraints'

        ccnt = 0; % Number of constraints generated so far

        d = dataset.dimension; %data dimension

        % Set options for quadprog -- MATLAB QP solver
        options = optimset('Display','off','Algorithm','active-set','MaxIter',1000);

        % Initialization:
        w = zeros(d, 1); % Model 'w'
        alpha = []; % Array of Lagrange multipliers

        xi = 0; % Slack variable

        x = dataset.features;  % Training examples: features
        y = dataset.label; % Training examples: labels {-1, 1}

        % Initialization: Kernel matrices (linear) computed on joint-feature maps
        Ker = []; % Kernel between every pair of output in 'current set of constraints'
        Kyy = []; % Kernel between pairs: <yideal, subset1> and <yideal, subset2> for every pair of (subset1, subset2) in neg_subset
        Kyyhat = [];  % Kernel between pairs: <yideal, subset1> and <y, subset2> for every such combination in 'current set of constraints'

        % Initialization: vector of losses for outputs in 'current set of constraints'
        loss = [];

        while(1)
            % Find most violated constraint for current 'w':
            [most_violated_ordering, most_violated_subset] = find_most_violated_constraint(w, x);    
                % where most_violated_ordering is an ordering-matrix
                % neg-subset is a 'subset of negatives' corresponding to
                % the FPR interval [alpha, beta]

            % Compute joint-feature map:
            psiyyhat = joint_feature_map(x, y, most_violated_subset); 
                % feature-map for output: <yideal, neg-subset>
            psiyhat = joint_feature_map(x, most_violated_ordering, most_violated_subset);
                % feature-map for output: <most_violated_ordering, subset of negatives>

            % Compute dot-product (linear kernel) between joint-feature maps of
            % outputs <yideal, subset> and <yideal, most_violated_subset> for
            % all 'subset' in neg_subset
            Knew = zeros(ccnt+1, ccnt+1);
            Knew(1:ccnt,1:ccnt) = Ker(1:ccnt,1:ccnt);
            for j = 1:ccnt
                Knew(ccnt+1,j) = psi_all_outputs{j}'*psiyhat;
                Knew(j,ccnt+1) = Knew(ccnt+1,j);
            end
            Knew(ccnt + 1, ccnt + 1) = psiyhat'*psiyhat;

            % Compute dot-product (linear kernel) between joint-feature maps of
            % (i)   outputs <y, subset> and <yideal, most_violated_subset> for
            %       all <y, subset> in 'current set of constraints'
            % (ii)  outputs <yideal, subset> and <most_violated_ordering, most_violated_subset> for
            %       all 'subset' in neg_subset

            Kyyhatnew = zeros(ccnt+1, ccnt+1);
            Kyyhatnew(1:ccnt,1:ccnt) = Kyyhat(1:ccnt,1:ccnt);
            for j = 1:ccnt
                Kyyhatnew(ccnt+1,j) = psiyyhat'*psi_all_outputs{j}; 
                Kyyhatnew(j,ccnt+1) = psi_ideal_output{j}'*psiyhat;
            end
            Kyyhatnew(ccnt + 1, ccnt + 1) = psiyyhat'*psiyhat;

            % Compute dot-product (linear kernel) between joint-feature maps of
            % outputs <yideal, subset> and <yideal, most_violated_subset> for
            % all 'subset' in neg_subset
            Kyynew = zeros(ccnt+1, ccnt+1);
            Kyynew(1:ccnt,1:ccnt) = Kyy(1:ccnt,1:ccnt);
            for j = 1:ccnt
                Kyynew(ccnt+1,j) = psi_ideal_output{j}'*psiyyhat;
                Kyynew(j,ccnt+1) = Kyynew(ccnt+1,j);
            end
            Kyynew(ccnt + 1, ccnt + 1) = psiyyhat'*psiyyhat;

            % Compute loss for most-violated constraint
            loss(ccnt+1,:) = loss_pAUC(most_violated_ordering,most_violated_subset);    % function sign changed by DD

            % Compute new slack variable value (after most violated constraint added)
            xinew = loss(ccnt+1) - wdotpsi(Knew(1:ccnt, ccnt + 1) - Kyyhatnew(ccnt + 1, 1:ccnt)' - Kyyhatnew(1:ccnt, ccnt + 1) + Kyynew(1:ccnt, ccnt+1), alpha);

            % *Sanity check: Does the new constraint yield higher slack value
            % than current set of constraints? (alibiet precision issues)
            if(xinew < xi - 1e-10)
                fprintf('ERROR: MVC wrong!!! (Difference between old slack and new slack values = %e)\n', xi - xinew);
                break;
            end

            % Convergence test: 
            % difference between old and new slack values <= epsilon
            if(xinew <= xi + epsilon)
                break; 
            end

            ccnt = ccnt + 1; % Increment number of constraints generated

            % Update set of  joint-feature maps with
            % those for the newly generated constraint
            psi_all_outputs{ccnt} = psiyhat; 
            psi_ideal_output{ccnt} = psiyyhat;

            % Update old matrices with the new ones
            Ker = Knew;   
            Kyy = Kyynew; 
            Kyyhat = Kyyhatnew; 

            % Compute ccnt x ccnt matrix 'Kquad' for QP
            for j = 1:ccnt
                Kquad(ccnt, j) = Kyy(ccnt, j) - Kyyhat(ccnt, j) - Kyyhat(j, ccnt) + Ker(ccnt, j);
                Kquad(j, ccnt) = Kquad(ccnt, j);
            end

            % Solve QP using MATLAB's quadprog        
            [alpha, fval] = quadprog(Kquad, -loss, ones(1, ccnt), C, [], [], zeros(ccnt,1), [], [], options); 
                % alpha - Lagrange multiplers
                % fval  - optimal objective value

            %%%Status Print%%%
            if(verbosity_level)
                fprintf('Iteration %d (Dual objective value: %d)\n', ccnt, fval);
            end
            %%%%%%%%%%%%%%%%%%

            % Compute model vector 'w'
            w = zeros(d, 1);
            for j=1:ccnt %for each support vector
                w = w + alpha(j)*(psi_ideal_output{j} - psi_all_outputs{j});
            end       

            % Update slack variable value for latest 'w'
            xi = 0;
            for i=1:ccnt
                xi = max(xi, loss(i) - wdotpsi(Kquad(i, 1:ccnt), alpha));
            end        
        end
end

function wpsi = wdotpsi(K, alpha)
        % Compute w.psi(x, y) = \sum_{i = 1}^{no. of svs} alpha(i)*K(i)
        %              , where alpha is a vector of k Lagrange multipliers
        %                and K is an appropriate k x 1 matrix

        svcnt = size(alpha,1); % No. of support vectors

        % Compute \sum_{i = 1}^{svcnt} alpha(i)*K(i)
        wpsi = 0;
        for i = 1:svcnt
            wpsi = wpsi + alpha(i)*K(i);
        end
end
