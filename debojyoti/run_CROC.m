clear
%% Load datasets
path_split = '/home/debojyoti/Dropbox/CSCDatasets/kdd08.splits.mat';
path_X = '/home/debojyoti/Dropbox/CSCDatasets/kdd08.X.mat';
path_y = '/home/debojyoti/Dropbox/CSCDatasets/kdd08.y.mat';
load(path_split,'splits')
load(path_X,'X')
load(path_y,'y')
n_train = splits.numTraining;
n_val = splits.numValidating;
ids = splits.IDs;
ids = ids(:,1);
X_train = X(:,ids(1:n_train));
y_train = y(ids(1:n_train));
X_val = X(:,ids(n_train+1:n_train+n_val));
y_val = y(ids(n_train+1:n_train+n_val));
flen = length(X(:,1));

% Parameter settings
options.alpha = 0.0;
options.beta = 0.3;
options.C = 5;

% Call pAUC function
w = svmpauc_tight(X_train', y_train, options.alpha, options.beta, options.C);

% Evaluate
auc = eval_pCROC(X_val,y_val,w,options);
fprintf('[After Training] pAUC(%f - %f): %f\n',options.alpha,options.beta,auc);