function pAUC = eval_pCROC(X,y,w,options)
tmp = w'*X;
[~,sorted_ind] = sort(tmp,'descend');
y = y(sorted_ind); 
neg_ind = find(y<=0);
n_neg = length(neg_ind);
n_pos = length(y)-n_neg;
j_alpha = ceil(n_neg*options.alpha);
j_beta = floor(n_neg*options.beta);
% Computing pAUC
pAUC = 0;
pAUC = pAUC + (j_alpha-n_neg*options.alpha)*count_preceding_pos(y,neg_ind,j_alpha);
for i=j_alpha+1:j_beta
    pAUC = pAUC + count_preceding_pos(y,neg_ind,i);
end
pAUC = pAUC + (n_neg*options.beta-j_beta)*count_preceding_pos(y,neg_ind,j_beta);
pAUC = pAUC/(n_pos*n_neg*(options.beta-options.alpha));
end

function num = count_preceding_pos(y,neg_ind,i)
    label_minus = min(y);
    if label_minus == -1
        num = 0.5*sum(y(1:neg_ind(i))+1);
    else % if label_minus==0
        num = sum(y(1:neg_ind(i)));
    end
end
