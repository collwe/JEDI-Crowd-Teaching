function [selectIdx] = JEDI_blackbox_v1(D, Y, learner, wo, step)
% implement the JEDI(Interactive Crowd Teaching with adJustable Exponential Decay Memory Learners)
% by -- Yao Zhou
% VERSION 1: use Ws, Wt
%  
% Input:
%   (D, Y)  : teaching pool
%   learner : the learners and their assets
%       wo  : the target concept
%      step : learninig rate
% Output:
%       selectIdx is the index of selected teaching examples
Xs = learner.Xs;
Ys = learner.Ys;
Ws = learner.Ws;
beta = learner.beta;

numData = length(Y);
fvalue = zeros(numData,1);
eta = step;
d = size(Xs,2);


% calculat the fs, where s = 1,2,...,t-1
tminus = length(learner.Ys);
product_s = diag( learner.Xs * Ws(:,1:end-1) );
epsilon_s = Ys .* product_s;
coeff = beta.^(tminus:-1:1);
FS = sum(repmat((-1*Ys)./(1+exp(epsilon_s)),1,d) .* Xs .* repmat(coeff',1,d), 1); 

w = Ws(:,end);
for id = 1:numData
    x = D(id,:);
    x = x';
    y = Y(id,:);
    epsilon = y * (w' * x);
    epsilon_o = y * (wo' * x);
    
    fvalue(id,1) = eta^2 * norm(-1*y*x/(1+exp(epsilon)))^2  ...
                 + 2 * eta^2 * FS * (-1*y*x/(1+exp(epsilon))) ...
                 - 2 * eta * ( log(1+exp(-1*epsilon)) - log(1+exp(-1*epsilon_o)) ); 
end

[~, selectIdx] = min(fvalue);

% x = D(selectIdx,:);
% x = x';
% y = Y(selectIdx,:);
% epsilon = y * (w' * x);
% term2value = 2 * eta^2 * FS * (-1*y*x/(1+exp(epsilon)))

end