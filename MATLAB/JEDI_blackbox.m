function [selectIdx, selectProb] = JEDI_blackbox(D, Y, learner, wo, step, A)
% implement the JEDI(Interactive Crowd Teaching with adJustable Exponential Decay Memory Learners)
% by -- Yao Zhou
% VERSION 1: use harmonic
%  
% Input:
%   (D, Y)  : teaching pool
%   learner : the learners and their assets
%       wo  : the target concept
%      step : learninig rate
% Output:
%       selectIdx is the index of selected teaching examples
Ws = learner.Ws;
Xs = learner.Xs;
Ys = learner.Ys;
Ysl = learner.Ysl;
Ysl_prob = learner.Ysl_prob;
beta = learner.beta;
order = learner.order;

numData = length(Y);
fvalue = zeros(numData,1);
eta = step;
d = size(Xs,2);
tminus = length(learner.Ys);

%% estimate p(yt = +1 or -1 | Xt, Ds, Ysl) using harmonic function:
% Ds,Ysl are teaching sequences (features and learner labels)

Prob = JEDI_harmonic(learner, Y, A);

% calculat the Ps, where s = 1,2,...,t-1
% Def:      Ps = 1/(1+exp(ys * w'_{s-1} * xs))
Ps = zeros(length(order),1);
for is = 1:length(order)
    ys = Ys(is);
    if ys == 1
        Ps(is) = Ysl_prob(is,2);
    else % ys == -1
        Ps(is) = Ysl_prob(is,1);
    end
end
coeff = beta.^(tminus:-1:1);
FS = sum(repmat((-1*Ys).*Ps,1,d) .* Xs .* repmat(coeff',1,d), 1); 


w = Ws(:,end);
for id = 1:numData % loop over all potential (xt, yt)
    x = D(id,:);
    y = Y(id,:);
    
    if y == 1
        pt = Prob(id, 2); % use harmonic to estimate pt = 1/(1+exp(yt * w'_{t-1} * xt))
        pnt = 1/Prob(id,1); % use harmonic to estimate pnt = 1 + exp(-1 * yt * w'_{t-1} * xt)
    else % y == -1
        pt = Prob(id, 1); % use harmonic to estimate pt = 1/(1+exp(yt * w'_{t-1} * xt))
        pnt = 1/Prob(id,2) ; % use harmonic to estimate pnt = 1 + exp(-1 * yt * w'_{t-1} * xt)
    end
    epsilon_o = y * (wo' * x');
    
    fvalue(id,1) = eta^2 * pt^2 * norm(x)^2  ...
                 + 2 * eta^2 * FS * (-1*y*x'*pt) ...
                 - 2 * eta * ( log(pnt) - log(1+exp(-1*epsilon_o)) ); 
end


% % calculat the fs, where s = 1,2,...,t-1
% tminus = length(learner.Ys);
% product_s = diag( learner.Xs * Ws(:,1:end-1) );
% epsilon_s = Ys .* product_s;
% coeff = beta.^(tminus:-1:1);
% FS = sum(repmat((-1*Ys)./(1+exp(epsilon_s)),1,d) .* Xs .* repmat(coeff',1,d), 1); 
% 
% w = Ws(:,end);
% for id = 1:numData
%     x = D(id,:);
%     x = x';
%     y = Y(id,:);
%     epsilon = y * (w' * x);
%     epsilon_o = y * (wo' * x);
%     
%     fvalue(id,1) = eta^2 * norm(-1*y*x/(1+exp(epsilon)))^2  ...
%                  + 2 * eta^2 * FS * (-1*y*x/(1+exp(epsilon))) ...
%                  - 2 * eta * ( log(1+exp(-1*epsilon)) - log(1+exp(-1*epsilon_o)) ); 
% end
% 

% results output
[~, selectIdx] = min(fvalue);
selectProb = Prob(selectIdx,:); % for JEDI harmonic use only


end