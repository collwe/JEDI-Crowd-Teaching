%% This demo is for JEDI teacher with harmonic function esimation
%  Dataset: random 10-D Gaussian data 

loadData = true;
%% Load (Generate) THE DATA
if loadData
    load('data/Data.mat')
else
    d = 10;
    mu1 = -.6*ones(10,1);
    mu2 = .6*ones(10,1);
    sigma1 = diag(round(rand(d,1)*10)); 
    numData = 1000;

    accu_LR = 0;
    count = 0;
    while accu_LR < 0.80 || accu_LR >= 0.95 % quality control of the generated data
        if count ~= 0
            fprintf('Random Guassian Data Generation # %d, accu_LR = %.2f...\n', count, accu_LR);
        end
        [D, Y] = gaussianData(mu1, mu2, sigma1, numData);

        % split into teaching set and evaluation set
        ratio = 0.2;
        randidx = randperm(numData*2);
        tidx = randidx(1:floor(numData*2*ratio))';
        eidx = setdiff(1:numData*2, tidx)';

        % learn the target concept w* on teaching set
        pathPRML = 'PRML/code';
        addpath(genpath(pathPRML));

        X = D(tidx,1:d)';
        [model, llh] = logitBin(X,(Y(tidx)'+1)/2);
        [y, p] = logitBinPred(model, X);
        pred_LR = y == ((Y(tidx)'+1)/2);
        accu_LR = sum(pred_LR)/length(y);
        wo_LR = model.w;

        count = count+1;
        rmpath(genpath(pathPRML))
        if count >= 20 % just rerun the code if the random seed is not good
            return
        end
    end
    % teaching set
    Dt = D(tidx,:); Yt = Y(tidx);
    % evaluation set
    De = D(eidx,:); Ye = Y(eidx);
    num_t = length(tidx);
end

%% generate the Affinity matrix A of the teaching set
X = Dt(:,1:d);
Xnorm = X./repmat(diag(sqrt(sigma1))',num_t,1);
[Edgeidx, Dist] = knnsearch(Xnorm, Xnorm, 'K', 11, 'IncludeTies', true);

fcn = @removeFirst; % remove the NN of node itself
Edgeidx =  cellfun(fcn, Edgeidx, 'UniformOutput', false);
Dist =  cellfun(fcn, Dist, 'UniformOutput', false);

fcn2 = @(x) exp(-1*x.^2); % from ICML 04 Harmonic paper
Dist = cellfun(fcn2, Dist, 'UniformOutput', false);
[NodesS, NodesT, EdgeWeights] = generateEdgeTable(Edgeidx, Dist);

A = full(sparse(NodesS, NodesT, EdgeWeights, num_t, num_t));
A = A + A';

%% GENERATE LEARNERS and TEACHING USING JEDI
maxIter = 500;
step_init = 0.03;

% learners 
Beta =      [0.0, 0.368, 0.5, 0.75, 0.875, 0.999];
numMemory = [0,   1,   2,   4,   8,   Inf];
numLearner = length(Beta);

% teacher assets
fvalue_JEDI = zeros(maxIter,numLearner);
teachingSetJEDI = zeros(maxIter,numLearner);

w0 = (-1 + rand(d+1,1)*2) .* ones(d+1,1);
selectIdxFirst = randperm(length(Yt),1);
for il = 1:numLearner
    beta = Beta(il);
    learner(il) = learnerClass(beta, w0);
    w = w0;
    
    for it = 1:maxIter
        if mod(it,100) == 0
            fprintf('JEDI for worker # %d of iteration %d...\n',il, it);
        end
        
        % teacher estimate and select examples to teach
%         step = step_init;
        step = step_init*20/(20+it);
        if it > 1
%             [selectIdx] = JEDI_blackbox_v1(Dt, Yt, learner(il), wo_LR, step);
            [selectIdx, selectProb] = JEDI_blackbox(Dt, Yt, learner(il), wo_LR, step, A);
        else % first teaching example
            selectIdx = selectIdxFirst;
            selectProb = [0.5, 0.5];
        end
        teachingSetJEDI(it,il) = selectIdx;
        
        % learner make prediction on (xt)
        x = Dt(selectIdx,:);
        ysl = sign(w' * x');
        
        % learner learns (For real human learner, the learning are assumed to be done within their mind.)
        y = Yt(selectIdx);
        epsilon = y * (w' * x');
        dw = 1/(1+exp(epsilon)) * (-1*y*x');
        w = w - step * dw;
        
        % update the learner assets
        learner(il).Ysl_prob = [learner(il).Ysl_prob; selectProb]; 
        learner(il).Ysl = [learner(il).Ysl; ysl];
        learner(il).Xs = [learner(il).Xs; x];
        learner(il).Ys = [learner(il).Ys; y];
        learner(il).Ws = [learner(il).Ws w];
        learner(il).order = [learner(il).order; selectIdx];
        
        % function objective
        fvalue_JEDI(it,il) = sum(log(1+exp(-1*Dt*w .* Yt)));
    end
    
    pred_JEDI_train = Dt * w;
    pred_JEDI_train = (pred_JEDI_train >= 0) == ((Yt+1)/2);
    accu_JEDI_train(il,1) = sum( pred_JEDI_train )/length(Yt);
    
    pred_JEDI_eval = De * w;
    pred_JEDI_eval = (pred_JEDI_eval >= 0) == ((Ye+1)/2);
    accu_JEDI_eval(il,1) = sum( pred_JEDI_eval )/length(Ye);
    
    fprintf('Teaching of learner #%d (beta = %0.3f) is done...\n\n', il, learner(il).beta);
end

%% plot the negative log-likelihood curve
figure; hold on
txt = cell(numLearner,1);
for il = 1:numLearner
    if il == 1
        plot(fvalue_JEDI(1:maxIter, il), '--', 'LineWidth', 1.2)
        txt{il} = 'IMT';
    else
        plot(fvalue_JEDI(1:maxIter, il),'LineWidth', 1.2)
        txt{il} = sprintf('JEDI: \\beta = %0.3f',learner(il).beta);
    end
end
title('Convergence comparison','Interpreter','latex');
xlabel('# Iterations');
ylabel('Negative log-likelihood')
legend(txt);

disp('THE END')