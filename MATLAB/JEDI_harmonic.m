function prob = JEDI_harmonic(learner, Y, A)

numData = length(Y);
numClass = length(unique(Y));
order = learner.order;
Ysl = learner.Ysl;

%% order could have duplicate, only use the latest ones, the same is for Ysl
[~, ia, ~] = unique(order, 'last');
uniqueIdx = sort(ia);
order_unique = order(uniqueIdx);
Ysl_unique = Ysl(uniqueIdx);

%% pad W with labeled nodes, get Wnew
Anew = padWuseLabeledData(A, order_unique);
Ynew = [Y; Y(order_unique)];
numDataNew = length(Ynew);

%% get labeled matrix fl (ground truth fl)
% fl_all = labelvec2matrixJEDI(Ynew, numClass);
% fl = fl_all(order,:);

%% get labeled matrix fl (learner provide labels)
fl = labelvec2matrixJEDI(Ysl_unique, numClass);

%% get reordered Wnew matrix
G = graph(Anew);
Yidx = (1:length(Y))';
Yidx(order_unique) = [];
reorderIdx = [order_unique; Yidx; (length(Y)+1:length(Ynew))'];
H = reordernodes(G,reorderIdx);

nn = numnodes(H);
[s,t] = findedge(H);
W = sparse(s,t,H.Edges.Weight,nn,nn); % this is an upper W
W = W + W';

%% run the code
[flu, ~] = harmonic_function(W, fl);

%% verify the accuracy
Yu = Ynew(length(order_unique)+1:end); 
[~,Predu] = max(flu');
accu = sum( (((-1)*Yu+1)/2 + 1) == Predu' )/numData;

%% output the probability w.r.t. the index of Dt
prob = zeros(numData, numClass);
prob(order_unique,:) = flu(end-length(order_unique)+1:end,:);
restIdx = setdiff( (1:numData)', order_unique);
prob(restIdx,:) = flu(1:end-length(order_unique),:);

end