%% demo_harmonic on labeled data as well
% -- Yao Zhou
clc;clear;

%% load data
% dataname = '20NEWS\pc_mac\';
dataname = '20NEWS\baseball_hockey\'; % 0.01 label with 90 percent accuracy
% dataname = '20NEWS\religion_atheism\';
% dataname = '20NEWS\windows_mac\';

datapath = ['data\' dataname];
load([datapath 'graph_10NNw']) % W matrix
load([datapath 'Y']); % Y vector
W_init = W;

%% find a random labeled set
numClass = 2;
numData = length(Y);
lidx = randperm(numData);
lpercent = 0.02;
order = lidx(1:floor(numData*lpercent));

%% pad W with labeled nodes, get Wnew
Wnew = padWuseLabeledData(W, order);
Ynew = [Y; Y(order)];
numDataNew = length(Ynew);

%% flip the label of first 5 labeled points, verify their prob after apply harmonic function
Ynew(order(1:5)) = 1- Ynew(order(1:5));

% get labeled matrix fl
fl_all = labelvec2matrix(Ynew, numClass);
fl = fl_all(order,:);

% get reordered Wnew matrix
G = graph(Wnew);
lidxNew = [lidx length(Y)+1:length(Ynew)];
H = reordernodes(G,lidxNew);
nn = numnodes(H);
[s,t] = findedge(H);
W = sparse(s,t,H.Edges.Weight,nn,nn); % this is an upper W
W = W + W';

% run the code
[fu, fu_CMN] = harmonic_function(W, fl);

% verify the accuracy
Yu = Ynew(lidxNew(floor(numData*lpercent)+1:end)); 
[~,Predu] = max(fu');
accu = sum(Yu == (Predu-1)')/numDataNew



