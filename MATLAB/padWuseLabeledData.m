function Wnew = padWuseLabeledData(W, order)
%% use this function to pad labeled points to the affinity matrix W
% -- Yao Zhou
% 
%   Input:  W     -- Original Affinity Matrix W
%           order -- Labeled data points index
%   Output: Wnew  -- Padded (using labelled points) Affinity Matrix W

%% Example to use
% lidx = [1 3];
% W = rand(4);
% W = W+ W';
% 
% Wnew = padWuseLabeledData(W,lidx);

n = size(W,1);
nl = length(order);

Wnew = W;
Wnew(end+1:end+nl,:) = 0;
Wnew(:,end+1:end+nl) = 0;

Wnew(n+1:end,:) = Wnew(order,:);
Wnew(:,n+1:end) = Wnew(:,order);
Wnew(n+1:n+nl, n+1:n+nl) = W(order,order); % not really necessary




end
