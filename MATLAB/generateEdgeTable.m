function [NodesS, NodesT, EdgeWeights] = generateEdgeTable(Edgeidx, Dist)
%% use this function to create the Edge Table

type = class(Edgeidx);

switch type
    case 'double'
        %% code for normal array
        [numS, numT] = size(Edgeidx);
        tempMat = repmat( (1:numS)',1, numT)';
        NodesS = tempMat(:);
        Edgeidx = Edgeidx';
        NodesT = Edgeidx(:);
    case 'cell'
        %% code for cell array
        numT = cellfun(@length, Edgeidx);
        NodesS = cell2mat(cellfun(@transpose, Edgeidx,'UniformOutput',false));
        NodesT = generateNodesT(numT);
end

% remove duplicate edge to support graph G
EdgeListSorted = sort([NodesS, NodesT],2);
[uniqueEdgeList, ia, ~] = unique(EdgeListSorted,'rows');
NodesS = uniqueEdgeList(:,1);
NodesT = uniqueEdgeList(:,2);

% remove the duplicate edge weights
EdgeWeightsAll = cell2mat(cellfun(@transpose,Dist,'UniformOutput',false));
EdgeWeights = EdgeWeightsAll(ia);

end



function NodesT = generateNodesT(numT)

NodesT = zeros(sum(numT),1);
count = 0;
for i = 1:length(numT)
    NodesT(count+1:count+numT(i),1) = i * ones(numT(i),1);
    count = count + numT(i);
end

end