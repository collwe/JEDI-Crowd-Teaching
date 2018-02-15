function [D, Y] = gaussianData(mu1, mu2, sigma, numData)
%% generate binary gaussian data

% 
% %% 2 task classification
% r1 = mvnrnd(mu1,sigma1,numData/2);
% r2 = mvnrnd(mu2,sigma1,numData/2);
% 
% r3 = mvnrnd(mu3,sigma2,numData/2);
% r4 = mvnrnd(mu4,sigma2,numData/2);
% 
% D = [r1;r3; r2;r4];
% D = [D, ones(numData*2, 1)];
% Y = vertcat(ones(numData,1), -1*ones(numData,1));


%% single task classification
r1 = mvnrnd(mu1,sigma,numData);
r2 = mvnrnd(mu2,sigma,numData);

D = [r1;r2];
D = [D, ones(numData*2, 1)];
Y = vertcat(ones(numData,1), -1*ones(numData,1));
end