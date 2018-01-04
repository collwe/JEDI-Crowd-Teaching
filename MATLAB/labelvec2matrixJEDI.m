function Y = labelvec2matrixJEDI(y, k) 
% k is # of classes (JEDI has 2 classes for the moment)
% y has labels of -1 and +1
% Y has labels [1 0] (for +1) and [0 1] (for -1)

m = length(y); 
Y = repmat(y(:),1,k) == repmat([+1 -1],m,1);

end