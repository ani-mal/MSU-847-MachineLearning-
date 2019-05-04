function [X_complete] = hardimpute(X_missing, Omega, r)
% Input:
% X missing -- a m-by-n input matrix, only values at Omega
% Omega -- a m-by-n binary matrix, indicating location of the missing values
% r -- rank

old = zeros(size(X_missing));
for i=1:r
    old(Omega)=0;
    X = X_missing+old;
    [U,S,V] = svd(X);
    new = U(:,1:i)*S(1:i,1:i)*V(:,1:i)';
    old = new;
end
X_complete = new;