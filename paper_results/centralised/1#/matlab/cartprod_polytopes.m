function [obj] = cartprod_polytopes(A, B)

% return the cartesian product of two sets
[Aa, Ha] = double(A);
[Ab, Hb] = double(B);
[nA, nvA] = size(Aa);
[nB, nvB] = size(Ab);

% 
Ac = [[Aa, zeros(nA, nvB)]; [zeros(nB, nvA), Ab]];
Hc = [Ha; Hb];
obj = polytope(Ac,Hc);