function [v,v1,v2,v3,v4,v5,polytopes_by_pair] = simultaneous_all(A_v, b_v, n_x, n_u, kv_1, kv_2, kv_3, kv_4, kv_5)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here
[Av1, Hv1] = double(kv_1);
[Av2, Hv2] = double(kv_2);
[Av3, Hv3] = double(kv_3);
[Av4, Hv4] = double(kv_4);
[Av5, Hv5] = double(kv_5);


joint_k = cartprod_polytopes(cartprod_polytopes(cartprod_polytopes(cartprod_polytopes(kv_1, kv_2), kv_3), kv_4), kv_5);

v = polytope([[full(A_v)]; [Av1, zeros(nconstr(kv_1),4*n_x)]; [zeros(nconstr(kv_2),n_x), Av2, zeros(nconstr(kv_2),3*n_x)]; [zeros(nconstr(kv_3),2*n_x), Av3,zeros(nconstr(kv_3),2*n_x)]; [zeros(nconstr(kv_4),3*n_x), Av4, zeros(nconstr(kv_3),n_x)];[zeros(nconstr(kv_5),4*n_x), Av5]], [-b_v; Hv1; Hv2; Hv3; Hv4; Hv5] );

n_dims = 10;  % 10 dimensions in your polytope
pairs = nchoosek(1:n_dims, 2);  % Generate all pairs of dimensions

% Create figure for plotting projections
polytopes_by_pair = struct();
i =1;
% Loop through each pair of dimensions
while i < size(pairs, 1)+1
    dim_pair = pairs(i, :);
    disp(['Computing projection onto dimensions ', num2str(dim_pair)]);
    
    % Compute the projection for the current pair of dimensions
    polytopes_by_pair.(sprintf('dim_pair_%d_%d', dim_pair(1), dim_pair(2))) = extreme(projection(v, dim_pair));
    i = i+1;

end


v1 = intersection(projection(v,1:n_x), kv_1);
v2 = intersection(projection(v,1+n_x:2*n_x), kv_2);
v3 = intersection(projection(v,2*n_x+1:3*n_x), kv_3);
v4 = intersection(projection(v,3*n_x+1:4*n_x), kv_4);
v5 = intersection(projection(v,4*n_x+1:5*n_x), kv_5);

end