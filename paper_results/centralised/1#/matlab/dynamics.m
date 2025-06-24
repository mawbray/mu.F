function [u_i1] = dynamics(Ai, Bi, v_i, u_i)
%   DYNAMICS FUNCTION
%   v_i - polytope, u_i polytope, Ai linear transformation of v_i, Bi
%   linear transformation of u_i
if isempty(u_i)
    u_i1 = minkowski_sum(linear_transform(Ai,v_i),  Bi);
else
    u_i1 = minkowski_sum(linear_transform(Ai,v_i), linear_transform(Bi,u_i));

end 