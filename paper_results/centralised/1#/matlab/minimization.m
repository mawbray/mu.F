function [min_vertex, truth] = minimization(Ai, Bi, v_i1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
poly_v = extreme(v_i1{2}); % find extrema of polytope
if length(v_i1.keys) > 2
    poly_u = extreme(v_i1{3});
    fn = @(vi, ui) Ai{2}*transpose(vi) + Bi{3}*transpose(ui);
    truth = true;
else
    fn = @(vi) Ai{2}*transpose(vi);
    poly_u = [];
    truth = false;
end

[min_val, min_vertex] = vertex_enumeration(poly_v, poly_u, fn); % find minimum of the upper level
   
end

