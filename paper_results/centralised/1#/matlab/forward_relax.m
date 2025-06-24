function [v_i,u_i,g_i,y_i] = forward_relax(Ai, Bi, v_i1, u_i1, n_u)
% inputs dynamics + constraint terms, + knowledge spaces.
% compute the forward relaxation given constraints
[v_i, u_i, g_i] = constraint_fn(Ai, Bi, v_i1, u_i1); % compute the constraint set 
if isempty(u_i1)
    y_i = Ai* g_i + Bi;
else 
    y_i = [Ai, Bi]* g_i;
end

end



