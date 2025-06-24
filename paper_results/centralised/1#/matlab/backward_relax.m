function [v,u, g, y] = backward_relax(Ai, Bi, v_i1, u_i1, g_i1, n_u, unit_index)
% BACKWARD RELAXATION STARTING FROM NODE N AND PROPAGATING BACKWARD
% Ai, Bi, v_i are all dictionaries (all have common keys, indiciating unit
% index
% ---------------------
[min_vertex, truth] = minimization(Ai,Bi,v_i1);

[Avu, Hvu] = double(g_i1); % take the previous constraint set from the forward pass.

if unit_index == 1 || unit_index ==2
    if truth
        cons_add_A = Bi{4}*Ai{1};
        nu = size(Bi{3});
        conds_add_b = Ai{2}*transpose(min_vertex(1, 1:nu(2))) +  Bi{4}*Bi{1} + Bi{3}*transpose(min_vertex(1, nu(2)+1:end));
    else
        cons_add_A = Bi{2}*Ai{1};
        conds_add_b = Ai{2}*transpose(min_vertex) +  Bi{3};
    end

else
    cons_add_A = [Bi{2}*Ai{1}, Bi{2}*Bi{1}];
    conds_add_b = Ai{2}*transpose(min_vertex);

end

g = polytope([Avu; cons_add_A], [Hvu; -conds_add_b]); % add the constraint from the downstream unit

if unit_index == 1|| unit_index ==2
    u = [];
    v = g;
    v_f = g_i1;
    y = Ai{1} * g + Bi{1}; 
else
    u = projection(g, dimension(v_i1{1})+1:dimension(g));
    v = projection(g, 1:dimension(v_i1{1}));
    v_f = projection(g_i1, 1:dimension(v_i1{1}));
    y = [Ai{1}, Bi{1}] * g;
end



% figure;
% sgtitle('Feasible extended sets of unit i after forward-backward iteration')
% subplot(1,2,1)
% plot(v)
% title('V projection')
% if unit_index > 2
%     if dimension(u_i1) <3 
%         subplot(1,2,2)
%         plot(u)
%         title('U projection')
%     end
% end
% 
% figure;
% sgtitle('forward pass vs forward-backward iteration')
% subplot(1,2,1)
% plot(v_f)
% title('forward')
% subplot(1,2,2)
% plot(v)
% title('forward-backward')

end








