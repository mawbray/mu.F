function [v, u, g] = constraint_fn(Ai, Bi, v_i, u_i)
%   CONSTRAINT FUNCTION
%   Detailed explanation goes here

if isempty(u_i)
    [Av, Hv] = double(v_i);
    v = polytope([Ai; Av] , [-Bi; Hv]);
    g= v
    u = [];
else
    [Av, Hv] = double(v_i);
    [Au, Hu] = double(u_i);
    n_au = size(Au);
    nai = size(Ai);
    g = polytope([[Ai, Bi];[zeros(n_au(1),dimension(v_i)), Au]; [Av, zeros(nconstr(v_i),dimension(u_i))]], [zeros(nai(1),1); Hu; Hv]);
    u = projection(g, dimension(v_i)+1:dimension(v_i)+dimension(u_i));
    v = projection(g, 1:dimension(v_i));

    % figure;
    % sgtitle('Feasible extended sets of unit i')
    % subplot(1,2,1)
    % plot(v)
    % title('V projection')
    % if n_au < 3
    %     subplot(1,2,2)
    %     plot(u)
    %     title('U projection')
    % end
    



end