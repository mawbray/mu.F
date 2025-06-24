function [current_min,minima] = vertex_enumeration(v_i, u_i, callable)
% MINIMIZATION OVER V DECISION VARIABLES
% FUNCTION TAKES V_I, evaluates all vertices and returns the min.
n_u = size(u_i);
if n_u(1,1) == 0 
    n_vertices_v = size(v_i);
    current_min = 1e8;
    for i=1:n_vertices_v(1)
        fn = max(callable(v_i(i,:))); % take max constraint violation (this is actually very sensitive)
        if fn < current_min
            current_min = fn;
            minima = [v_i(i,:)];

        end
    end
else
    n_vertices_v = size(v_i);
    n_vertices_u = size(u_i);
    current_min = 1e8;
    for j=1:n_vertices_u(1)
        for i=1:n_vertices_v(1)
            fn = max(callable(v_i(i,:), u_i(j,:))); % take max constraint violation (this is actually very sensitive)
            if fn < current_min
                current_min = fn;
                minima = [v_i(i,:), u_i(j,:)];

            end
        end
    end
end