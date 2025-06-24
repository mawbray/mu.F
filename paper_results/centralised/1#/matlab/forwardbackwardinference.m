% script to get closed form for the simulataneous space and to check
% whether terminal node of forward relaxation produces an equivalent set to
% that of the simultaneous approach. If yes then one forward-backward
% iteration should recover the feasible space for DAG process networks.

%% dynamics and constraint matrices (both are equivalent based on the study) - change matrices to check is not CS dependent
N = 1;
n_x = 2;
n_u = 1;
x = true;
counter = {}; 
k = 0;
while x
    % A1 = randn(n_u, n_x) ; %- 0.5;
    % A2 = randn(n_u, n_x);
    % A_34 = randn(n_u, n_x);
    % A_35 = randn(n_u, n_x);
    % A4 = randn(n_u, n_x);
    % B1 = randn(n_u,1);  %- 0.5;  % seems to be only feasible if coefficients are negative.
    % B2 = randn(n_u, 1) ;  %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_314 = randn(n_u, n_u) ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_315 = randn(n_u, n_u); %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_324 = randn(n_u, n_u) ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_325 = randn(n_u, n_u) ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B4 = randn(n_u, n_u)  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % A5 = randn(n_u, n_x) ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B5 = randn(n_u, n_u) ;%-10000  ; %- 0.5; % seems to be only feasible if coefficients are negative.

    % A1 = [0.280760520932057,0.463706174446316] ; %- 0.5;
    % A2 = [-1.655103224914643,-0.370045753471811];
    % A_34 = [-0.439218250485741,0.059728361785703];
    % A_35 = [-1.359403730073284,-2.202241365485294];
    % A4 = [-0.006506641987944,-1.200776736727198];
    % B1 = [-0.683232521583820] ;  %- 0.5;  % seems to be only feasible if coefficients are negative.
    % B2 = [-0.232258581158790] ;  %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_314 = [1.579711436561928] ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_315 = [0.583590208451222]; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_324 = [0.870551972197199] ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_325 = [-0.531351160960495] ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B4 = [-1.3059255566201941]  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % A5 = [-0.009617962323625,1.349253742536704] ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B5 = [0.320454875840786] ;%-10000  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    
    
    %% Knowledge spaces
    K_v_1 = polytope(transpose([diag(ones(n_x,1)),-diag(ones(n_x,1))]), transpose([ones(1,n_x), ones(1,n_x)])*1);
    K_v_2 = polytope(transpose([diag(ones(n_x,1)),-diag(ones(n_x,1))]), transpose([ones(1,n_x), ones(1,n_x)])*1);
    K_v_3 = polytope(transpose([diag(ones(n_x,1)),- diag(ones(n_x,1))]), transpose([ones(1,n_x), ones(1,n_x)])*1);
    K_v_4 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x), ones(1,n_x)])*1);
    K_v_5 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x), ones(1,n_x)])*1);
    

    % A1 = -zeros(n_u, n_x) +1 ; %- 0.5;
    % A2 = eye(n_u, n_x)*2 + 1 ;
    % A_34 = zeros(n_u, n_x) +1;
    % A_35 = eye(n_u, n_x)*1 ;
    % A4 = eye(n_u, n_x)*2+ 1 ;
    % B1 = -eye(n_u, 1)*1 ;  %- 0.5;  % seems to be only feasible if coefficients are negative.
    % B2 = -eye(n_u, 1)*1  ;  %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_314 = -eye(n_u, n_u)*1 +1  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_315 = -eye(n_u, n_u)*1 -3; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_324 = zeros(n_u, n_u)+ 1  +2 ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B_325 = -eye(n_u, n_u)*0.1   ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B4 = eye(n_u, n_u)*10 ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % A5 = eye(n_u, n_x)+ 10  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % B5 = eye(n_u, n_u)+10  ; %- 0.5; % seems to be only feasible if coefficients are negative.
    % 
    % %% Knowledge spaces
    % K_v_1 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x), 1*ones(1,n_x)])*10);
    % K_v_2 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x), 1*ones(1,n_x)])*50);
    % K_v_3 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x),1*eye(1,n_x)])*10);
    % K_v_4 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x),0*ones(1,n_x)])*10);
    % K_v_5 = polytope(transpose([diag(ones(n_x,1)), -diag(ones(n_x,1))]), transpose([ones(1,n_x), 1*ones(1,n_x)])*10);
    
    % plot KS polytopes
    % figure;
    % sgtitle('Original Knowledge spaces')
    % subplot(1, 4, 1);
    % plot(K_v_1);
    % title('Unit 1 KS');
    % subplot(1, 4, 2);
    % plot(K_v_2);
    % title('Unit 2 KS');
    % subplot(1, 4, 3);
    % plot(K_v_3);
    % title('Unit 3 KS');
    % subplot(1, 4, 4);
    % plot(K_v_3);
    % title('Unit 4 KS');
    
    extrema = extreme(K_v_1);
    
    %% simultaneous
    [A_v, b_v] = construct_v(A1, B1, A2, B2, A_34, A_35, B_314, B_324, B_315, B_325, A4, B4, A5, B5); %A_1, B_1, A_2, B_2, A_34, A_35, B_314, B_324, B_315, B_325, A_4, B_4, A_5, B_5
    [v_sim, v_sim1,v_sim2, v_sim3, v_sim4, v_sim5] = simultaneous(A_v, b_v, n_x, n_u, K_v_1, K_v_2, K_v_3, K_v_4, K_v_5);
    [v_sim, v_sim1,v_sim2, v_sim3, v_sim4, v_sim5, cell] = simultaneous_all(A_v, b_v, n_x, n_u, K_v_1, K_v_2, K_v_3, K_v_4, K_v_5);
    
    bounded1 = isbounded(v_sim);
    bounded2 = isbounded(v_sim3);
    
    % with knowledge space restriction
    % figure;
    % sgtitle('Simultaneous feasible space')
    % subplot(1, 5, 1);
    % plot(v_sim1);
    % title('Unit 1 Sim.');
    % subplot(1, 5, 2);
    % plot(v_sim2);
    % title('Unit 2 Sim.');
    % subplot(1, 5, 3);
    % plot(v_sim3);
    % title('Unit 3 Sim.');
    % subplot(1, 5, 4);
    % plot(v_sim4);
    % title('Unit 4 Sim.');
    % subplot(1, 5, 5);
    % plot(v_sim5);
    % title('Unit 5 Sim.');
    
    reduction = dictionary([1,2,3,4,5], {K_v_1, K_v_2, K_v_3, K_v_4, K_v_5});
    params = dictionary([1,2,34,35,4,5], {{A1, B1}, {A2, B2}, {A_34, B_314, B_324}, {A_35, B_315, B_325}, {A4, B4}, {A5, B5}});
    epsilon = dictionary([1,2,3,4,5], {0,0,zeros(2),0,0})
    for i=1:N
        reductionf = forwardpropagation(params,reduction);
        reductionb = backwardpropagation(params, reduction);
        reduction1 = forwardbackwardpropagation(params, reduction);
    end
        
    %
    % t3 = mean(mean(abs(sortrows(round(extreme(reduction1{3}),4,'significant'),[1,2]) - sortrows(round(extreme(v_sim3),4,'significant'),[1,2])) < 1e-4));
    % t1 = mean(mean(abs(sortrows(round(extreme(reduction1{4}),4,'significant'),[1,2]) - sortrows(round(extreme(v_sim4),4,'significant'),[1,2])) < 1e-4));
    % t2 = mean(mean(abs(sortrows(round(extreme(reduction1{5}),4,'significant'),[1,2]) - sortrows(round(extreme(v_sim5),4,'significant'),[1,2])) < 1e-4));
    % t4 = mean(mean(abs(sortrows(round(extreme(reduction1{2}),4,'significant'),[1,2]) - sortrows(round(extreme(v_sim2),4,'significant'),[1,2])) < 1e-4));
    % t5 = mean(mean(abs(sortrows(round(extreme(reduction1{1}),4,'significant'),[1,2]) - sortrows(round(extreme(v_sim1),4,'significant'),[1,2])) < 1e-4));
    %
    % x = t5 + t2 + t1  + t2 + t3 ;
    % counter = {counter, {t5,t4,t3,t2,t1} };
    % k = k +1; 
    % if k > 1000 
    x = false;
    % end

end

% t1=0;
% t2=0;
% t3=0;
% t4=0;
% t5=0;
% cond=true;
% while cond
%     array = counter{1,1}{2};
%     arr = array;
%     t1 = t1+arr{1};
%     t2 = t2+arr{2};
%     t3 = t3+arr{3};
%     t4 = t4+arr{4};
%     t5 = t5+arr{5};
%     counter = counter{1,1}{1};
%     if length(counter{1,1})<1
%         cond=false;
%     end
% end


% if x == false
%     save("leaf_node_forward_propagation_convergence.mat","A1", "B1", "A2", "B2", "A_34", "A_35", "B_314", "B_324", "B_315", "B_325", "A4", "B4", "A5", "B5")
%     save("forward_propagation_no_convergence_projections.mat","reduction1")
% 
% end