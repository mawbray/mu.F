function [reduction] = backwardpropagation(param_dict,KS)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
n_x = 2;
n_u = 1; %%
[A1, B1] = param_dict{1}{:};
[A2, B2] = param_dict{2}{:};
[A_34, B_314, B_324] = param_dict{34}{:};
[A_35, B_315, B_325] = param_dict{35}{:};
[A4, B4] = param_dict{4}{:};
[A5, B5] = param_dict{5}{:};

K_v_1 = KS{1};
K_v_2 = KS{2};
K_v_3 = KS{3}; 
K_v_4 = KS{4};
K_v_5 = KS{5};

%% simultaneous
[A_v, b_v] = construct_v(A1, B1, A2, B2, A_34, A_35, B_314, B_324, B_315, B_325, A4, B4, A5, B5); %A_1, B_1, A_2, B_2, A_34, A_35, B_314, B_324, B_315, B_325, A_4, B_4, A_5, B_5
[v_sim, v_sim1,v_sim2, v_sim3, v_sim4, v_sim5] = simultaneous(A_v, b_v, n_x, n_u, K_v_1, K_v_2, K_v_3, K_v_4, K_v_5);

%% simultaneous unit outputs 
y_1s = dynamics(A1, B1, v_sim1, []);
y_2s = dynamics(A2, B2, v_sim2, []);
u_3s = cartprod_polytopes(y_1s,y_2s);
y_34s = dynamics(A_34, [B_314, B_324], v_sim3, u_3s);
y_35s = dynamics(A_35, [B_315, B_325], v_sim3, u_3s); %, polytope(eye(n_u), zeros(n_u,1)));
y_4s = dynamics(A4, B4, v_sim4, y_34s); %, polytope(eye(n_u), zeros(n_u,1)));
y_5s = dynamics(A5, B5, v_sim5, y_35s); %, polytope(eye(n_u), zeros(n_u,1)));

%% plot unit outputs
figure;
sgtitle('sim unit outputs')
subplot(2, 6, 1)
plot(y_1s);
subplot(2, 6, 2);
plot(y_2s);
subplot(2, 6, 3);
plot(y_34s);
subplot(2, 6, 4);
plot(y_35s);
subplot(2, 6, 5);
plot(y_4s);
subplot(2, 6, 6);
plot(y_5s);

%% forward propagation 
%% unit 1 forward propagation 
y_i_1 = dynamics(A1, B1, K_v_1, []);

%% unit 2 forward propagation 
y_i_2 = dynamics(A2, B2, K_v_2, []);

%% unit 3 forward propagation
u_3_i = cartprod_polytopes(y_i_1,y_i_2);
y_i_3 = dynamics([A_34; A_35], [B_314, B_324; B_315, B_325], K_v_3, u_3_i);

%% unit 4 forward propagation 
u_4_i = projection(y_i_3, 1:n_u); 
y_i_4 = dynamics(A4, B4, K_v_4, u_4_i);

%% unit 5 forward propagation 
u_5_i = projection(y_i_3, n_u+1:2*n_u);
y_i_5 = dynamics(A5, B5, K_v_5, u_5_i);

g_1f = K_v_1;
g_2f = K_v_2;
g_3f = cartprod_polytopes(K_v_3, u_3_i);
g_4f = cartprod_polytopes(K_v_4, u_4_i);
g_5f = cartprod_polytopes(K_v_5, u_5_i);

%% plot unit outputs
figure;
sgtitle('unconstrained unit outputs')
subplot(2, 5, 1)
plot(y_i_1);
subplot(2, 5, 2);
plot(y_i_2);
subplot(2, 5, 3);
plot(y_i_3);
subplot(2, 5, 4);
plot(y_i_4);
subplot(2, 5, 5);
plot(y_i_5);



%% plot forward vs backward
% figure;
% sgtitle('Simultaneous v forward')
% subplot(2, 5, 1)
% plot(v_sim1);
% subplot(2, 5, 2);
% plot(v_sim2);
% subplot(2, 5, 3);
% plot(v_sim3);
% subplot(2, 5, 4);
% plot(v_sim4);
% subplot(2, 5, 5);
% plot(v_sim5);
% subplot(2, 5, 6);
% plot(v_1f);
% subplot(2, 5, 7);
% plot(v_2f);
% subplot(2, 5, 8);
% plot(v_3f);
% subplot(2, 5, 9);
% plot(v_4f);
% subplot(2, 5, 10);
% plot(v_5f);

%% backward propagation
[g_1fb, g_2fb, g_3fb, g_4fb, g_5fb] = forward_backward_propagaton(param_dict, dictionary([1,2,3,4,5], {g_1f, g_2f, g_3f, g_4f, g_5f}));

v_1fb = projection(g_1fb, 1:n_x) ;
v_2fb = projection(g_2fb, 1:n_x) ;
v_3fb = projection(g_3fb, 1:n_x) ;
v_4fb = projection(g_4fb, 1:n_x) ;
v_5fb = projection(g_5fb, 1:n_x) ;

u_1fb = [] ;
u_2fb = [] ;
u_3fb = projection(g_3fb, n_x+1:n_x+2*n_u) ;
u_4fb = projection(g_4fb, n_x+1:n_x+n_u) ;
u_5fb = projection(g_5fb, n_x+1:n_x+n_u) ;

reduction = dictionary([1,2,3,4,5], {v_1fb, v_2fb, v_3fb, v_4fb, v_5fb});

% % %% plot forward vs backward
figure;
sgtitle('Simultaneous v backward')
subplot(2, 5, 1)
plot(v_sim1);
subplot(2, 5, 2);
plot(v_sim2);
subplot(2, 5, 3);
plot(v_sim3);
subplot(2, 5, 4);
plot(v_sim4);
subplot(2, 5, 5);
plot(v_sim5);
subplot(2, 5, 6);
plot(v_1fb);
subplot(2, 5, 7);
plot(v_2fb);
subplot(2, 5, 8);
plot(v_3fb);
subplot(2, 5, 9);
plot(v_4fb);
subplot(2, 5, 10);
plot(v_5fb);