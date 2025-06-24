function [g_1f,g_2f,g_3_fb,g_4f,g_5f] = forward_propagation(param_dict,KS)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
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


%% ----- forward relaxation ------------- %%
%% unit 1 forward relax
[v_1f, u_1f, g_1f, y_1f] = forward_relax(A1, B1, K_v_1, [], n_u);

%% unit 2 forward relax  
[v_2f, u_2f, g_2f, y_2f] = forward_relax(A2, B2, K_v_2, [], n_u);
u_3_f = cartprod_polytopes(y_1f,y_2f);

%% unit 3 forward relax
[v_34f, u_34f, g_34f, y_34f] = forward_relax(A_34, [B_314,B_324], K_v_3, u_3_f, n_u);
[v_3f, u_3f, g_3f, y_35f] = forward_relax([A_35, B_315,B_325], zeros(1*n_u,1), g_34f, [], n_u);
v_3f = projection(g_3f, 1:n_x);
u_3f = projection(g_3f, n_x+1:n_x+2*n_u);
y_35f = [A_35, B_315,B_325] * g_3f;
y_34 = [A_34, B_314,B_324] * g_3f;

%% unit 4 forward relax
[v_4f, u_4f, g_4f, y_4f] = forward_relax(A4, B4, K_v_4, y_34, n_u);

%% restrict unit 3 based on unit 4 result 
A_34bar = dictionary([1,2],{A_34, A4}); 
B_34bar = dictionary([1,2],{[B_314, B_324], B4});
v_34bar = dictionary([1,2],{v_3f, v_4f});
[v_3_fb, u_3_fb, g_3_fb, y_34_fb] = backward_relax(A_34bar, B_34bar, v_34bar, u_3f, g_3f, n_u, 3); % coupling to unit 4


%% unit 5 forward relax
y_35f = [A_35, B_315,B_325]* g_3_fb;
[v_5f, u_5f, g_5f, y_5f] = forward_relax(A5, B5, K_v_5, y_35f, n_u);

%% plot unit outputs
% figure;
% sgtitle('forward unit outputs')
% subplot(2, 6, 1)
% plot(y_1f);
% subplot(2, 6, 2);
% plot(y_2f);
% subplot(2, 6, 3);
% plot(y_34);
% subplot(2, 6, 4);
% plot(y_35f);
% subplot(2, 6, 5);
% plot(y_4f);
% subplot(2, 6, 6);
% plot(y_5f);


end