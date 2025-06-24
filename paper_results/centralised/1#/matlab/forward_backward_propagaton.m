function [g_1_fb,g_2_fb,g_3_fb_p,g_4f,g_5f] = forward_backward_propagaton(param_dict,KS)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
n_x = 2;
n_u = 1; %%
[A1, B1] = param_dict{1}{:};
[A2, B2] = param_dict{2}{:};
[A_34, B_314, B_324] = param_dict{34}{:};
[A_35, B_315, B_325] = param_dict{35}{:};
[A4, B4] = param_dict{4}{:};
[A5, B5] = param_dict{5}{:};

g_1f = KS{1};
g_2f = KS{2};
g_3f = KS{3}; 
g_4f = KS{4};
g_5f = KS{5};

v_1f = projection(g_1f, 1:n_x) ;
v_2f = projection(g_2f, 1:n_x) ;
v_3f = projection(g_3f, 1:n_x) ;
v_4f = projection(g_4f, 1:n_x) ;
v_5f = projection(g_5f, 1:n_x) ;

u_1f = [] ;
u_2f = [] ;
u_3f = projection(g_3f, n_x+1:n_x+2*n_u) ;
u_4f = projection(g_4f, n_x+1:n_x+n_u) ;
u_5f = projection(g_5f, n_x+1:n_x+n_u) ;

%% unit 5
[v_5f, u_5f, g_5f] = constraint_fn([A5, B5], zeros(1), g_5f, []);
v_5f = projection(g_5f, 1:n_x) ;
u_5f = projection(g_5f, n_x+1:n_x+n_u) ;
%% unit 4
[v_4f, u_4f, g_4f] = constraint_fn([A4, B4], zeros(1), g_4f, []);
v_4f = projection(g_4f, 1:n_x) ;
u_4f = projection(g_4f, n_x+1:n_x+n_u) ;

%% ----- forward - backward relaxation ------------- %%
%% unit 3
[v_3f, u_3f, g_3f] = constraint_fn([A_34, B_314,B_324], zeros(1), g_3f, []);
[v_3f, u_3f, g_3f] = constraint_fn([A_35, B_315, B_325], zeros(1), g_3f, []);
v_3f = projection(g_3f, 1:n_x) ;
u_3f = projection(g_3f, n_x+1:n_x+2*n_u) ;

% coupling to unit 5
A_35bar = dictionary([1,2],{A_35, A5}); 
B_35bar = dictionary([1,2],{[B_315, B_325], B5});
v_35bar = dictionary([1,2],{v_3f, v_5f});
[v_3_fb, u_3_fb, g_3_fb, y_3_fb] = backward_relax(A_35bar, B_35bar, v_35bar, u_3f, g_3f, n_u, 3); % coupling to unit 5

% coupling to unit 4
A_34bar = dictionary([1,2],{A_34, A4}); 
B_34bar = dictionary([1,2],{[B_314, B_324], B4});
v_34bar = dictionary([1,2],{v_3f, v_4f});
[v_3_fb, u_3_fb, g_3_fb, y_3_fb] = backward_relax(A_34bar, B_34bar, v_34bar, u_3f, g_3_fb, n_u, 3); % coupling to unit 5


% % propagate forward to unit 4
% y_34fb = [A_34, B_314, B_325] * g_3_fb;
% 
% %% unit 4 forward-backward relax
% [v_4fb, u_4fb, g_4fb, y_4fb] = forward_relax(A4, B4, v_4f, y_34fb, n_u);

%% --- unit 2 given unit 3 --- % 
[v_2f, u_2f, g_2f] = constraint_fn(A2, B2, g_2f, []);


% coupling based on feasibility of decision vars in unit 5
A_2bar = dictionary([1,2], {A2, A_35}); 
B_2bar = dictionary([1,2,3],{B2, B_325, B_315});
v_2bar = dictionary([1,2],{zeros(n_x), projection(g_3_fb, 1:n_x+n_u)});

[v_2_fb, u_2_fb, g_2_fb, y_2_fb] = forwardbackward_relax(A_2bar, B_2bar, v_2bar, [], g_2f, n_u, 2);


%% plot unit outputs
% figure;
% sgtitle('comparison between restriction of unit 2 outputs and unit 3 inputs (1)')
% subplot(2, 2, 1)
% plot(y_2f);
% subplot(2, 2, 2);
% plot(y_2_fb);
% subplot(2,2, 3);
% plot(projection(u_3_fb, n_u+1:2*n_u));

% coupling based on feasibility of decision vars in unit 4
A_2bar = dictionary([1,2], {A2, A_34}); 
B_2bar = dictionary([1,2,3],{B2, B_324, B_314});
v_2bar = dictionary([1,2],{zeros(n_x), projection(g_3_fb, 1:n_x+n_u)});

[v_2_fb, u_2_fb, g_2_fb, y_2_fb] = forwardbackward_relax(A_2bar, B_2bar, v_2bar, [], g_2_fb, n_u, 2);

% %% plot unit outputs
% figure;
% sgtitle('comparison between restriction of unit 2 outputs and unit 3 inputs (2)')
% subplot(2, 2, 1)
% plot(y_2f);
% subplot(2, 2, 2);
% plot(y_2_fb);
% subplot(2,2, 3);
% plot(projection(u_3_fb, n_u+1:2*n_u));

% restrict input of unit 3 based on the new unit 2
[Av1, Hv1] = double(g_3_fb);
[Av2p, Hv2p] = double(y_2_fb);
r = size(Av2p);
Av2 = [zeros(r(1),n_x),zeros(r(1), n_u), Av2p];
Hv2 = Hv2p;

g_3_fb_p = g_3_fb ;% polytope([Av1;Av2], [Hv1;Hv2]);


%% --- unit 1 given unit 2 --- %
[v_1f, u_1f, g_1f] = constraint_fn(A1, B1, g_1f, []);
% coupling based on feasibility of decision vars in unit 5
A_1bar = dictionary([1,2], {A1, A_35}); 
B_1bar = dictionary([1,2,3],{B1, B_315, B_325});
v_1bar = dictionary([1,2],{zeros(n_x), projection(g_3_fb_p, [1:n_x,n_x+n_u+1:n_x+2*n_u])});

[v_1_fb, u_1_fb, g_1_fb, y_1_fb] = forwardbackward_relax(A_1bar, B_1bar, v_1bar, [], g_1f, n_u, 1);

%% plot unit outputs
% figure;
% sgtitle('comparison between restriction of unit 1 outputs and unit 3 inputs (1)')
% subplot(2, 2, 1)
% plot(y_1f);
% subplot(2, 2, 2);
% plot(y_1_fb);
% subplot(2,2, 3);
% plot(projection(u_3_fb, 1:n_u));

% coupling based on feasibility of decision vars in unit 4
A_1bar = dictionary([1,2], {A1, A_34}); 
B_1bar = dictionary([1,2,3],{B1, B_314, B_324});
v_1bar = dictionary([1,2],{zeros(n_x), projection(g_3_fb_p, [1:n_x,n_x+n_u+1:n_x+2*n_u])});

[v_1_fb, u_1_fb, g_1_fb, y_1_fb] = forwardbackward_relax(A_1bar, B_1bar, v_1bar, [], g_1_fb, n_u, 1);
end