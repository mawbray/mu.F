function [A_v,b_v] = construct_v(A_1, B_1, A_2, B_2, A_34, A_35, B_314, B_324, B_315, B_325, A_4, B_4, A_5, B_5)
% lower diagonal matrix defining the joint polytope
%  
[n_u, n_x] = size(A_1);
A_v = zeros(6*n_u, 5*n_x); % nu is the dimensionality of the inputs, nx is the dimensionality of the parameter
block1a4 = B_314*A_1;
block1b4 = B_324*A_2;
block2a4 = B_4*block1a4;
block2b4 = B_4*block1b4;
block1a5 = B_315*A_1;
block1b5 = B_325*A_2;
block2a5 = B_5*block1a5;
block2b5 = B_5*block1b5;
block2c4 = B_4*A_34;
block2c5 = B_5*A_35;


% allocation
for i = 1:n_u
    for j = 1:n_x
        A_v(i,j) = A_1(i,j);
        A_v(n_u+i,n_x+j) = A_2(i,j);
        A_v(2*n_u+i,2*n_x+j) = A_34(i,j);
        A_v(3*n_u+i,2*n_x+j) = A_35(i,j);
        A_v(4*n_u+i,3*n_x+j) = A_4(i,j);
        A_v(5*n_u+i,4*n_x+j) = A_5(i,j);
 
        A_v(2*n_u+i,j) = block1a4(i,j);
        A_v(2*n_u+i,n_x+j) = block1b4(i,j);
        A_v(3*n_u+i,j) = block1a5(i,j);
        A_v(3*n_u+i,n_x+j) = block1b5(i,j);
        A_v(4*n_u+i,j) = block2a4(i,j);
        A_v(4*n_u+i,n_x+j) = block2b4(i,j);
        A_v(4*n_u+i,2*n_x+j) = block2c4(i,j);
        A_v(5*n_u+i,j) = block2a5(i,j);
        A_v(5*n_u+i,n_x+j) = block2b5(i,j);
        A_v(5*n_u+i,2*n_x+j) = block2c5(i,j);
    end
end

b_v = cat(1, [B_1; B_2; B_314*B_1 + B_324*B_2; B_315*B_1 + B_325*B_2; B_4*B_314*B_1 + B_4*B_324*B_2; B_5*B_315*B_1 + B_5*B_325*B_2 ]);

end