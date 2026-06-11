function [Q, q, A, qs, b, x0] = genProblem(m, n, p, type, dbar)

sp = .1;
% b = ones(m, 1);
b = .1 + .9*rand(m, 1);
switch type
    case 1 % convex QCQP
        % f(x) = \| Q x - q \|^2
        Q = randn(p, n);
        Q = Q./vecnorm(Q, 2, 1);
        q = 10 + randn(p, 1);
        
        % initial point
        [U, S, V] = svd(Q, 'econ');
        x0 = V*((U'*q)./diag(S)); 

        % constraints
        A = zeros(m, n, n);
        qs = zeros(m, n);
        for i=1:m
            [U, ~] = qr(randn(n));
            Lambda = 100*sprand(n, 1, sp);
            Atemp =  (U.*Lambda')*U';
            % Add 1e-14 to enhance numerical stability of CVX
            A(i, :, :) = .5*(Atemp + Atemp') + 1e-14*eye(n); 
            qs(i, :) = 10+randn(n, 1);
        end
    
    case 2 % nonconvex homogeniuos constraint
        % orthogonal matrix
        [U, ~] = qr(randn(n));
        Lambda = 20*(2*rand(n, 1) - 1);
        Q = (U.*Lambda')*U'; % n x n
        Q = .5*(Q + Q'); 
        q = 10 + randn(n, 1);
        
        % initial point
        x0 = -(Q\q);

        A = 2*rand([m,n*ones(1,dbar-1)])-1; 
        A = sym_array(A, 2:dbar);      % partial symetrize
        qs = zeros(m, n);
end


end