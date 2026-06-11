function P1_val = P1_fun(x, J, tau)
n = length(x);
nr = mod(n, J);
n_bar = n - nr;
if J==1 % lasso
    P1_val = tau*sum(abs(x));
else
    % P1: Group lasso
    if nr==0 % all groups have the same length J
        x_G = reshape(x, J, []);
        x_Gnorm = vecnorm(x_G, 2, 1);
        P1_val = tau*sum(x_Gnorm);
    else % the last group has length in (J, 2J)
        x_G1 = reshape(x(1:n_bar), J, []);
        x_G2 =  x(n_bar+1:end);
        x_Gnorm1 = vecnorm(x_G1, 2, 1);
        x_Gnorm2 = vecnorm(x_G2, 2, 1);
        P1_val = tau*( sum(x_Gnorm1) + x_Gnorm2 );
    end
end
end