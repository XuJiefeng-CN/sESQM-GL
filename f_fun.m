function [f, Qx] = f_fun(Q, q, x, isLS)
% calculate the function value of f
% isLS: is least square?
if isLS==1 % case 1: f(x) = .5*( \| Q*x - q\|^2  - \|q\|^2)
    Qx = Q*x-q;
    f = .5*(Qx'*Qx) - .5*(q'*q);
else % case 2: f(x) = .5*x'*Q*x + q'*x
    Qx = Q*x;
    f = x'*(.5*Qx + q);
end
end