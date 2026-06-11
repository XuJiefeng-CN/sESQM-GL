function df = df_fun(Q, q, Qx, isLS)
% calculate the gradient of f
% isLS: is least square?
if isLS==1 % case 1: f(x) = .5*\| Q*x - q\|^2
    % Qx = Q*x - q
    df = Q'*Qx; 
else % case 2: f(x) = .5*x'*Q*x + q'*x
    % Qx = Q*x
    df = Qx + q;
end
end