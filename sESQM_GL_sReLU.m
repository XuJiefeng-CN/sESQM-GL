function [x, psi, g, out] = sESQM_GL_sReLU(Q, q, tau, J, A, qs, b, M, opt)
% The implementaion of sESQM 
% & \min_{x} .5 x^T Q x + q^T x + tau * \sum_{J \in \mathcal{J}} \|x_{J}\|_{2} 
% s.t. c_{i}(x) = \sum_{j_l, 2<= l <= d} a_{i, j_2, \ldots, j_d} x_{j_2} \cdots x_{j_d} 
%                                           -  b_i \le 0, i \in [m],
% s.t. x \in \mathcal{C} = \{ z : \| z_{J} \|_{2} \le M for all J\in \mathcal{J} \}.
% Convex setting: J=1, b>0; Q and A_i are PSD

% hmu = h^+ squar 1/(2*mu)\| \|_2^2 + mu/2

if ~isfield(opt, 'mu0')
    opt.mu0 = 10^6;
end

if ~isfield(opt, 'gamma0')
    opt.gamma0 = 10;
end

if opt.isdisp
    st = tic;
    % print
    fprintf(' --------------------------------------------------- sESQM --------------------------------------------------\n');
    fprintf('%5s | %5s | %5s | %10s | %14s | %9s | %9s | %8s | %8s | %8s |%8s | %8s | %8s | %8s\n', ...
            'k', 'ls', 't', 'dif_pen', 'psi', 'g', 'hmu', 'mu', 'gamma', 'L', 'ttime', 'res1', 'res2', 'res3');
    fprintf(' -------------------------------------------------------------------------------------------------------------\n');
end

x = opt.x0;         % initial point
n = length(x);
d = ndims(A);       % order of A
% m = size(A, 1);

% termination parameters
epsilon = opt.epsilon;
rs_num = 3;
ress = zeros(rs_num, 3);

% line search parameters
L_low = 1e-11;
L_up = 1e11;
c1 = .001;   
L = 1;

% smoothing parameters
mu0 = opt.mu0;
mu = mu0;
n0 = 200;
nu0 = 1/(n0 + 1);
r0 = 0.01;
s0 = 0;
sbar = opt.sbar;
rbar = opt.rbar;
alpha3_varphi = 1/8;
K = opt.K;

% penalty parameter: gamma(t) = gamma0 (1+t)^(-p)
p = 0.5;
gamma0 = opt.gamma0;    
gamma = gamma0;
c2 = 1e-3/max(1, gamma0*mu0);
% c2 = 1;
c3 = 2;

% recorders
psi_s = [];         % recording the values of objective f
gs = [];            % g = max_i g_i


% ----------------------------- x0 -------------------------------------- %
% f, df, psi
[f, Qx] = f_fun(Q, q, x, opt.type);
df = df_fun(Q, q, Qx, opt.type);

% P1 
x_G = reshape(x, J, []);
x_Gnorm = vecnorm(x_G, 2, 1);
P1 = tau*sum(x_Gnorm);

psi = f + P1;

% c, Dc, g
Dc = reshape(A, [], n)*x;
for jj = 1:d-3
    Dc = reshape(Dc, [], n)*x;
end
Dc = reshape(Dc, [], n);
cx = (Dc+qs)*x - b; % A(*x)^(d-2) in R^{m x n}
Dc = (d-1)*Dc + qs; % Dc(x)
g = max(cx);

% calculate nabla hmu(cx) = Proj_{partial h(0)}(cx/mu)
d_hmu = simplex_proj(cx/mu, 1);
hmu = max(cx - mu*d_hmu) + mu/2*( d_hmu'*d_hmu + 1);

% gmu(x) = varphi_mu(hmu(cx))
if abs(hmu)>mu/2
    gmu = max(0, hmu); 
else
    gmu = (hmu/2)*(hmu/mu + 1) + mu/8;
end

% Hmu(x) = hmu(c(x))
d_Hmu = (d_hmu'*Dc)';     

% nabla varphi_mu ( hmu)
d_varphi_mu = min(1, max(0, hmu/mu + .5));  

% nabla gmu(x)
d_gmu = d_varphi_mu*d_Hmu;

% recorder
psi_s = [psi_s; psi];
% gs = [gs; g];
gs = [gs; g/max(1, norm(x))];

k = 0;
t = 0;
iter_ls = 0;

if opt.isdisp
    ttime = toc(st);
    % Print
    fprintf('%5d | %5d | %5d | %10.3g |     %10.7g | %9.2e | %9.2e | %8.2e | %8.2e | %8.2e |%8.3g\n', ...
               k,  0,    t,     0,        psi,          g,   hmu,  mu, gamma, L, ttime);
end

while 1
    pen = gamma * psi + gmu;  % exact penalty function
    d_pen = gamma * df + d_gmu; % gradient of smoothing exact penalty function

    % line search
    ls_suc = 0;
    % fprintf('%2s | %8s | %8s | %8s \n', 'ii', 'pen', 'pen1', 'desc')
    for ii = 0:40
        alpha = mu/L;      % stepsize
        z = x - alpha * d_pen; % center of prox
        
        % solve subproblem: prox_{gamma*alpha*tau*Sum_J \| x_J \|_2 + delta_{max \| x_J \|_2 <= M} }( z )
        
        z_G = reshape(z, J, []);
        z_Gnorm = vecnorm(z_G, 2, 1);
        idx = z_Gnorm>0;
        x1_G = zeros(J, n/J);
        x1_Gnorm = min(M, max(0, z_Gnorm - gamma*alpha*tau));
        x1_G(:, idx) = ( z_G(:,idx) ./ z_Gnorm(idx) ).* x1_Gnorm(idx);

        x1 = x1_G(:);
        P1 = tau*sum(x1_Gnorm);

        % f(x1)
        [f, Qx] = f_fun(Q, q, x1, opt.type);

        % psi(x1)
        psi = f + P1;

        % x1: g, c, Ax1^{d-2} 
        Dc = reshape(A, [], n)*x1;
        for jj = 2:d-2
            Dc = reshape(Dc, [], n)*x1;
        end
        Dc = reshape(Dc, [], n);    % A(*x1)^(d-2) in R^{m x n}
        cx = (Dc+qs)*x1 - b;             % c(x1)
        
        % nabla hmu(cx) = Proj_{partial h(0)}(cx/mu)
        d_hmu = simplex_proj(cx/mu, 1);
        hmu = max(cx - mu*d_hmu) + mu/2*( d_hmu'*d_hmu + 1);
        
        % gmu(x) = varphi_mu(hmu(cx))
        if abs(hmu)>mu/2
            gmu_x1 = max(0, hmu); 
        else
            gmu_x1 = (hmu/2)*(hmu/mu + 1) + mu/8;
        end
        
        % check sufficient descent condition
        dif_x = x1 - x;
        norm_diff_x = norm(dif_x);
        pen_hat = gamma*psi + gmu_x1;
        % fprintf('%2d | %8.5e | %8.5e | %8.5e \n', ii, pen, pen_hat, (c1/(2*mu)) * norm_diff_x^2)
        if pen_hat <= pen - (c1/(2*mu)) * norm_diff_x^2 % + 1e-12
            ls_suc = 1;
            break
        end

        L = 2*L;
    end
    g = max(cx);                % g(x1)
    iter_ls = iter_ls + ii;
    
    %% termination conditions
    ktemp = mod(k, rs_num)+1;
    ress(ktemp, 1) = norm_diff_x/( min(1, gamma * mu) );
    % ress(ktemp, 1) = (pen - pen_hat)/gamma;
    
    d_varphi_mu = min(1, max(0, hmu/mu + .5));  
    lambda = d_varphi_mu/gamma;
    ress(ktemp, 2) = max(g, abs(lambda*g));
    ress(ktemp, 3) = g - cx'*d_hmu;
    ress(ktemp, :) = ress(ktemp, :)/max(norm(x1), 1);
    if ~ls_suc || max(ress(:))<=epsilon || k >= opt.maxIter

        if opt.isdisp
            ttime = toc(st);
            % Print
            fprintf('%5d | %5d | %5d | %10.1e | %10.7e | %9.2e | %9.2e | %8.2e | %8.2e | %8.2e |%8.3g | %4.2e | %4.2e | %4.2e\n', ...
                       k+1, ii,  t, pen_hat-pen, psi,      g,      hmu,     mu,     gamma,    L, ttime, max(ress(:, 1)), max(ress(:, 2)), max(ress(:, 3)));
        end
        if ls_suc
            x = x1;
            psi_s = [psi_s; psi];
            % gs = [gs; g];
            gs = [gs; g/max(1, norm(x))];
        else
            out.status = 2;
            k = k-1;
            disp('Fail to line-search!!!')
            break;
        end
        if max(ress(:))<=epsilon
            out.status = 1;
            break
        end
        if k >= opt.maxIter
            out.status = 3;
            break
        end
    end

    %% mark for updating gamma
    if norm_diff_x <= c2*gamma * mu  && hmu > c3 * alpha3_varphi * mu
        t = t + 1;
    end
    
    % update mu
    k1 = mod(k+1, n0+1);           % remainder
    kprod = k+1 - k1;
    kbar = kprod + nu0*k1;
    r = r0 + min(1, kbar/K)*(rbar-r0);     % r(k+1)
    s_hat = s0 + min(1, kbar/K)*(sbar - s0);
    mu = mu0*((kbar + 1).^(-r))./(log(3+kbar).^s_hat);

    % nabla f(x1)
    df1 = df_fun(Q, q, Qx, opt.type);

    % Dc(x1)
    Dc = (d-1)*Dc + qs;
    
    % calculate g_mu1(x1) and nabla g_mu1(x1)
    d_hmu = simplex_proj(cx/mu, 1);
    hmu = max(cx - mu*d_hmu) + mu/2*( d_hmu'*d_hmu + 1);
    if abs(hmu)>mu/2
        gmu = max(0, hmu); 
    else
        gmu = (hmu/2)*(hmu/mu + 1) + mu/8;
    end
    d_varphi_mu = min(1, max(0, hmu/mu + .5));
    d_gmu1_x1 = d_varphi_mu*(d_hmu'*Dc)';                   % nabla g_mu1(x1)
    
    dif_grad = gamma*(df1 - df) + d_gmu1_x1 - d_gmu;
    L = min(L_up, max(L_low, .5*L));
    
    %% calculate BB stepsize: L_BB1
    % if norm_diff_x > 1e-12
    %     Lmu_BB = abs(dif_grad'*dif_x/(norm_diff_x^2));      % BB stepsize of gamma*f + gmu
    %     L_BB = Lmu_BB*mu; 
    %     if L_BB >= L_low && L_BB <= L_up
    %         L = L_BB;
    %     end
    % end

    % calculate BB stepsize: L_BB2
    if sqrt(abs(dif_grad'*dif_x))>1e-12
        L_BB = mu*(dif_grad'*dif_grad)/abs(dif_grad'*dif_x); % BB stepsize of gmu          
        if L_BB >= L_low && L_BB <= L_up
            L = L_BB;
        end
    end
    
    % update
    gamma = gamma0 * (1 + t)^(-p); % update gamma
    x = x1;
    df = df1;
    d_gmu = d_gmu1_x1;

    psi_s = [psi_s; psi];
    % gs = [gs; g];
    gs = [gs; g/max(1, norm(x))];
    
    %% print
    if opt.isdisp && (mod(k, ceil(K/10))==0 || k<=5)
        ttime = toc(st);
        % Print
            fprintf('%5d | %5d | %5d | %10.1e | %10.7e | %9.2e | %9.2e | %8.2e | %8.2e | %8.2e |%8.3g | %4.2e | %4.2e | %4.2e\n', ...
                       k+1, ii,  t, pen_hat-pen, psi,      g,      gmu,     mu,     gamma,    L, ttime, max(ress(:, 1)), max(ress(:, 2)), max(ress(:, 3)));
    end
    k = k + 1;
end



out.mu0 = mu0;
out.mu = mu;
out.Lmu = L/mu;
out.gamma0 = gamma0;
out.gamma = gamma;
out.t = t;
out.iter_ls = iter_ls/k;
out.k = k;
out.psi_vals = psi_s;
out.gs = gs;
end

function d = simplex_proj(c,tau)
n = max(size(c));
p = -c;
pmax = max(p);
sm = sum(p);
if sm >= n*pmax - tau
  lambda = (tau+sm)/n;
  d = max(0, c + lambda);
  clear p;
  return;
end 

p = sort(p);

sm = 0;
for i = 1:n-1
  smnew = sm + i*(p(i+1) - p(i));
  if smnew >= tau
    break
  end
  sm = smnew;
end

k = i;
delta = (tau - sm)/k;
lambda = p(k) + delta;
d = max(0, c + lambda);
clear p;
end