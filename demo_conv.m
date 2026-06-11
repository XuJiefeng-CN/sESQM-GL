%% min_{x in R^n} 0.5*x^T Q x + q^T x + tau * sum_{J in \cal{J}} \|x_J\|_2 
% s.t. c_{i}(x) = Ax^(d-1) + qs*x -  b <= 0
% x in C := { z in R^n : \| z_J \|_2 \le M  for all J in \cal{J} }.


clear; clc;
close all
prob = 'e';

tau = 1;  % regularize parameter
J = 2;    % the length of each group: mod(n, J) = 0

M = 20;   % parameter of the set {cal C}
K = 5000;

switch prob
    case 'a'
        % instance b
        rand('seed', 2029)
        randn('seed', 2029)

        % dimensions
        n = 1000; 
        m = 500;
        p = 100;
        
        % convex, least square f
        type = 1; dbar = 2;        
        
        % generate symthetic random data
        [Q, q, A, qs, b, x0] = genProblem(m, n, p, type, dbar);

        % Decrease rate of mu_k
        rbars = [.3  0.6 .9]; 
        sbars = [3 6];

    case 'e'
        % instance e
        % rand('seed', 2029)
        % randn('seed', 2029)

        % dimensions
        n = 100; 
        m = 150;
        p = 100;
        
        % convex, least square f
        type = 1; dbar = 2;        
        
        % generate symthetic random data
        [Q, q, A, qs, b, x0] = genProblem(m, n, p, type, dbar);

        % Decrease rate of mu_k
        rbars = [.3  0.6 .9]; 
        sbars = [3 6];
end

%% initial point: project to C
x0_G = reshape(x0, J, []);
x0_Gnorm = vecnorm(x0_G, 2, 1);
idx = x0_Gnorm>0; 
x0_G_new = zeros(J, n/J);
x0_G_new(:, idx) = x0_G(:,idx).*min(M./x0_Gnorm(idx) , 1);
x0 = x0_G_new(:);
if any(vecnorm(x0_G_new, 2, 1)>M+1e-14)
    disp('x0 not in C!')
end


%% print
% recorder
timestamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss');
foldername = '../results';
if ~exist(foldername, 'dir')
    mkdir(foldername);
end
fID = fopen([foldername '/log.txt'], 'a');

fprintf(fID, 'Convex settings: (p,n,m) = (%3d, %3d,%3d), runing at %10s \n', p, n, m, char(timestamp));
fprintf(fID, '-------------------------------------------------------------------------------------------\n');
fprintf(fID, '(%5s, %5s)   %8s   %9s   %8s   %5s   %7s   %4s   %8s   %8s   %8s   %8s    %8s\n', ...
             'rbar', 'sbar', 'f', 'g', 'time', 'iter', 'mean ls', 'mu0', 'mu', 'gamma0', 'gamma', 'L_mu', 'status');
fprintf(fID, '-------------------------------------------------------------------------------------------\n');


% run sESQM
t = 0;
for ii = 1:length(sbars)
    sbar = sbars(ii);
    for jj = 1:length(rbars)
        %% parameters
        t = t+1;
        rbar = rbars(jj);
        fprintf('(rbar,sbar) = (%5.2g, %2d) ...\n', rbar, sbar)
        Mths{t} = sprintf('(%.2g,%1d)', rbar, sbar);
        opt = [];

        opt.x0 = x0;
        opt.type = type; % type of f
        
        % mu_k
        opt.K = K;
        opt.mu0 = 10^6;
        opt.gamma0 = 10;
        opt.rbar = rbar;
        opt.sbar = sbar; 
    
        % termination
        opt.maxIter = opt.K;
        opt.epsilon = 1e-5;

        opt.isdisp = 1; % display
    
        % run sESQM for Group Lasso
        tic_temp = tic;
        [x{t}, psi_val{t}, gval{t}, out{t}] = sESQM_GL_sReLU(Q, q, tau, J, A, qs, b, M, opt);	
        time(t) = toc(tic_temp);
    
        % print: log.txt
        fprintf(fID, '(%5.2g, %2d) | %10.5f | %9.2e | %8.5f | %5d | %7.2f | %4.2g | %8.2e | %8.2e | %8.2e | %8.2e | %6d\n', ...
            rbar, sbar, psi_val{t}, gval{t}, time(t), ...
            out{t}.k, out{t}.iter_ls, out{t}.mu0, out{t}.mu, out{t}.gamma0, out{t}.gamma, out{t}.Lmu, out{t}.status);

    
    end
end
tt = t;
fprintf(fID, '-------------------------------------------------------------------------------------------\n');


%% cvx solver
QQ = cell(m, 1);
qq = cell(m, 1);
As = cell(m, 1);
for ii = 1:m
    QQ{ii} = squeeze(A(ii, :, :));
    qq{ii} = qs(ii, :)';
    As{ii} = squeeze(A(ii, :, :));
end

fprintf('cvx...\n')
fprintf('-------------------------------------------------------------------------------------------\n');
st = tic;
cvx_solver SDPT3
cvx_begin
    variable x_cvx(n)
    
    % Group lasso term
    P1 = 0;
    for ii = 1:(n/J)
        P1 = P1 + tau * norm(x_cvx(1+(ii-1)*J:ii*J), 2);
    end
    % P1 = tau*sum(vecnorm(reshape(x_cvx, J, n/J), 2, 1))

    minimize( .5* sum_square(Q* x_cvx - q) - 0.5*(q'*q)  + P1 )
    subject to
    for i=1:m
        quad_form(x_cvx, As{i}) + qs(i, :) * x_cvx - b(i) <=0;
    end
    for ii = 1:n/J
        norm(x_cvx(1+(ii-1)*J: ii*J), 2) <= M;
    end
    % vecnorm(reshape(x_cvx, J, n/J), 2, 1) <= M;
cvx_end
time_cvx = toc(st);

%%
Ax_cvx = reshape(A, [], n)*x_cvx;
g_cvx = max((reshape(Ax_cvx, [], n)+qs)*x_cvx -b);
cvx_G_norm = norm(x_cvx(1+floor(n/J)*J: end), 2);
g_cvx = max(g_cvx,  cvx_G_norm- M);
for ii = 1:floor(n/J)
    cvx_G_norm = norm(x_cvx(1+(ii-1)*J: ii*J), 2) - M;
    g_cvx = max(g_cvx,  cvx_G_norm- M);
end
fprintf(fID, '%4s %3s | %10.7f | %9.2e | %8.5f\n', ...
    '', 'cvx', cvx_optval, g_cvx, time_cvx);
fprintf(fID, '-------------------------------------------------------------------------------------------\n\n');


%% print
fprintf('----------------------------------------------------------------------------------------\n');
fprintf('%6s   %8s   %9s   %8s   %5s   %7s   %4s   %8s   %8s   %8s   %8s\n', ...
             '(rbar, sbar)', 'f', 'g', 'time', 'iter', 'mean ls', 'mu0', 'mu', 'gamma', 'L_mu', 'status');
fprintf('----------------------------------------------------------------------------------------\n');
for t=1:tt
    psit = f_fun(Q, q, x{t}, type) + P1_fun(x{t}, J, tau);
    Axt = reshape(A, [], n)*x{t};
    for ii=1:dbar-3
        Axt = reshape(Axt, [], n)*x{t};
    end    
    cxt = (reshape(Axt, [], n)+qs)*x{t} -b;
    gt = max(cxt);

    fprintf('%6s | %10.5f | %9.2e | %8.5f | %5d | %7.2f | %4.2g | %8.2e | %8.2e | %8.2e | %3d\n', ...
        Mths{t}, psit, gt, time(t), ...
        out{t}.k, out{t}.iter_ls, out{t}.mu0, out{t}.mu, out{t}.gamma, out{t}.Lmu, out{t}.status);
end
fprintf('%7s | %10.5f | %9.2e | %8.5f\n', ...
     'cvx', cvx_optval, g_cvx, time_cvx);
fprintf('-------------------------------------------------------------------------------------------\n\n');

%% print table
fID_table = fopen([foldername '/' prob, '-table.txt'], 'w');
t = 0;
fprintf(fID_table, '\\begin{tabular}{cccc} \n');
fprintf(fID_table, '\\multicolumn{4}{c}{{(%s)} Convex setting}\\\\ \n', prob);
fprintf(fID_table, '\\hline \n');
fprintf(fID_table, '$(\\bar r, \\bar s)$ & obj. & constr. & \\text{time} \\\\\n');
fprintf(fID_table, '\\hline \n');
for ii = 1:length(sbars)
    for jj = 1:length(rbars)
        t = t + 1;
        if gval{t}<0
            constr = '0';
        else
            constr = sprintf('%9.2e', max(0, gval{t}));
        end

        fprintf(fID_table, '$(%4.2g, %1d)$ & %12.3f & %s & %7.1f\\\\\n',...
                rbars(jj), sbars(ii), psi_val{t}, constr, time(t));
    end
end
if g_cvx<0
    constr = '0';
else
    constr = sprintf('%9.2e', max(0, g_cvx));
end
fprintf(fID_table, '%10s  & %12.3f & %s & %7.1f\\\\ \n', 'CVX', cvx_optval, constr, time_cvx);
fprintf(fID_table, '\\hline \n');
fprintf(fID_table, '\\end{tabular}');
fclose(fID_table);

%% plot objective
close all
figure;
hold on
pos = [1 1 2.9 2.6];
set(gcf,'Units','Inches');
set(gcf,'Position',pos);
% line_style = {'--k', ':r', '-b','oc', '+m', '^g'};

line_style = {'-', '--', ':', 'o', '^', 'x'};

colors = {'r', 'b', 'c', 'g', [0.8500 0.3250 0.0980], 'm'};

for t = 1:tt
    temp = abs(out{t}.psi_vals -cvx_optval)/max(1,abs(cvx_optval));
    temp = log10(temp);
    index = [0:200:out{t}.k+1];
    
    if index(end)~=out{t}.k+1
        index(end+1) = out{t}.k+1;
    end
    temp = temp(index+1);

    plot(index, temp, line_style{t}, 'Color', colors{t});
    ytickformat('10^{%.2g}')
    xlim([0, opt.K])
end
legend(Mths, 'Location','southwest', 'FontSize', 8);
title(sprintf('\\ \\ {(%s)} Convex setting', prob), 'Interpreter','latex')
xlabel('iteration $k$','Interpreter','latex')
xticks([0 1000:2000:opt.K])

ylabel('$ \Delta_{k} $', 'Interpreter','latex')
exportgraphics(gcf, [foldername '/' prob, '-psi.pdf'], 'ContentType', 'vector');

%% plot constraint
figure;
hold on
pos = [1 1 2.9 2.6];
set(gcf,'Units','Inches');
set(gcf,'Position',pos);
min_pos = inf;
for t = 1:tt
    ispos = out{t}.gs>0;
    min_pos = min(min_pos, min(out{t}.gs(ispos)));
end
log0 = floor(log10(min_pos))-1;

for t = 1:tt
    % yyaxis left
    ylabel('$ g^{+}(x^{k})/{\rm max}\{1, \|x^{k}\|\} $', 'Interpreter','latex')
    temp = max(10^log0, out{t}.gs);
    temp = log10(temp);
    index0 = find(temp > log0, 1);
    if isempty(index0)
        index = [0:350:out{t}.k+1];
    else
        index = [0:10:index0 index0:350:out{t}.k+1];
    end
    % index = [0:10:out{t}.k+1];
    if index(end)~=out{t}.k+1
        index(end+1) = out{t}.k+1;
    end
    temp = temp(index+1);
    plot(index', temp, line_style{t}, 'Color', colors{t});
end

title(sprintf('{(%s)} Convex setting', prob), 'Interpreter','latex');
lgd = legend(Mths, 'Location','Southwest', 'FontSize', 8);
xlabel('iteration $k$','Interpreter','latex')
xlim([0, opt.K])
ytickformat('10^{%.2g}')
a = [min(yticks), max(yticks)];
a(1) = log0;
set(gca, 'YTick', a(1):floor((a(2) - a(1))/4):a(2));
a = yticklabels;
a{1} = '0';
set(gca, 'YTickLabel', a);

% yticks(floor(mintick):2:ceil(maxtick))
xticks([0 1000:2000:opt.K])
exportgraphics(gcf, [foldername '/' prob, '-g.pdf'], 'ContentType', 'vector');
