%% min_{x in R^n} 0.5*x^T Q x + q^T x + tau * sum_{J in \cal{J}} \|x_J\|_2 
% s.t. c_{i}(x) = Ax^(d-1) + qs*x -  b <= 0
% x in C := { z in R^n : \| z_J \|_2 \le M  for all J in \cal{J} }.


clear; clc;
close all
prob = 'b';

tau = 1;                % regularize parameter
J = 2;                  % the length of each group
M = 20;                 % parameter of the set {cal C}
K = 5000;

switch prob
    case 'b'
        % instance b
        rand('seed', 2031)
        randn('seed', 2031)

        % dimensions
        n = 180;
        m = 100;
        
        % nonconvex
        type = 2; dbar = 4;        
        
        % generate symthetic random data
        [Q, q, A, qs, b, x0] = genProblem(m, n, '', type, dbar);

        % Decrease rate of mu_k
        rbars = [.3  0.6 .9]; 
        sbars = [3 6];

    case 'e' % random instance
        % instance d
        % rand('seed', 201)
        % randn('seed', 201)

        % dimensions
        n = 60;
        m = 40;
        
        % nonconvex
        type = 2; dbar = 4;        
        
        % generate symthetic random data
        [Q, q, A, qs, b, x0] = genProblem(m, n, '', type, dbar);

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
% Set up a results recorder
timestamp = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss');
foldername = '../results';
if ~exist(foldername, 'dir')
    mkdir(foldername);
end
fID = fopen([foldername '/log.txt'], 'a');

fprintf(fID,'Nonconvex settings: (d,n,m) = (%1d,%3d,%3d), runing at %10s \n', dbar, n, m, char(timestamp));
fprintf(fID, '(%5s, %5s)   %8s   %9s   %8s   %5s   %7s   %4s   %8s   %8s   %8s   %8s\n', ...
             'rbar', 'sbar', 'f', 'g', 'time', 'iter', 'mean ls', 'mu0', 'mu', 'gamma', 'L_mu', 'status');
fprintf(fID, '-------------------------------------------------------------------------------------------\n');
fprintf('(d, n,m) = (%3d,%3d,%3d), runing at %10s \n', dbar, n, m, char(timestamp));
fprintf('-------------------------------------------------------------------------------------------\n');

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
        fprintf(fID, '(%5.2g, %2d) | %10.5f | %9.2e | %8.5f | %5d | %7.2f | %4.2g | %8.2e | %8.2e | %8.2e\n', ...
            rbar, sbar, psi_val{t}, gval{t}, time(t), ...
            out{t}.k, out{t}.iter_ls, out{t}.mu0, out{t}.mu, out{t}.gamma, out{t}.Lmu);

    end
end
fprintf(fID, '-------------------------------------------------------------------------------------------\n');
tt = t;

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
fprintf('-------------------------------------------------------------------------------------------\n\n');

%% print table
fID_table = fopen([foldername '/' prob, '-table.txt'], 'w');
t = 0;
fprintf(fID_table, '\\begin{tabular}{cccc} \n');
fprintf(fID_table, '\\multicolumn{4}{c}{{(%s)} Nonconvex setting}\\\\ \n', prob, n, m);
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

        fprintf(fID_table, '$(%4.2g, %1d)$ & %12.1f & %s & %7.1f\\\\\n',...
                rbars(jj), sbars(ii), psi_val{t}, constr, time(t));
    end
end
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
    ylabel('$ \psi(x^{k}) $', 'Interpreter','latex')
    temp = out{t}.psi_vals;
    % index = [0:30:150 180:250:out{t}.k+1];
    index = [0:400:out{t}.k+1];
    % index = 0:300:out{t}.k+1;
    
    if index(end)~=out{t}.k+1
        index(end+1) = out{t}.k+1;
    end
    temp = temp(index+1);

    plot(index, temp, line_style{t}, 'Color', colors{t});
    xlim([0, opt.K])
end
legend(Mths, 'Location','northeast', 'FontSize', 8);
title(sprintf('\\ \\ \\ \\ {(%s)} Nonconvex setting', prob), 'Interpreter','latex')
xlabel('iteration $k$','Interpreter','latex')
xticks([0 1000:2000:opt.K])
exportgraphics(gca, [foldername '/' prob, '-psi.pdf'], 'ContentType', 'vector');

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
        index = [0:10:index0 index0:500:out{t}.k+1];
    end
    % index = [0:10:out{t}.k+1];
    if index(end)~=out{t}.k+1
        index(end+1) = out{t}.k+1;
    end
    temp = temp(index+1);
    plot(index', temp, line_style{t}, 'Color', colors{t});
end

title(sprintf('{(%s)} Nonconvex setting', prob), 'Interpreter','latex')
lgd = legend(Mths, 'Location','Northeast', 'FontSize', 8);
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