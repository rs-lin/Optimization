clear;
%% HW1 

data = importdata('GMM.txt');

% L = (norm([ones(50, 1) data]).^2);
% stepsize = 0.9*2/L;


maxiter = 500;
iterates = zeros(maxiter, 2);
iterates_GD = zeros(maxiter,2);
iterates_BB = zeros(maxiter,2);
iterates_newt = zeros(maxiter,2);
function_values = zeros(maxiter, 1);
function_values_GD = zeros(maxiter,1);
function_values_newt = zeros(maxiter,1);
function_values_BB = zeros(maxiter,1);

% 2(c) starting value
xcur = [1;-1];
xcur_GD = [1;-1];
xcur_BB = [1;-1];
xcur_newt = [1;-1];
% 2(d) starting value
% xcur = [-0.7258;1.3732];
% xcur = [1.3732;-0.7258];
tol = 1E-6;
iter = 0;
gmm_xcur = gmm(data,xcur_newt(1),xcur_newt(2));
tau = 0.5;
gamma = 0.8;
tol = 1E-8;
%% Gradient Descent with backtracking line search

% while iter < maxiter && norm(gradf(data,xcur(1),xcur(2))) > tol 
%     gr = gradf(data,xcur(1),xcur(2));
%     H =  hessian_gmm(data,xcur(1),xcur(2));
%     dir = H \ gr;
% 
%     m = 0;
%     flag = false;
%     new_dir = xcur - gamma^m * dir;
%     while gmm_xcur - gmm(data,new_dir(1),new_dir(2)) < tau * gamma^m * dot(-dir, gr)  
%         m = m+1;
%         if gamma^m < tol^2
%             flag = true;
%            break;
%         end
%     end
%     if  flag
%         break; 
%     end
%     
%     
%     xcur = xcur - gamma^m * dir;
%     iter = iter + 1;
%     iterates(iter,:) = xcur;
%     function_values(iter) = gmm(data,xcur(1),xcur(2));
%     
% end
% 
% iterates = iterates(1:iter, :);
% function_values = function_values(1:iter, :);
% 
% iter_GDarmijo = iter;

%% Plain gradient descent
iter = 0;
stepsize = 0.02; 
tStart = cputime;
while iter < maxiter && norm(gradf(data,xcur_GD(1),xcur_GD(2))) > tol 
    xcur_GD = xcur_GD - stepsize * gradf(data,xcur_GD(1),xcur_GD(2));
    iter = iter + 1;
    iterates_GD(iter,:) = xcur_GD;
    function_values_GD(iter) = gmm(data,xcur_GD(1),xcur_GD(2));
end
tEnd_GD = cputime - tStart;
iterates_GD = iterates_GD(1:iter, :);
function_values_GD = function_values_GD(1:iter, :);

iter_GD = iter;

%% Nesterov's
iter = 0;
iterates_acc = zeros(maxiter, 2);
function_values_acc = zeros(maxiter, 1);
xprev_acc= zeros(maxiter, 1);
xcur_acc = [1;-1];
xcur_acc_prev = [1;-1];


tol = 1E-8;
iter = 0;

tStart = cputime;
while iter < maxiter && norm(gradf(data,xcur_acc(1),xcur_acc(2))) > tol 
   
    z = xcur_acc + ((iter - 1) / (iter + 2)) * (xcur_acc - xcur_acc_prev);
    xcur_acc_prev = xcur_acc;
    xcur_acc = z - stepsize * gradf(data,xcur_acc(1),xcur_acc(2));
    
    iter = iter + 1;
    iterates_acc(iter,:) = xcur_acc;
    function_values_acc(iter) = gmm(data,xcur_acc(1),xcur_acc(2));
    
end
tEnd_acc = cputime - tStart;
iterates_acc = iterates_acc(1:iter, :);
function_values_acc = function_values_acc(1:iter, :);

iter_acc = iter;
%% Newton Method
iter = 0;
tau = 0.5;
gamma = 0.8;
tol = 1E-8;

gmm_xcur = gmm(data,xcur_newt(1),xcur_newt(2));
tStart = cputime;
while iter < maxiter && norm(gradf(data,xcur_newt(1),xcur_newt(2))) > tol 
   
    gr = gradf(data,xcur_newt(1),xcur_newt(2));
    H =  hessian_gmm(data,xcur_newt(1),xcur_newt(2));
    dir = H \ gr;
    
    % select step-size via Armijo rule
    
    m = 0;
    flag = false;
    new_dir = xcur_newt - gamma^m * dir;
    while gmm_xcur - gmm(data,new_dir(1),new_dir(2)) < tau * gamma^m * dot(-dir, gr)  
        m = m+1;
        if gamma^m < tol^2
            flag = true;
           break;
        end
    end
    if  flag
        break; 
    end
    
    
    
    xcur_newt = xcur_newt - gamma^m * dir;
    iter = iter + 1;
    iterates_newt(iter,:) = xcur_newt;
    gmm_xcur = gmm(data,xcur_newt(1),xcur_newt(2));
    function_values_newt(iter) = gmm_xcur;
    
end
tEnd_newt = cputime - tStart;

iterates_newt = iterates_newt(1:iter, :);
function_values_newt = function_values_newt(1:iter, :);

iter_newt = iter;


%% BB method

iter = 0;
tStart = cputime;
while iter < maxiter && norm(gradf(data,xcur_BB(1),xcur_BB(2))) > tol 
   
    if iter >= 1
        gradcur = gradf(data,xcur_BB(1),xcur_BB(2));
        if mod(iter, 2) == 0
            scaling = sum((xcur_BB - xprev).^2)/dot(xcur_BB - xprev, gradcur - gradprev);
        else
            scaling = dot(xcur_BB - xprev, gradcur - gradprev) / sum((gradcur - gradprev).^2);
        end
        
        %scaling
        
        xprev = xcur_BB;
        gradprev = gradcur;
        xcur_BB = xcur_BB - scaling * gradcur;
        
    else
        gradprev = gradf(data,xcur_BB(1),xcur_BB(2));
        xprev = xcur_BB;
        xcur_BB = xcur_BB - stepsize * gradprev; % usual 2/L step-size in first round
    end
    
   
    iter = iter + 1;
    iterates_BB(iter,:) = xcur_BB;
    function_values_BB(iter) = gmm(data,xcur_BB(1),xcur_BB(2));
    
end
tEnd_BB = cputime - tStart;
iterates_BB = iterates_BB(1:iter, :);
function_values_BB = function_values_BB(1:iter, :);
iter_BB = iter;

%% iterations
iter_all = [iter_GD; iter_acc; iter_newt; iter_BB];
%% CPU time
time_all = [tEnd_GD; tEnd_acc; tEnd_newt; tEnd_BB];
%% Plot
xgrid = -linspace(-2,2,50);
ygrid = xgrid;
z = zeros(numel(xgrid), numel(ygrid));

for i=1:numel(xgrid)
    for j=1:numel(ygrid)
    z(i,j) = gmm(data,xgrid(i),ygrid(j));
    end
end

figure
hold on
contour(xgrid, ygrid, z', 50)
colorbar
plot([0;iterates_GD(:,1)], [0;iterates_GD(:,2)], '-*','Color','#77AC30','MarkerSize',5)
%plot([0;iterates(:,1)], [0;iterates(:,2)], '-*','Color','#0072BD')
plot([0;iterates_acc(:,1)], [0;iterates_acc(:,2)], '-*','Color','#EDB120','MarkerSize',5)
plot([0;iterates_newt(:,1)], [0;iterates_newt(:,2)], '-*','Color','#7E2F8E','MarkerSize',10)
plot([0;iterates_BB(:,1)],[0;iterates_BB(:,2)],'-*','Color','#D95319','MarkerSize',10)
legend({'contour', 'Gradient descent','Nesterov acclerated GD','Newton Method','BB method'}, 'FontSize', 20,'Location','northeast')



figure
hold on
plot(function_values_GD(1:min(size(function_values_GD, 1),100)), '-*','Color','#77AC30','MarkerSize',10)
%plot(function_values(1:min(size(function_values, 1),100)), '-*','Color','#0072BD','MarkerSize',10)
plot(function_values_acc(1:min(size(function_values_acc, 1),100)), '-*','Color','#EDB120','MarkerSize',10)
plot(function_values_newt(1:min(size(function_values_newt,1),100)),'-*','Color','#7E2F8E','MarkerSize',10)
plot(function_values_BB(1:min(size(function_values_BB,1),100)),'-*','Color','#D95319','MarkerSize',10)
legend({'Gradient descent','Nesterov acclerated GD','Newton Method','BB method'}, 'FontSize', 20,'Location','northeast')
xlabel('Iterations')
ylabel('Objective')