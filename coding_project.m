%% Coding project
clc
clear all
%% data generation
rng('default')
d = 20; %10 (10,100,1000,2000)
n = 50; % (50,500,5000) 50
% beta ~ Unif({-1,1}^d)
beta = sign(randn(d, 1));
beta0 = 0;
% xi_i ~ ber(0.9)
xi = binornd(1,ones(n,1)*0.9);
% x_i ~ N(0_d,I_d)
X = randn(n,d);
% Y = (2xi-1).sign(x^T beta)
y = (2*xi - ones(n,1)).*sign(X*beta);
%% Define objective function

L = (norm([ones(n, 1) X]).^2);
stepsize = 0.9*2/L;

f = @(theta) theta(1) * ones(n, 1) + X * theta(2:end);

foo = @(f) sum(log(1 + exp(-y .* f)));

grad = @(theta) sum([ones(n, 1) X]' * diag(-y ./ (1 + exp(y .* f(theta)))), 2);

maxiter = 200;

iter = 0;
tol = 1E-8;

%% plain gradient descent

xcur = zeros(d + 1, 1);
xcur_acc_prev = zeros(d + 1, 1);
xcur_acc = zeros(d + 1, 1);

iterates = zeros(maxiter, d + 1);
function_values = zeros(maxiter, 1);

tol = 1E-8;
iter = 0;

tStart = cputime;
while iter < maxiter && norm(grad(xcur)) > tol 
   
    xcur = xcur - stepsize * grad(xcur);
    iter = iter + 1;
    iterates(iter,:) = xcur;
    function_values(iter) = foo(f(xcur));
    
end
tEnd_GD = cputime - tStart;
iterates = iterates(1:iter, :);
function_values = function_values(1:iter, :);
iter_GD = iter;
%% Nesterov's accelerated gradient descent

iterates_acc = zeros(maxiter, d + 1);

xcur_acc = zeros(d + 1, 1);
xprev_acc= zeros(d + 1, 1);
function_values_acc = zeros(maxiter, 1);

tol = 1E-8;
iter = 0;
tStart = cputime;
while iter < maxiter && norm(grad(xcur_acc)) > tol 
   
    z = xcur_acc + ((iter - 1) / (iter + 2)) * (xcur_acc - xprev_acc);
    xprev_acc = xcur_acc;
    xcur_acc = z - stepsize * grad(z);
    
    iter = iter + 1;
    iterates_acc(iter,:) = xcur_acc;
    function_values_acc(iter) = foo(f(xcur_acc));
    
end
tEnd_acc = cputime - tStart;
iterates_acc = iterates_acc(1:iter, :);
function_values_acc = function_values_acc(1:iter, :);

iter_acc = iter;
%% Newton Method
hess = @(theta) [ones(n, 1)  X]'  * (repmat(exp(-y .* f(theta)) ./ ((1 + exp(-y .* f(theta))).^2), [1 (d+1)]) .* [ones(n, 1)  X]);

maxiter = 200;
iterates_newt = zeros(maxiter, d + 1); % iterates from accelerated gradient method
function_values_newt = zeros(maxiter, 1); % function values from accelerated gradient method

xcur_newt = zeros(d + 1, 1);

tau = 0.5;
gamma = 0.8;
tol = 1E-8;
iter = 0;
foo_xcur = foo(f(xcur_newt));
tStart = cputime;
while iter < maxiter && norm(grad(xcur_newt)) > tol 
   
    gr = grad(xcur_newt);
    H =  hess(xcur_newt);
    dir = H \ gr;
    
    % select step-size via Armijo rule
    
    m = 0;
    flag = false;
    while foo_xcur - foo(f(xcur_newt - gamma^m * dir)) < tau * gamma^m * dot(-dir, gr)  
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
    iterates(iter,:) = xcur_newt;
    foo_xcur = foo(f(xcur_newt));
    function_values_newt(iter) = foo_xcur;
    
end
tEnd_newt = cputime - tStart;

iterates = iterates(1:iter, :);
function_values_newt = function_values_newt(1:iter, :);
iter_newt = iter;
%% BB method

iterates_BB = zeros(maxiter, d+1);
function_values_BB = zeros(maxiter, 1);
xcur_BB = zeros(d+1,1);

iter = 0;

tStart = cputime;
while iter < maxiter && norm(grad(xcur_BB)) > tol 
   
    if iter >= 1
        gradcur = grad(xcur_BB);
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
        gradprev = grad(xcur_BB);
        xprev = xcur_BB;
        xcur_BB = xcur_BB - stepsize * gradprev; % usual 2/L step-size in first round
    end
    
    iter = iter + 1;
    iterates_BB(iter,:) = xcur_BB;
    function_values_BB(iter) = foo(f(xcur_BB));
    
end
tEnd_BB = cputime - tStart;
iterates_BB = iterates_BB(1:iter, :);
function_values_BB = function_values_BB(1:iter, :);
iter_BB = iter;

%% iter compare
iter_all = [iter_GD; iter_acc; iter_newt; iter_BB];
%% CPU time
time_all = [tEnd_GD; tEnd_acc; tEnd_newt; tEnd_BB];

%% plot
figure
hold on
plot(function_values(1:size(function_values, 1)), '-*','Color','#0072BD','MarkerSize',10)
plot(function_values_acc(1:size(function_values_acc, 1)), '-*','Color','#EDB120','MarkerSize',10)
plot(function_values_newt(1:size(function_values_newt,1)),'-*','Color','#7E2F8E','MarkerSize',10)
plot(function_values_BB(1:size(function_values_BB,1)),'-*','Color','#D95319','MarkerSize',10)
legend({'Gradient descent','Nesterov acclerated GD','Newton Method','BB method'}, 'FontSize', 20,'Location','northeast')
xlabel('Iterations')
ylabel('Objective')


%% PCA
% banknote = readtable("banknote.csv");
% 
% banknote1 = banknote(:,1:6);
% 
% n = 200;
% d = 6;
% k = 2;
% 
% Xb = table2array(banknote1);
% 
% Z = zeros(n,k);
% idx = randsample(200,100);
% Z(idx) = ones(100,1);
% Z(setdiff(1:200,idx),2) = ones(100,1);
% Z = Z/norm(Z);
% 
% W = zeros(k,d);
% idx2 = randsample(6,3); 
% W(1,idx2) = ones(1,3);
% W(2,setdiff(1:6,idx2)) = ones(1,3);
% W = W/norm(W);
% 
% iter = 0;
% maxiter = 2000;
% tol = 1E-8;
% 
% iterates = zeros(maxiter, d + 1);
% function_values = zeros(maxiter, 1);
% stepsize = 0.9*2/ (norm([ones(n, 1) Xb]).^2);
% 
% while iter < maxiter && norm(gradW(Z,Xb))+norm(gradZ(Xb,W)) > tol
%     Z = max(Z-stepsize*gradZ(W,Xb),0);
%     W = max(W-stepsize*gradW(Z,Xb),0);
%     iter = iter + 1;
%     function_values(iter) = norm(Z*W-Xb,"fro"); % Frobenius Norm
% end
% 
% iterates = iterates(1:iter, :);
% function_values = function_values(1:iter, :);

