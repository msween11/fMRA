%%% SETUP %%% --------------------------------------------------------

seed = 1:20;
nrange = 10:.5:20;

avg_table = 0;
tables = {0};

h_opt = zeros(1,length(nrange));
l8_errs_hfix_space = zeros(1,length(nrange));
l8_errs_hopt_space = zeros(1,length(nrange));
l8_errs_freq = zeros(1,length(nrange));

N = 1; l = 5; B = 10;
lambda = 0.1; sigma = 1; hfix = 0.035; r = 0.01; 
zero_cond = 0; 
signal = "f1"; shifts = "unif"; thresh = 0;


for s = seed
for q = nrange

n = floor(2^q);
idx  = find(nrange == q);

[~, ~, ~, l8_errs_freq(idx),l8_errs_hfix_space(idx), l8_errs_hopt_space(idx), ~, h_opt(idx)] =...
    FUNC_singlerun(s, N,l,B,lambda,sigma,n, hfix, r, zero_cond,signal, shifts, thresh);
idx
%end of recovery loop
end

%% data saving
nn = 2.^nrange;
hf = repelem(hfix, 1,length(nrange));
reg = repelem(r, 1, length(nrange));
lam = repelem(lambda, 1, length(nrange));
sig = repelem(sigma, 1, length(nrange));

T = table(nn', lam', sig', h_opt', hf', reg', l8_errs_freq',...
    l8_errs_hopt_space',l8_errs_hfix_space');
T.Properties.VariableNames = ["n","lambda", "sigma", "h_opt", "hfix", "r",...
    "l8_errs_freq", "l8_errs_hopt_space", "l8_errs_hfix_space"];
avg_table = avg_table + T;
tables{s} = T;

%end of rng loop below
s
end
avg_table = avg_table./length(seed);
writetable(avg_table, strjoin(['mu_' num2str(length(seed)) 'avg_' shifts '_' signal '.txt'], ''));

 %%
errs = 0;
for s = seed
   errs = (tables{s} - avg_table).^2;
end

errs = 1/(length(seed)-1).*errs;
errs = sqrt(errs);

writetable(errs, strjoin(['s_' num2str(length(seed)) 'avg_' shifts '_' signal '.txt'], ''));



