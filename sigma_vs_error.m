%%% SETUP %%% --------------------------------------------------------

seed = 1:20;
sigma_range = -1:.1:1;

avg_table = 0;
tables = {0};

h_opt = zeros(1,length(sigma_range));
l8_errs_hfix_space = zeros(1,length(sigma_range));
l8_errs_hopt_space = zeros(1,length(sigma_range));
l8_errs_freq = zeros(1,length(sigma_range));

N = 1; l = 5; B = 10;
lambda = 0.1; hfix = 0.035; r = 0.0001; 
zero_cond = 1; 
signal = "f4"; shifts = "unif"; thresh = 0.001;
nl = 1000; 


for s = seed

parfor q = 1:length(sigma_range)
sigma = 2^sigma_range(q);
n = nl*sigma^4;
[~, ~, ~, l8_errs_freq(q),l8_errs_hfix_space(q), l8_errs_hopt_space(q), ~, h_opt(q)] =...
    FUNC_singlerun(s, N,l,B,lambda,sigma,n, hfix, r, zero_cond,signal, shifts, thresh);
end

%% data saving
ss = 2.^sigma_range;
nn = nl*ss.^4;
hf = repelem(hfix, 1,length(sigma_range));
reg = repelem(r, 1, length(sigma_range));
lam = repelem(lambda, 1, length(sigma_range));
T = table(nn', lam', ss', h_opt', hf', reg', l8_errs_freq',...
    l8_errs_hopt_space',l8_errs_hfix_space');
T.Properties.VariableNames = ["n","lambda", "sigma", "h_opt", "hfix", "r",...
    "l8_errs_freq", "l8_errs_hopt_space", "l8_errs_hfix_space"];
avg_table = avg_table + T;
tables{s} = T;

%end of rng loop below
s
end
avg_table = avg_table./length(seed);
writetable(avg_table, strjoin(['mu_' num2str(length(seed)) 'avg_sigma_' shifts '_' signal '.txt'], ''));

 %%
errs = 0;
for s = seed
   errs = (tables{s} - avg_table).^2;
end

errs = 1/(length(seed)-1).*errs;
errs = sqrt(errs);

writetable(errs, strjoin(['s_' num2str(length(seed)) 'avg_sigma_' shifts '_' signal '.txt'], ''));




