%%% SETUP %%% --------------------------------------------------------

seed = 1:20;
lambda_range = -2:.02:-1;

avg_table = 0;
tables = {0};

    
h_opt = zeros(1,length(lambda_range));
l8_errs_hfix_space = zeros(1,length(lambda_range));
l8_errs_hopt_space = zeros(1,length(lambda_range));
l8_errs_freq = zeros(1,length(lambda_range));


N = 1; l = 7; B = 10;
sigma = 1; hfix = 0.035; r = 0.001; 
zero_cond = 0; 
signal = "f1"; shifts = "unif"; thresh = 0;
nl = 60;


for s = seed


parfor q = 1:length(lambda_range)
lambda = 10^lambda_range(q);
n = ceil( nl*(log(1/lambda)));

[~, ~, ~, l8_errs_freq(q),l8_errs_hfix_space(q), l8_errs_hopt_space(q), ~, h_opt(q)] =...
    FUNC_singlerun(s, N,l,B,lambda,sigma,n, hfix, r, zero_cond,signal, shifts, thresh);
end


%% data saving
ll = 10.^lambda_range;
hf = repelem(hfix, 1,length(lambda_range));
reg = repelem(r, 1, length(lambda_range));
nn = ceil(nl*(log(1./ll)));
sig = repelem(sigma, 1, length(lambda_range));

T = table(nn', ll', sig', h_opt', hf', reg', l8_errs_freq',...
    l8_errs_hopt_space',l8_errs_hfix_space');
T.Properties.VariableNames = ["n","lambda", "sigma", "h_opt", "hfix", "r",...
    "l8_errs_freq", "l8_errs_hopt_space", "l8_errs_hfix_space"];
avg_table = avg_table + T;
tables{s} = T;

%end of rng loop below
end
avg_table = avg_table./length(seed);
writetable(avg_table, strjoin(['mu_' num2str(length(seed)) 'avg_lambda_' shifts '_' signal '.txt'], ''));

 %%
errs = 0;
for s = seed
   errs = (tables{s} - avg_table).^2;
end

errs = 1/(length(seed)-1).*errs;
errs = sqrt(errs);

writetable(errs, strjoin(['s_' num2str(length(seed)) 'avg_lambda_' shifts '_' signal '.txt'], ''));

