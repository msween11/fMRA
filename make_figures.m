
%% N data SIGMA = 0.5
 uf2 = readtable('data\sigma.5\mu_20avg_sig.5_unif_f2.txt');
uf1 = readtable('data\sigma.5\mu_20avg_sig.5_unif_f1.txt');
 uf4 = readtable('data\sigma.5\mu_20_c_avg_sig.5_unif_f4.txt');
uf3 = readtable('data\sigma.5\mu_20avg_sig.5_unif_f3.txt');

af2 = readtable('data\sigma.5\mu_20avg_sig.5_aper_f2.txt');
af1 = readtable('data\sigma.5\mu_20avg_sig.5_aper_f1.txt');
af4 = readtable('data\sigma.5\mu_20_c_avg_sig.5_aper_f4.txt');
af3 = readtable('data\sigma.5\mu_20avg_sig.5_aper_f3.txt');

%errors now
 uf2s = readtable('data\sigma.5\s_20avg_sig.5_unif_f2.txt');
uf1s = readtable('data\sigma.5\s_20avg_sig.5_unif_f1.txt');
 uf4s = readtable('data\sigma.5\s_20_c_avg_sig.5_unif_f4.txt');
uf3s = readtable('data\sigma.5\s_20avg_sig.5_unif_f3.txt');

af2s = readtable('data\sigma.5\s_20avg_sig.5_aper_f2.txt');
af1s = readtable('data\sigma.5\s_20avg_sig.5_aper_f1.txt');
af4s = readtable('data\sigma.5\s_20_c_avg_sig.5_aper_f4.txt');
af3s = readtable('data\sigma.5\s_20avg_sig.5_aper_f3.txt');

%% N data SIGMA = 1
 uf2sig1 = readtable('data\sigma1\mu_20avg_unif_f2.txt');
uf1sig1 = readtable('data\sigma1\mu_20avg_unif_f1.txt');
 uf4sig1 = readtable('data\sigma1\mu_20_c_avg_sig1_unif_test_f4.txt');
uf3sig1 = readtable('data\sigma1\mu_20avg_unif_f3.txt');

af2sig1 = readtable('data\sigma1\mu_20avg_aper_f2.txt');
af1sig1 = readtable('data\sigma1\mu_20avg_aper_f1.txt');
af4sig1 = readtable('data\sigma1\mu_20_c_avg_sig1_aper_f4.txt');
af3sig1 = readtable('data\sigma1\mu_20avg_aper_f3.txt');

%errors now
 uf2sig1s = readtable('data\sigma1\s_20avg_unif_f2.txt');
uf1sig1s = readtable('data\sigma1\s_20avg_unif_f1.txt');
 uf4sig1s = readtable('data\sigma1\s_20_c_avg_sig1_unif_test_f4.txt');
uf3sig1s = readtable('data\sigma1\s_20avg_unif_f3.txt');

af2sig1s = readtable('data\sigma1\s_20avg_aper_f2.txt');
af1sig1s = readtable('data\sigma1\s_20avg_aper_f1.txt');
af4sig1s = readtable('data\sigma1\s_20_c_avg_sig1_aper_f4.txt');
af3sig1s = readtable('data\sigma1\s_20avg_aper_f3.txt');




%% sigma data
sf2 = readtable('data\mu_20avg_sigma_unif_f2.txt');
sf1 = readtable('data\mu_20avg_sigma_unif_f1.txt');
sf4 = readtable('data\mu_20avg_sigma_unif_f4.txt');
sf3 = readtable('data\mu_20avg_sigma_unif_f3.txt');

nsf1 = readtable('data\mu_20avg_sigma_nchange_unif_f1.txt');
nsf2 = readtable('data\mu_20avg_sigma_nchange_unif_f2.txt');
nsf3 = readtable('data\mu_20avg_sigma_nchange_unif_f3.txt');
nsf4 = readtable('data\mu_20avg_sigma_nchange_unif_f4.txt');

nsf1s = readtable('data\s_20avg_sigma_nchange_unif_f1.txt');
nsf2s = readtable('data\s_20avg_sigma_nchange_unif_f2.txt');
nsf3s = readtable('data\s_20avg_sigma_nchange_unif_f3.txt');
nsf4s = readtable('data\s_20avg_sigma_nchange_unif_f4.txt');

% errors now
sf2s = readtable('data\s_20avg_sigma_unif_f2.txt');
sf1s = readtable('data\s_20avg_sigma_unif_f1.txt');
sf4s = readtable('data\s_20avg_sigma_unif_f4.txt');
sf3s = readtable('data\s_20avg_sigma_unif_f3.txt');
%% lambda data
lf2 = readtable('data\mu_20avg_lambda_unif_f2.txt');
lf1 = readtable('data\mu_20avg_lambda_unif_f1.txt');
lf3 = readtable('data\mu_20avg_lambda_unif_f3.txt');
lf4 = readtable('data\mu_20avg_lambda_unif_f4.txt');

% errors now
lf2s = readtable('data\s_20avg_lambda_unif_f2.txt');
lf1s = readtable('data\s_20avg_lambda_unif_f1.txt');
lf3s = readtable('data\s_20avg_lambda_unif_f3.txt');
lf4s = readtable('data\s_20avg_lambda_unif_f4.txt');

%% PLOTTING N, SIGMA = 1

nerrssig1 = tiledlayout(1,2, 'TileSpacing', 'tight')
set(gcf,'units','normalized','outerposition',[0 0 .6 .6])

ax1 = nexttile;
b1 = round(polyfit(log2(uf2sig1.n), log2(uf2sig1.l8_errs_hopt_space),1),2);
b2 = round(polyfit(log2(uf1sig1.n), log2(uf1sig1.l8_errs_hopt_space),1),2);
b3 = round(polyfit(log2(uf3sig1.n), log2(uf3sig1.l8_errs_hopt_space),1 ),2);
b4 = round(polyfit(log2(uf4sig1.n), log2(uf4sig1.l8_errs_hopt_space),1),2);

errorbar(ax1,log2(uf1sig1.n), log2(uf1sig1.l8_errs_hopt_space), ...
    1.4427*uf1sig1s.l8_errs_hopt_space./uf1sig1.l8_errs_hopt_space)
hold on
errorbar(ax1,log2(uf2sig1.n), log2(uf2sig1.l8_errs_hopt_space), ...
    1.4427*uf2sig1s.l8_errs_hopt_space./uf2sig1.l8_errs_hopt_space)
errorbar(log2(uf3sig1.n), log2(uf3sig1.l8_errs_hopt_space),...
    1.4427*uf3sig1s.l8_errs_hopt_space./uf3sig1.l8_errs_hopt_space)
errorbar(log2(uf4sig1.n), log2(uf4sig1.l8_errs_hopt_space),...
     1.4427*uf4sig1s.l8_errs_hopt_space./uf4sig1.l8_errs_hopt_space)
xlabel('$\log_2(N)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
axis square
 legend('Location','southwest');
legend(['f_1 ' ' m = ', num2str(b2(1)), ', \beta = 2'],['f_2  m = ', num2str(b1(1)), ', \beta = 4'],...
    ['f_3  m = ', num2str(b3(1)), ', SS'],['f_4  m = ', num2str(b4(1)), ', SS'])
ax2 = nexttile;
 b1 = round(polyfit(log2(af2sig1.n), log2(af2sig1.l8_errs_hopt_space),1),2);
b2 = round(polyfit(log2(af1sig1.n), log2(af1sig1.l8_errs_hopt_space),1),2);
b3 = round(polyfit(log2(af3sig1.n), log2(af3sig1.l8_errs_hopt_space),1 ),2);
b4 = round(polyfit(log2(af4sig1.n), log2(af4sig1.l8_errs_hopt_space),1),2);
errorbar(log2(af1sig1.n), log2(af1sig1.l8_errs_hopt_space),...
    1.4427*af1sig1s.l8_errs_hopt_space./af1sig1.l8_errs_hopt_space)
hold on
errorbar(ax2,log2(af2sig1.n), log2(af2sig1.l8_errs_hopt_space),...
    1.4427*af2sig1s.l8_errs_hopt_space./af2sig1.l8_errs_hopt_space)
errorbar(log2(af3sig1.n), log2(af3sig1.l8_errs_hopt_space),...
   1.4427*af3sig1s.l8_errs_hopt_space./af3sig1.l8_errs_hopt_space )
errorbar(log2(af4sig1.n), log2(af4sig1.l8_errs_hopt_space),...
    1.4427*af4sig1s.l8_errs_hopt_space./af4sig1.l8_errs_hopt_space)
 
xlabel('$\log_2(N)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
axis square
legend('Location','southwest')
legend(['f_1 ' ' m = ', num2str(b2(1)), ', \beta = 2'],['f_2  m = ', num2str(b1(1)), ', \beta = 4'],...
    ['f_3  m = ', num2str(b3(1)), ', SS'],['f_4  m = ', num2str(b4(1)), ', SS'])
exportgraphics(nerrssig1, 'plots/n_sig1_error.png', 'ContentType', 'vector');

%% N ERROR SIGMA = 0.5
fig1_sig5 = tiledlayout(1,2, 'TileSpacing', 'tight')
set(gcf,'units','normalized','outerposition',[0 0 .6 .6])

ax1 = nexttile;
b1 = round(polyfit(log2(uf2.n), log2(uf2.l8_errs_hopt_space),1),2);
b2 = round(polyfit(log2(uf1.n), log2(uf1.l8_errs_hopt_space),1),2);
b3 = round(polyfit(log2(uf3.n), log2(uf3.l8_errs_hopt_space),1 ),2);
b4 = round(polyfit(log2(uf4.n), log2(uf4.l8_errs_hopt_space),1),2);
errorbar(ax1,log2(uf1.n), log2(uf1.l8_errs_hopt_space), ...
    1.4427*uf1s.l8_errs_hopt_space./uf1.l8_errs_hopt_space)
hold on
errorbar(ax1,log2(uf2.n), log2(uf2.l8_errs_hopt_space), ...
    1.4427*uf2s.l8_errs_hopt_space./uf2.l8_errs_hopt_space)
errorbar(log2(uf3.n), log2(uf3.l8_errs_hopt_space),...
    1.4427*uf3s.l8_errs_hopt_space./uf3.l8_errs_hopt_space)
 errorbar(log2(uf4.n), log2(uf4.l8_errs_hopt_space),...
     1.4427*uf4s.l8_errs_hopt_space./uf4.l8_errs_hopt_space)
xlabel('$\log_2(N)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
axis square
legend('Location','southwest')
legend(['f_1 ' ' m = ', num2str(b2(1)), ', \beta = 2'],['f_2  m = ', num2str(b1(1)), ', \beta = 4'],...
    ['f_3  m = ', num2str(b3(1)), ', SS'],['f_4  m = ', num2str(b4(1)), ', SS'])

ax2 = nexttile;
 b1 = round(polyfit(log2(af2.n), log2(af2.l8_errs_hopt_space),1),2);
b2 = round(polyfit(log2(af1.n), log2(af1.l8_errs_hopt_space),1),2);
b3 = round(polyfit(log2(af3.n), log2(af3.l8_errs_hopt_space),1 ),2);
b4 = round(polyfit(log2(af4.n), log2(af4.l8_errs_hopt_space),1),2);
errorbar(log2(af1.n), log2(af1.l8_errs_hopt_space),...
    1.4427*af1s.l8_errs_hopt_space./af1.l8_errs_hopt_space)
hold on
errorbar(ax2,log2(af2.n), log2(af2.l8_errs_hopt_space),...
    1.4427*af2s.l8_errs_hopt_space./af2.l8_errs_hopt_space)
errorbar(log2(af3.n), log2(af3.l8_errs_hopt_space),...
   1.4427*af3s.l8_errs_hopt_space./af3.l8_errs_hopt_space )
errorbar(log2(af4.n), log2(af4.l8_errs_hopt_space),...
    1.4427*af4s.l8_errs_hopt_space./af4.l8_errs_hopt_space)
 
xlabel('$\log_2(N)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
axis square
legend('Location','southwest')
legend(['f_1 ' ' m = ', num2str(b2(1)), ', \beta = 2'],['f_2  m = ', num2str(b1(1)), ', \beta = 4'],...
    ['f_3  m = ', num2str(b3(1)), ', SS'],['f_4  m = ', num2str(b4(1)), ', SS'])
 exportgraphics(fig1_sig5, 'plots/n_sig.5_error.png', 'ContentType', 'vector');

%% PLOTTING SIGMA

figsigma = figure
set(gcf,'units','normalized','outerposition',[0 0 .5 .5])
s2 = round(polyfit(log2(sf2.sigma(1:end)), log2(sf2.l8_errs_hopt_space(1:end)),1),2);
s1 = round(polyfit(log2(sf2.sigma(1:end)), log2(sf1.l8_errs_hopt_space(1:end)),1),2);
s3 = round(polyfit(log2(sf2.sigma(1:end)), log2(sf3.l8_errs_hopt_space(1:end)),1 ),2);
s4 = round(polyfit(log2(sf2.sigma(1:end)), log2(sf4.l8_errs_hopt_space(1:end)),1),2);

errorbar(log2(sf1.sigma(1:end)), log2(sf1.l8_errs_hopt_space(1:end)),...
    1.4427*sf1s.l8_errs_hopt_space(1:end)./sf1.l8_errs_hopt_space(1:end))
hold on
errorbar( log2(sf2.sigma(1:end)), log2(sf2.l8_errs_hopt_space(1:end)),...
    1.4427*sf2s.l8_errs_hopt_space(1:end)./sf2.l8_errs_hopt_space(1:end))
errorbar(log2(sf3.sigma(1:end)), log2(sf3.l8_errs_hopt_space(1:end)),...
    1.4427*sf3s.l8_errs_hopt_space(1:end)./sf3.l8_errs_hopt_space(1:end))
errorbar(log2(sf4.sigma(1:end)), log2(sf4.l8_errs_hopt_space(1:end)),...
    1.4427*sf4s.l8_errs_hopt_space(1:end)./sf4.l8_errs_hopt_space(1:end))

xlabel('$\log_2(\sigma)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
axis square
legend('Location','southeast')
legend(['f_1 ' ' m = ', num2str(b2(1)), ', \beta = 2'],['f_2  m = ', num2str(b1(1)), ', \beta = 4'],...
    ['f_3  m = ', num2str(b3(1)), ', SS'],['f_4  m = ', num2str(b4(1)), ', SS'])

 exportgraphics(figsigma, 'plots/sigma_error.png', 'ContentType', 'vector');
%% PLOTTING LAMBDA
lambdafig = figure
set(gcf,'units','normalized','outerposition',[0 0 .4 .4])
plot(log10(lf2.lambda), log10(lf1.l8_errs_hopt_space))
hold on
plot( log10(lf2.lambda), log10(lf2.l8_errs_hopt_space))
plot(log10(lf2.lambda), log10(lf3.l8_errs_hopt_space))
plot(log10(lf2.lambda), log10(lf4.l8_errs_hopt_space))

xlabel('$\log_{10}(\lambda)$','interpreter','latex')
ylabel(['$$ \log_{10}( || f - \widehat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
legend('Location','southeast')
legend(['f_1, \beta = 2'],['f_2, \beta = 4'],['f_3, SS'],...
    ['f_4, SS'])
axis square
 exportgraphics(lambdafig, 'plots/lambda_error.png', 'ContentType', 'vector');

 %% PLOTTING SIGMA NCHANGE

signchangefig = figure
set(gcf,'units','normalized','outerposition',[0 0 .5 .5])
plot( log2(nsf2.sigma), log2(nsf1.l8_errs_hopt_space))
hold on
plot( log2(nsf2.sigma), log2(nsf2.l8_errs_hopt_space))
plot( log2(nsf2.sigma), log2(nsf3.l8_errs_hopt_space))
 plot(log2(nsf2.sigma), log2(nsf4.l8_errs_hopt_space))

xlabel('$\log_2(\sigma)$','interpreter','latex')
ylabel(['$$ \log_2( || f - \hat{f} ||_\infty / ||f||_\infty) $$'], 'Interpreter','latex')
grid on
legend('Location','southwest')
legend(['f_1, \beta = 2'],['f_2, \beta = 4'],...
    ['f_3, SS'],['f_4, SS']  )
axis square
  exportgraphics(signchangefig, 'plots/sigma_nchange_error.png', 'ContentType', 'vector');


%% FOUR SIGNALS PLUS RECOVERY PLOT
seed = 1; N = 1; l = 5; B = 10; nn = 15; % <- number of samples to disp.
n = 100000; lambda = .1; sigma = 1;

x=-N:2^(-l):N-2^(-l);
X=-2*N:2^(-l):2*N-2^(-l);
w=-pi*(2^l):(pi/N):pi*(2^l)-(pi/N); 
W=-pi*(2^l):(pi/(2*N)):pi*(2^l)-(pi/(2*N)); 
Wext = -pi*(2^l):(pi/(2*N*B)):pi*(2^l)-(pi/(2*N*B)); 
symfft = @(x) ifftshift(fft(fftshift(x)));
x1 = length(x)/2;


%TRIANGULAR PULSE
f1 = @(x) max(1-abs(x),0);
f1 = @(x) f1(x) + f1(pi*x); 

nrm = @(x) max(abs(f1(x)));
f1 = @(x) f1(x) / nrm(x);

%GAUSSIAN MIXTURE
means = [-.4;.4]; devs = [.02];
f4 = gmdistribution(means, devs);
f4 = @(x) pdf(f4, x')';

nrm = @(x) max(abs(f4(x)));
f4 = @(x) f4(x) / nrm(x);
f4 = @(x) f4(x-.1);

%GABOR WAVELET
k=8;
f3 = @(x) exp(-20*x.^2).*cos(k*x);
f3 = @(x) f3(x-.3);

nrm = @(x) max(abs(f3(x)));
f3 = @(x) f3(x) / nrm(x);

%THIRD ORDER SPLINE
 f2 = @(x) (1/6)*( 0.* (x<0)+...
    (x.^3) .* (0<= x & x < 1)+...
    (-3*(x-1).^3 + 3*(x-1).^2+3*(x-1)+1) .* (1<= x & x < 2)+...
    (3*(x-2).^3-6*(x-2).^2+4) .* (2 <= x & x < 3)+...
    ((4-x).^3) .* (3 <=x & x < 4)+...
    0 .* (x >= 4));

f2 = @(x) f2(2*x+2);
f2 = @(x) f2(x) + f2(pi*x);

nrm = @(x) max(abs(f2(x)));
f2 = @(x) f2(x) / nrm(x);


fcns = {'f1', 'f2', 'f3', 'f4'}; sampfuncs = {f1,f2,f3,f4};
shifts = 'unif'; thresh = 0.001;
freqrecs = zeros(length(fcns), length(Wext));
freqtrues = zeros(length(fcns), length(Wext));
spacerecs = zeros(length(fcns),length(X));
hfix = [0.035,0.035, 0.04, 0.07];
r = [0.01,0.01,0.01,0.0001];
zero_cond = [0,0,0,1];
fsamples = {0};

covr = zeros(length(X));
for j = 1:length(X)
    for k = 1:length(X)
        covr(j,k) = (sigma^2)*exp(-norm(X(j)-X(k))^2/(2*lambda^2)); 
    end
end

noise = mvnrnd(zeros(1,length(covr)), covr, n);
dist = makedist('Uniform', 'lower', -1, 'upper', 1);
sampleshifts = random(dist,1, n);

for q = 1:length(fcns)
    [freqrecs(q,:),~, spacerecs(q,:),  ~,~, ~, freqtrues(q,:), ~] =...
    FUNC_singlerun(seed, N,l,B,lambda,sigma,n, hfix(q),...
    r(q), zero_cond(q),fcns{q}, shifts, thresh);

    samples = zeros(n, length(X));
    for j = 1:n
        samples(j,:) = sampfuncs{q}(X-sampleshifts(j))+noise(j,:); 

    end
    fsamples{q} = samples(1:nn,:); 
q
end
%%
figintroplot = tiledlayout(1,4, 'TileSpacing','tight')
set(gcf,'units','normalized','outerposition',[0 0 1 1])

ax1 = nexttile;
plot(ax1, x,f1(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
for j = 1:10
    sample_plot = fsamples{1}(j,:); 
    plot(x, sample_plot(1,x1+1:end-x1), 'Color', [0,0,1,0.1], 'LineWidth', 2);
end
plot(x,spacerecs(1,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax2 = nexttile;
plot(ax2, x,f2(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
for j = 1:10
    sample_plot = fsamples{2}(j,:); 
    plot(x, sample_plot(1,x1+1:end-x1), 'Color', [0,0,1,0.1], 'LineWidth', 2);
end
plot(x,spacerecs(2,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax3 = nexttile;
plot(ax3, x,f3(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
for j = 1:10
    sample_plot = fsamples{3}(j,:); 
    plot(x, sample_plot(1,x1+1:end-x1), 'Color', [0,0,1,0.1], 'LineWidth', 2);
end
plot(x,spacerecs(3,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax4 = nexttile;
plot(ax4, x,f4(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
for j = 1:10
    sample_plot = fsamples{4}(j,:); 
    plot(x, sample_plot(1,x1+1:end-x1), 'Color', [0,0,1,0.1], 'LineWidth', 2);
end
plot(x,spacerecs(4,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

 exportgraphics(figintroplot, 'plots/intro_plot.png', 'ContentType', 'vector');
%% signal recovery examples

figsignals = tiledlayout(2,4, 'TileSpacing','tight')
set(gcf,'units','normalized','outerposition',[0 0 1 1])

ax1 = nexttile;
plot(ax1, x,f1(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
plot(x,spacerecs(1,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax2 = nexttile;
plot(ax2, x,f2(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
plot(x,spacerecs(2,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax3 = nexttile;
plot(ax3, x,f3(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
plot(x,spacerecs(3,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax4 = nexttile;
plot(ax4, x,f4(x),'Color', [0,0,1,1], 'LineWidth', 2)
hold on
plot(x,spacerecs(4,x1+1:end-x1),'Color', [1,0,0,1], 'LineWidth', 1)
axis square

ax5 = nexttile;
plot(ax5, Wext, real(freqtrues(1,:)),'Color', [0,0,1,.8],'LineWidth', 2)
hold on
plot(Wext, real(freqrecs(1,:)))
axis square
xlim([-100 100])

ax6 = nexttile;
plot(ax6, Wext, real(freqtrues(2,:)),'Color', [0,0,1,.5],'LineWidth', 2)
hold on
plot(Wext, real(freqrecs(2,:)))
axis square
xlim([-100 100])

ax7 = nexttile;
plot(ax7, Wext, real(freqtrues(3,:)),'Color', [0,0,1,.7],'LineWidth', 2)
hold on
plot(Wext, real(freqrecs(3,:)))
axis square
xlim([-100 100])

ax8 = nexttile;
plot(ax8, Wext, real(freqtrues(4,:)),'Color', [0,0,1,.6],'LineWidth', 2)
hold on
plot(Wext, real(freqrecs(4,:)))
axis square
xlim([-100 100])

 exportgraphics(figsignals, 'plots/signals.png', 'ContentType', 'vector');


