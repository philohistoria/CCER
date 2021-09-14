setwd("..") ## move to parent folder

library(devtools)
install("CCER")
library(CCER)

## then simulate data and run regression

m = 300 ## size of estimate
n = 5000 ## size of primary
b0 = 0.5
b1 = 1
b2 = 1

eps = rnorm(n + m, 0)
xtr = rnorm(n + m, mean = 0, sd = 1)
# xt = sigmoid::logistic(xtr)
xtb = ifelse(xtr> 0, 1, 0)
fpr_s = 0.1
fnr_s = 0.1
# mis classification index
misclass_fp = rbinom(n+m, 1,fpr_s)
misclass_fn = rbinom(n+m, 1,fnr_s)
# combined misclassification index
misclass = misclass_fn * (xtb == 1) | misclass_fp * (xtb == 0)
# misclassify true $Y$
xeb =  ifelse(misclass, 1 - xtb, xtb)

w = rnorm(n + m, 0)

y = b0 + b1 * xtb  + b2 * w + eps


valdata = data.frame(y = y[1:m], xeb = xeb[1:m], xtb = xtb[1:m], w =w[1:m])
mu = prop.table(table(valdata$xtb))[2]

# ## only X_estimate and Y are available for primary data
primary = data.frame(y = y[(m+1):(n+m)], xeb = xeb[(m+1):(n+m)],  w = w[(m+1):(n+m)])

pool= data.frame(y = y, xeb = xeb,  xtb = xtb, w = w)


formula = y ~ xeb
fn <- FN (valdata$xtb, valdata$xeb)
fp <-  FP(valdata$xtb, valdata$xeb)
tp <- TP(valdata$xtb, valdata$xeb)
tn <- TN(valdata$xtb, valdata$xeb)
ppv <- PPV (valdata$xtb, valdata$xeb)
npv <- NPV(valdata$xtb, valdata$xeb)

tau_prior = 0.001
bugs_dir = "/usr/bin/multibugs"
working_dir = "/home/han/Dropbox/Codes/measurement_error_model/bugs_results/"
n.iter = 6000
n.burnin = n.iter / 2
data = primary
predicted_col_name = "xeb"

tl = valdata$xtb
pl = valdata$xeb

d = bayesian_linear_indep (formula = formula, data = primary,fn = fn, fp = fp, tp = tp, tn = tn, ppv = ppv, npv = npv, tau_prior = 0.001, predicted_col_name = predicted_col_name,  working_dir = "/home/han/Dropbox/Codes/measurement_error_model/bugs_results/", n.iter = n.iter)
print (d$bayesian)


MLE <- MLE_linear(formula, data, predicted_col_name =  "xeb", ppv = ppv, npv = npv)
print (MLE)


MLEb <- CCER::MLE_linear_indep_bootstrap(formula, data, predicted_col_name =  "xeb", tl = valdata$xtb, pl = valdata$xeb, nboot = 1000)
