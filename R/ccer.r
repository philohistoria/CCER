# copied from self-implemented-glm.r
# has MLE for linear regression and logistic regressions
# library(dqrng)
# library("coda")
# library(lattice)
# library(ggplot2)
# library(sigmoid)
# library(R2jags)
# library(MASS)

#' @export
FN <- function(y, ypred){
  tab <- table(y, ypred)
  return(tab[2,1])
}
#' @export
FP <- function(y, ypred){
  tab <- table(y, ypred)
  return(tab[1,2])
}
#' @export
TP <- function(y, ypred){
  tab <- table(y, ypred)
  return(tab[2,2])
}
#' @export
TN <- function(y, ypred){
  tab <- table(y, ypred)
  return(tab[1,1])
}

#' @export
FNR <- function(y, ypred){
  # FN / p
  tab <- table(y, ypred)
  return((tab[2,1])/(tab[2,1]+tab[2,2]))
}
#' @export
TPR <- function(y, ypred){
  # tp / p
  tab <- table(y, ypred)
  return((tab[2,2])/(tab[2,1]+tab[2,2]))
}

FPR <- function(y, ypred){
  # FP / N
  tab <- table(y, ypred)
  return(tab[1,2]/(tab[1,1]+tab[1,2]))
}
#' @export
PPV <- function(y, ypred){
  # tp / (tp + fp)
  tab <- table(y, ypred)
  return(tab[2,2]/(tab[1,2]+tab[2,2]))
}
#' @export
NPV <- function(y, ypred){
  # tn / (tn + fn)
  tab <- table(y, ypred)
  return(tab[1,1]/(tab[1,1]+tab[2,1]))
}

logit = function(mX, vBeta) {
  return(exp(mX %*% vBeta)/(1+ exp(mX %*% vBeta)) )
}

cdf_logit= function(mX, vBeta) {
  return(1/(1+ exp(-1 * (mX %*% vBeta))) )
}

# ytr = rnorm(n + m, mean = b0 + b1 * x + b2 * w + eps, sd = 1)
complement <- function(y, rho, x) {
  if (missing(x)) x <- rnorm(length(y)) # Optional: supply a default if `x` is not given
  y.perp <- residuals(lm(x ~ y))
  rho * sd(y.perp) * y + y.perp * sd(y) * sqrt(1 - rho^2)
}



# MLE --------------
## MLE for ML classifications as independent variables (linear regression) -----------------
linear_density <- function(vY, mX, vBeta){
  return (dnorm(vY - mX %*% vBeta ))
}

logLikelihood_linear = function(vBeta, mX, vY, predicted_col_name, ppv = 0, npv = 0) {
  mX1 = mX
  mX1[,predicted_col_name] <- 1

  mX0 = mX
  mX0[,predicted_col_name] <- 0

  p1 = linear_density(vY, mX1, vBeta) # when vD = 1
  p0 = linear_density(vY, mX0, vBeta) # when vD = 0
  return(-sum(
    mX[,predicted_col_name] * log( ppv * p1 + (1 - ppv) * p0) +
      (1- mX[,predicted_col_name]) * log( (1-npv) * p1 + npv * p0)
  )
  )
}

#' MLE estimator for linear regression with binary classification as *independent* variables
#'
#' This functions takes binary classification as *independent* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param predicted_col_name string or integer indicating the column of the variable that is predicted
#' @param ppv positive predicted value
#' @param npv negative predicted value
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
MLE_linear <- function(formula, data, predicted_col_name, ppv = 0, npv = 0) {

  initModel = lm(formula,
                 data = data)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.01)
  ss = coef(summary(initModel))[, 2]
  inff = coef(initModel) - 100*ss
  uppf = coef(initModel) + 100*ss


  # optimize
  optimModel <- optim(vBetfnr, logLikelihood_linear,
                      mX = mX, vY = vY,
                      predicted_col_name = predicted_col_name,
                      ppv = ppv, npv = npv,
                      method = 'L-BFGS-B',
                      lower = rep(inff, ncol(mX)), upper = rep(uppf, ncol(mX)),
                      hessian=TRUE,  control = list(maxit = 300))
  # construct output
  coef = optimModel$par
  coef.sd = sqrt(diag(solve( optimModel$hessian)))
  tv  <- coef  / coef.sd
  pv <- 2 * pt(tv, df = nrow(mX) - ncol(mX), lower.tail = F) ## two-tailed
  d = data.frame("Estimate" = coef,  "Std. Error" = coef.sd, "z value" = tv,  "Pr(>|z|)" = pv, check.names = FALSE)
  return (d)
}

## MLE for ML classification as outcomes (logistic regression) -----------------
logLikelihoodLogit = function(vBeta, mX, vY, fnr = 0, fpr = 0) {
  return(-sum(
    vY * log( fpr + (1 - fnr - fpr) * logit(mX, vBeta) ) +
      (1-vY)* log( 1 - fpr - (1- fnr - fpr) *  logit(mX, vBeta))
  )
  )
}

#' MLE estimator for logistic regression with binary classification as *outcome* variable
#'
#' This functions takes binary classification as *outcome* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
MLE_logit <- function(formula, data, fnr = 0, fpr = 0) {

  initModel = glm(formula,
                      data = data,
                      family = binomial)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.05)
  ss = coef(summary(initModel))[, 2]
  inff = coef(initModel) - 100*ss
  uppf = coef(initModel) + 100*ss

  # optimize
  optimModel <- optim(vBetfnr, logLikelihoodLogit,
                     mX = mX, vY = vY, fnr = fnr, fpr = fpr,
                     method = 'L-BFGS-B',
                     lower = rep(inff, ncol(mX)), upper = rep(uppf, ncol(mX)),

                     hessian=TRUE,  control = list(maxit =300))
  # construct output
  coef = optimModel$par
  coef.sd = sqrt(diag(solve( optimModel$hessian)))
  tv  <- coef  / coef.sd
  pv <- 2 * pt(tv, df = nrow(mX) - ncol(mX), lower.tail = F) ## two-tailed
  d = data.frame("Estimate" = coef,  "Std. Error" = coef.sd, "z value" = tv,  "Pr(>|z|)" = pv, check.names = FALSE)
  return (d)
}





## MLE for ML classification as outcome (linear regression or linear prob model or LPM) -----------------

logLikelihood_LPM = function(vBeta, mX, vY, fnr = 0, fpr = 0) {

  p1 = mX %*% vBeta
  # constrain the probabilities to be within 0 and 1
  p1[p1 >= 1] <- 1
  p1[p1 <= 0] <- 0# .000001

  # p1 = mX %*% vBeta
  # table((1 - fnr - (1- fnr - fpr) * p1) < 0)
  return(-sum(
    vY * log( fpr + (1 - fnr - fpr) * p1 ) +
      (1-vY)* log( 1 - fpr - (1- fnr - fpr) * p1)
  , na.rm =  T)
  )
}



#' MLE estimator for linear regression with binary classification as *outcome* variable (linear probability model)
#'
#' This functions takes binary classification as *outcome* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
MLE_LPM <- function(formula, data,  fnr = 0, fpr = 0) {

  initModel = lm(formula,
                 data = data)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.05)

  # optimize
  optimModel <- optim(vBetfnr, logLikelihood_LPM,
                      mX = mX, vY = vY,
                      fpr = fpr, fnr = fnr,
                      # method = 'L-BFGS-B',
                      method = 'BFGS',
                      hessian=TRUE,  control = list(maxit =200))
  # construct output
  coef = optimModel$par
  coef.sd = sqrt(diag(solve( optimModel$hessian)))
  tv  <- coef  / coef.sd
  pv <- 2 * pt(tv, df = nrow(mX) - ncol(mX), lower.tail = F) ## two-tailed
  d = data.frame("Estimate" = coef,  "Std. Error" = coef.sd, "z value" = tv,  "Pr(>|z|)" = pv, check.names = FALSE)
  return (d)
}

# Bootstrap ---------------------

## bootstrap for logistic regresion (outcome case)------

#' Bootstrap estimator for logistic regression with binary classification as *outcome* variable
#'
#' This functions takes binary classification as *outcome* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param tl vector of true values
#' @param pl vector of predicted values; needs to be of the same dimension of the vevtor of true values.
#' @param nboot number of bootstrap samples
#' @return_boot_samples whether to return raw bootstrap estimates
#' @return if return_boot_samples is false, return mean/standard error/quantile statistics from bootstrap sample only; otherwise return a list, the first element of which is a dataframe of `nboot` rows, with each row as one estimated coefficients from a bootstrap resample, and the second element of which a dataframe of `nboot` rows, with each row as one pair  of estimated error rates: FPR and FNR. Powerful users can set `return_boot_samples` to be true to calculate the confidence interval of estimates by their own.
#' @export
MLE_logit_bootstrap <- function(formula, data, tl , pl, nboot = 10000, return_boot_samples = FALSE) {

  initModel = glm(formula,
                  data = data,
                  family = binomial)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.05)
  ss = coef(summary(initModel))[, 2]
  inff = coef(initModel) - 100*ss
  uppf = coef(initModel) + 100*ss

  res <- c()
  err <- c()
  for (i in 1:nboot){

    # resample predicted label and true label
    # to recalulate bootstrapped FNR and FPR
    # pl_boot <- dqsample(pl, length(pl), replace = T)
    idx <- dqsample.int(length(pl), length(pl), replace = T)
    pl_boot <- pl[idx]
    tl_boot <- tl[idx]

    # fnr0 <- FNR (tl, pl)
    # fpr0 <- FPR (tl, pl)

    fnr <- FNR (tl_boot, pl_boot)
    fpr <- FPR (tl_boot, pl_boot)
    # print (paste(fnr0, fpr0, fnr, fpr))
    # optimize
    optimModel <- optim(vBetfnr, logLikelihoodLogit,
                        mX = mX, vY = vY, fnr = fnr, fpr = fpr,
                        # method = 'L-BFGS-B',
                        # method = 'BFGS',
                        method = 'L-BFGS-B',
                        lower = rep(inff, ncol(mX)), upper = rep(uppf, ncol(mX)),
                        hessian=TRUE,  control = list(maxit = 300))
    coef = optimModel$par
    coef.sd = sqrt(diag(solve( optimModel$hessian)))
    # res <- rbind(res, data.frame("Estimate" = coef,  "Std. Error" = coef.sd))
    res <- rbind(res, t(coef))
    err <- rbind(err, c(fnr, fpr))
  }
  # construct output

  est <- colMeans(res)
  sdd <- apply(res, 2, sd)

  fnr <- mean(err[,1])
  fpr <- mean(err[,2])
  fnr.sd <- sd(err[,1])
  fpr.sd <- sd(err[,2])

  d0 = data.frame("Estimate" = est,  "Std. Error" = sdd, check.names = FALSE)
  d = rbind(d0, data.frame("Estimate" = c(fnr, fpr),  "Std. Error" = c(fnr.sd, fpr.sd), check.names = FALSE))

  # return coefficient estimates as well as raw res
  if (return_boot_samples ){
    return (list(resamples = res, raw_error = err))
    }else{
  return (d)
    }
}

## bootstrap for linear probability model (outcome case)------

#' Bootstrap estimator for linear regression with binary classification as *outcome* variable
#'
#' This functions takes binary classification as *outcome* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param tl vector of true values
#' @param pl vector of predicted values; needs to be of the same dimension of the vevtor of true values.
#' @param nboot number of bootstrap samples
#' @return_boot_samples whether to return raw bootstrap estimates
#' @return if return_boot_samples is false, return mean/standard error/quantile statistics from bootstrap sample only; otherwise return a list, the first element of which is a dataframe of `nboot` rows, with each row as one estimated coefficients from a bootstrap resample, and the second element of which a dataframe of `nboot` rows, with each row as one pair  of estimated error rates: FPR and FNR. Powerful users can set `return_boot_samples` to be true to calculate the confidence interval of estimates by their own.
#' @export
MLE_LPM_bootstrap <- function(formula, data, tl , pl, nboot = 10000, return_boot_samples = FALSE) {

  initModel = lm(formula,
                 data = data)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.05)
  ss = coef(summary(initModel))[, 2]
  inff = coef(initModel) - 10*ss
  uppf = coef(initModel) + 10*ss

  res <- c()
  err <- c()
  for (i in 1:nboot){

    idx <- dqsample.int(length(pl), length(pl), replace = T)
    pl_boot <- pl[idx]
    tl_boot <- tl[idx]


    fnr <- FNR (tl_boot, pl_boot)
    fpr <- FPR (tl_boot, pl_boot)

    optimModel <- optim(vBetfnr, logLikelihood_LPM,
                        mX = mX, vY = vY, fnr = fnr, fpr = fpr,
                        # method = 'BFGS',
                        method = 'L-BFGS-B',
                        lower = rep(inff, ncol(mX)), upper = rep(uppf, ncol(mX)),
                        hessian=TRUE,  control = list(maxit = 300))
    coef = optimModel$par
    coef.sd = sqrt(diag(solve( optimModel$hessian)))
    # res <- rbind(res, data.frame("Estimate" = coef,  "Std. Error" = coef.sd))
    res <- rbind(res, t(coef))
    err <- rbind(err, c(fnr, fpr))
  }
  # construct output

  est <- colMeans(res)
  sdd <- apply(res, 2, sd)

  fnr <- mean(err[,1])
  fpr <- mean(err[,2])
  fnr.sd <- sd(err[,1])
  fpr.sd <- sd(err[,2])

  d0 = data.frame("Estimate" = est,  "Std. Error" = sdd, check.names = FALSE)
  d = rbind(d0, data.frame("Estimate" = c(fnr, fpr),  "Std. Error" = c(fnr.sd, fpr.sd), check.names = FALSE))

  # return coefficient estimates as well as raw res
  if (return_boot_samples ){
    return (list(resamples = res, raw_error = err))
  }else{
    return (d)
  }
}


## bootstrap for liner regression (predicted independent variable)------


#' Bootstrap estimator for linear regression with binary classification as *independent* variables
#'
#' This functions takes binary classification as *independent* variables, apply MLE estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param predicted_col_name string or integer indicating the column of the variable that is predicted
#' @param tl vector of true values
#' @param pl vector of predicted values; needs to be of the same dimension of the vevtor of true values.
#' @param nboot number of bootstrap samples
#' @return_boot_samples whether to return raw bootstrap estimates
#' @return if return_boot_samples is false, return mean/standard error/quantile statistics from bootstrap sample only; otherwise return a list, the first element of which is a dataframe of `nboot` rows, with each row as one estimated coefficients from a bootstrap resample, and the second element of which a dataframe of `nboot` rows, with each row as one pair  of estimated error rates: PPV and NPV. Powerful users can set `return_boot_samples` to be true to calculate the confidence interval of estimates by their own.
#' @export
MLE_linear_indep_bootstrap <- function(formula, data,predicted_col_name, tl , pl, nboot = 10000, return_boot_samples = FALSE) {


  initModel = lm(formula,
                 data = data)
  mX = model.matrix(initModel)
  vY = model.frame(initModel)[,1]
  vBetfnr = coef(initModel) + rnorm(ncol(mX), 0, 0.05)
  ss = coef(summary(initModel))[, 2]
  inff = coef(initModel) - 100*ss
  uppf = coef(initModel) + 100*ss

  res <- c()
  err <- c()
  d <- c()
  for (i in 1:nboot){

    idx <- dqrng::dqsample.int(length(pl), length(pl), replace = T)
    pl_boot <- pl[idx]
    tl_boot <- tl[idx]

    ppv <- PPV (tl_boot, pl_boot)
    npv <- NPV (tl_boot, pl_boot)

    # optimize
    optimModel <- optim(vBetfnr, logLikelihood_linear,
                        mX = mX, vY = vY,
                        predicted_col_name = predicted_col_name,
                        ppv = ppv, npv = npv,
                        method = 'L-BFGS-B',
                        lower = rep(inff, ncol(mX)), upper = rep(uppf, ncol(mX)),
                        # method = 'BFGS',
                        hessian=TRUE,  control = list(maxit = 300))
    coef = optimModel$par
    coef.sd = sqrt(diag(solve( optimModel$hessian)))
    res <- rbind(res, t(coef))
    err <- rbind(err, c(ppv, npv))
  }
  # construct output

  est <- colMeans(res)
  sdd <- apply(res, 2, sd)

  fnr <- mean(err[,1])
  fpr <- mean(err[,2])
  fnr.sd <- sd(err[,1])
  fpr.sd <- sd(err[,2])


  d0 = data.frame("Estimate" = est,  "Std. Error" = sdd, check.names = FALSE)
  d = rbind(d0, data.frame("Estimate" = c(fnr, fpr),  "Std. Error" = c(fnr.sd, fpr.sd), check.names = FALSE))
  # return coefficient estimates as well as raw res
  if (return_boot_samples ){
    return (list(resamples = d, raw_error = err))
    }else{
  return (d)
    }
}



# Bayesian ------------
## Bayesian logistic regression -------------

#' Bayesian estimator for logistic regression with binary classification as *outcome* variable
#'
#' This functions takes binary classification as *outcome* variables, apply Bayesianestimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @param fp number of false positives in validation data
#' @param fn number of false negatives in validation data
#' @param tp number of true positives in validation data
#' @param tn number of true negatives in validation data
#' @param tau_prior prior values of hyperparameters of Bayesian regression variance
#' @param working_dir directory to store JAGS temporary files; default is the same directory of the script
#' @param n.iter number of Bayesian draws
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
bayesian_logistic <- function(formula, data, fn = fn, fp = fp, tp = tp, tn = tn, fpr = fpr, fnr = fnr, tau_prior = 0.001, working_dir = "", n.iter = n.iter){

  cmodel <- function(){

    ## original setup
    # for( i in 1 : N) {
    #   y[i] ~ dbern(q[i])
    #   # q[i] <- p[i]*(1-lambda[1])+(1-p[i])*lambda[2]
    #   q[i] <- p[i]*(1-lambda[1] - lambda[2]) + lambda[2]
    #
    #   logit(p[i]) <- inprod(x[i,], beta[])
    # }

    ## new setup
    for( i in 1 : N) {
      y[i] ~ dbern(q[i])
      ## one sampling approach
      q[i] <- p[i]*(1-lambda[1] - lambda[2]) + lambda[2]

      ## alternative sampling approach
      # q[i] <- (( 1 - lambda[1])  * ty[i]) + (lambda[2]  * (1 - ty[i]))
      # ty[i] ~ dbern(p[i]) ## true y's probability

      logit(p[i]) <- inprod(x[i,], beta[])
    }

    # hieaarchical prior
    # beta[1:num_param] ~ dmnorm(mea[], preci[,])
    # for( k in 1:num_param){
    #   # mea[k] ~ dnorm(0, 0.0001)
    # mea[k] <- beta_guess[k]
    # }
    # for( k in 1: num_param){
    #   for (j in 1:num_param){
    #     preci[k,j] <- eta * tau * invx[k, j]
    #   }
    # }
    # tau ~ dgamma(tau_prior, tau_prior)

    # standard logistic
    for( l in 1:num_param){
      beta[l] ~ dnorm(0, tau)
    }

    tau ~ dgamma(tau_prior, tau_prior)

    # fpr/fnr estimate
    lambda[1] ~ dbeta(fn , tp ) ## FNR
    lambda[2] ~ dbeta(fp , tn ) ## FPR



    ### or using another approach:

    # for (j in 1:2){
    #   lambda[j] ~ dnorm(fr[j], 0.001)
    # }
    #
    # for (j in 1:2)
    # {
    #   fr[j] <-  co[(j-1)*2 +1] / (co[(j-1)*2 +1] + co[(j-1)*2 +2])
    # }
    #
    # co[1:4] ~ ddirch(confusion[])
    # confusion <- c(fn, tp, fp, tn)

  }


  ## get an initial model (to obtain data matrix)
  ps = na.omit(data)
  initModel = lm(formula, data = ps)
  mX = as.matrix(model.matrix(initModel))
  vY = model.frame(initModel)[,1]

  MLE <- MLE_logit(formula, ps, fnr = fnr, fpr = fpr )

  vBetfnr = MLE[,1]

  # invx = solve(t(mX) %*% mX)

  # dat <- list(y = vY, x = mX, fn = fn, fp = fp, tp = tp, tn = tn, N = nrow(ps), num_param = ncol(mX), tau_prior = tau_prior, invx = invx, beta_guess = vBetfnr)
  dat <- list(y = vY, x = mX, fn = fn, fp = fp, tp = tp, tn = tn, N = nrow(ps), num_param = ncol(mX), tau_prior = tau_prior)


  # parameter initialization
  param_inits <- function(){
    list(lambda=c(fnr, fpr), beta = vBetfnr)
  }

  param <- c("lambda",  "beta")

  ## to use autojags one has to have at least 2 chains
  # otherwise the Gelman-Rubin statistics cannot be calculated
  out1 <- R2jags::jags(data = dat, inits = param_inits, parameters.to.save = param, model.file = cmodel, n.chains = 2, n.burnin = 200,  n.iter = 400,  working.directory = working_dir)
  out <- R2jags::autojags(out1, n.iter = n.iter)

  p1 <- coda::as.mcmc(out)
  p2 <- summary(p1)

  df1 = data.frame(p2$statistics[, c(1,2)])
  names(df1) <- c("Estimate",  "Std. Error")

  return (list(bayesian = df1, summary = p2, raw = p1 ))
}






## Bayesian linear regression for predicted independent variable ----------------

#' Bayesian estimator for linear with binary classification as *independent* variable
#'
#' This functions takes binary classification as *independent* variables, apply Bayesian estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @param fp number of false positives in validation data
#' @param fn number of false negatives in validation data
#' @param tp number of true positives in validation data
#' @param tn number of true negatives in validation data
#' @param predicted_col_name string or integer indicating the column of the variable that is predicted
#' @param tau_prior prior values of hyperparameters of Bayesian regression variance
#' @param working_dir directory to store JAGS temporary files; default is the same directory of the script
#' @param n.iter number of Bayesian draws
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
bayesian_linear_indep <- function(formula, data, fn = fn, fp = fp, tp = tp, tn = tn, npv = npv, ppv = ppv, predicted_col_name = "xeb", tau_prior = 0.001,  working_dir = "", n.iter = n.iter){


  cmodel <- function(){
    for( i in 1 : N) {
      y[i] ~ dnorm(q[i], tau)

      # # this will make error on the W estimate?
      # q[i] <- x[i,2] * (e1[i]* lambda[1] + e0[i] * (1 - lambda[1])) +
      #   (1 - x[i,2]) * (e1[i]* (1 - lambda[2]) + e0[i] * lambda[2])
      #
      # e1 is the case of assuming predicted X (key control) is 1
      # but need some rewriting here
      # also it may impact other settings
      # e1[i] <- beta[1] + beta[2]  + beta[3]*x[i,3]
      # e0[i] <- beta[1] + beta[3]*x[i,3]
      #
      ## new scripts (testing)
      # # this will make error on the W estimate?
      q[i] <- ind[i] * (e1[i]* lambda[1] + e0[i] * (1 - lambda[1])) +
        (1 - ind[i]) * (e1[i]* (1 - lambda[2]) + e0[i] * lambda[2])

      e1[i] <- inprod(x1[i,], beta[])
      e0[i] <- inprod(x0[i,], beta[])
    }

    # empirical bayes for setting priors
    lambda[1] ~ dbeta(tp, fp)
    lambda[2] ~ dbeta(tn, fn)

    # for (j in 1:2){
    #   lambda[j] ~ dnorm(fr[j], 0.001)
    # }
    #
    # for (j in 1:2)
    # {
    #   fr[j] <-  co[(j-1)*2 +1] / (co[(j-1)*2 +1] + co[(j-1)*2 +2])
    # }
    #
    # co[1:4] ~ ddirch(confusion[])
    # # confusion <- c(fn, tp, fp, tn)
    # confusion <- c(tp, fp, tn, fn)


    # noninformative prior for regression coefficients
    for( l in 1:num_param){
      beta[l] ~ dnorm(0, tau)
    }

    ## not using Gamma prior but instead use sigma^2 prior
    tau ~ dgamma(tau_prior, tau_prior)


    # sigma ~ dunif(0, 10) # standard deviation
    # tau <- 1 / (sigma * sigma) # sigma^2 doesn't work in JAGS
    #
  }

  ps = na.omit(data)
  initModel = lm(formula, data = ps)
  mX = as.matrix(model.matrix(initModel))
  vY = model.frame(initModel)[,1]

  MLE <- MLE_linear(formula, ps, predicted_col_name =  predicted_col_name, ppv = ppv, npv = npv)

  vBetfnr = MLE[,1]

  # invx = solve(t(mX) %*% mX)

  mX1 = mX
  mX0 = mX
  mX1[,predicted_col_name]<-1
  mX0[,predicted_col_name]<-0
  ind <- as.integer(mX[,predicted_col_name]==1)

  ## the idea is to create two artificial columns
  ## one assuming that the predicted column is all 1
  ## the other assuming that the predicted column is all 0
  ## then pass these two dataframes into the data for Bayesian

  dat <- list(y = vY,  x1 = mX1, x0=mX0, ind = ind, fn = fn, fp = fp, tp = tp, tn = tn, N = nrow(ps),num_param = ncol(mX), tau_prior = tau_prior)


  # parameter initialization
  param_inits <- function(){
    list(lambda=c(ppv, npv), beta = vBetfnr)
    # list(lambda=c(ppv, npv), beta = rep(0, ncol(mX)))
  }

  param <- c("lambda",  "beta")

  ## to use autojags one has to have at least 2 chains
  # otherwise the Gelman-Rubin statistics cannot be calculated
  out1 <- R2jags::jags(data = dat, inits = param_inits, parameters.to.save = param, model.file = cmodel, n.chains = 2, n.burnin = 200,  n.iter = 1500,  working.directory = working_dir)

  out <- R2jags::autojags(out1, n.iter = n.iter)

  p1 <- coda::as.mcmc(out)

  p2 <- summary(p1)

  df1 = data.frame(p2$statistics[, c(1,2)])
  names(df1) <- c("Estimate",  "Std. Error")

  return (list(bayesian = df1, summary = p2, raw = p1 ))
}


## Bayesian linear regression for binary outcome (i.e., Linear Probability Model )   -------------

#' Bayesian estimator for linear regression with binary classification as *dependent* variable
#'
#' This functions takes binary classification as *dependent* variables for linear probability  model, apply Bayesian estimator, and returned corrected coefficient estimates.
#'
#' @param formula R formula, such as y ~ x
#' @param data a data.frame object
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @param fp number of false positives in validation data
#' @param fn number of false negatives in validation data
#' @param tp number of true positives in validation data
#' @param tn number of true negatives in validation data
#' @param tau_prior prior values of hyperparameters of Bayesian regression variance
#' @param working_dir directory to store JAGS temporary files; default is the same directory of the script
#' @param n.iter number of Bayesian draws
#' @return a dataframe contining estimated values, standard errors, z values and p values
#' @export
bayesian_LPM <- function(formula, data, fn = fn, fp = fp, tp = tp, tn = tn, fpr = fpr, fnr = fnr, tau_prior = 0.001, working_dir = "", n.iter = n.iter){

  cmodel <- function(){

    ## new setup
    for( i in 1 : N) {
      y[i] ~ dbern(q[i])

      q[i] <- w[i] * (1-lambda[1] - lambda[2]) + lambda[2]

      w[i] <- ifelse(u[i]>1, 0.99999, u[i])
      u[i] <- ifelse(p[i]<0, 0.00001, p[i])

      p[i] <- inprod(x[i,], beta[])
    }

    # standard logistic
    for( l in 1:num_param){
      beta[l] ~ dnorm(0, tau)
    }

    tau ~ dgamma(tau_prior, tau_prior)

    # fpr/fnr estimate
    lambda[1] ~ dbeta(fn , tp ) ## FNR
    lambda[2] ~ dbeta(fp , tn ) ## FPR

  }


  ## get an initial model (to obtain data matrix)
  ps = na.omit(data)
  initModel = lm(formula, data = ps)
  mX = as.matrix(model.matrix(initModel))
  vY = model.frame(initModel)[,1]

  MLE <- MLE_LPM(formula, ps, fnr = fnr, fpr = fpr )

  vBetfnr = MLE[,1]

  dat <- list(y = vY, x = mX, fn = fn, fp = fp, tp = tp, tn = tn, N = nrow(ps), num_param = ncol(mX), tau_prior = tau_prior)


  # parameter initialization
  param_inits <- function(){
    list(lambda=c(fnr, fpr), beta = vBetfnr)
  }

  param <- c("lambda",  "beta")

  ## to use autojags one has to have at least 2 chains
  # otherwise the Gelman-Rubin statistics cannot be calculated
  out1 <- jags(data = dat, inits = param_inits, parameters.to.save = param, model.file = cmodel, n.chains = 2, n.burnin = 200,  n.iter = 400,  working.directory = working_dir)
  out <- autojags(out1, n.iter = n.iter)

  p1 <- as.mcmc(out)
  p2 <- summary(p1)

  df1 = data.frame(p2$statistics[, c(1,2)])
  names(df1) <- c("Estimate",  "Std. Error")

  return (list(bayesian = df1, summary = p2, raw = p1 ))
}



## validation from error rates
#' Produce a list of pseudo-true labels / predictions
#' used for Bootstrap estimator, if only FNR/FPR are known

#'
#' @param n Size of intended validation dataset
#' @param mu_pred mean of positive class in the predicted labels
#' @param fnr false negative rate
#' @param fpr false positive rate
#' @return a list, the first element is a vector containing true labels, the second element is a vector containing predicted labels.
#' @export
produce_valdata_from_errorrate <- function(n, mu_pred, fpr, fnr)
{

  mu <- (mu_pred - fpr) / ( 1-fpr-fnr)
  num1 <- round (n*mu) ## number of positive in truth
  num0 <- n - num1 ## number of negative in truth

  tl1 <- rep(1, num1)
  tl0<- rep(0, n - num1)

  ## then flip labels based on FNR/FPR
  fpc <- round (num0 * fpr)
  fnc <- round (num1 * fnr)

  pl0 <- c(rep(1, fpc), rep(0, num0 - fpc))
  pl1 <- c(rep(0, fnc), rep(1, num1 - fnc))

  return (list(tl =c(tl1, tl0), pl = c(pl1, pl0) ))

}
