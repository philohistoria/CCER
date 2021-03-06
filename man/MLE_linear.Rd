% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/ccer.r
\name{MLE_linear}
\alias{MLE_linear}
\title{MLE estimator for linear regression with binary classification as \emph{independent} variables}
\usage{
MLE_linear(formula, data, predicted_col_name, ppv = 0, npv = 0)
}
\arguments{
\item{formula}{R formula, such as y ~ x}

\item{data}{a data.frame object}

\item{predicted_col_name}{string or integer indicating the column of the variable that is predicted}

\item{ppv}{positive predicted value}

\item{npv}{negative predicted value}
}
\value{
a dataframe contining estimated values, standard errors, z values and p values
}
\description{
This functions takes binary classification as \emph{independent} variables, apply MLE estimator, and returned corrected coefficient estimates.
}
