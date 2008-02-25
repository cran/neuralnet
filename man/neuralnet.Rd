\name{neuralnet}
\alias{neuralnet}
\alias{print.nn}
\title{Training of neural networks}
\description{
    \code{neuralnet} is used to train neural networks using the Resilient Backpropagation with (Riedmiller, 1994) or without 
    weightbacktracking (Riedmiller and Braun, 1993) or the modified
    globally convergent version by Anastasiadis et al. (2005). The
    function allows flexible settings through custom-choice of error-
    and activation-function. Furthermore the calculation of 
    generalized weights (Intrator O. and Intrator N., 1993) is
    implemented.
}
\details{
If family is stated, a glm model will additionally be calculated for comparison reasons, but only if the response consists of only one parameter.
The globally convergent algorithm is based on the Resilient Backpropagation without weightbacktracking and additionally modifies one learning rate, either the learningrate associated with the smallest absolute gradient (sag) or the smallest learningrate (slr) itself. The learning rates in the grprop algorithm are limited to the boundaries defined in learningrate.limit. 
}
\usage{
neuralnet(formula, data, hidden = 1, threshold = c(0.001),
    stepmax = 1e+05, rep = 1, weights.mean = 0, weights.variance = 1,
    startweights = NULL, learningrate.limit = NULL, 
    learningrate.factor = list(minus = 0.5, plus = 1.2), lifesign = "none",
    lifesign.step = 1000, algorithm = "rprop+", err.fct = "sse", 
    act.fct = "logistic", linear.output=TRUE, family = NULL)
}
\arguments{
  \item{formula}{ a symbolic description of the model to be fitted. }
  \item{data}{ a data frame in which the variables specified in \code{formula} will be found.  }
  \item{hidden}{ a vector of integers specifying the number of hidden neurons (vertices) in each layer. }
  \item{threshold}{ a vector of integers for the threshold of the minimal error. Threshold as a vector of size n, trains n different repetitions of the neural network with a component wise threshold.  }
  \item{stepmax}{ the maximum steps for the training of the neural network. Reaching this maximum leads to a stop of the neural network's training process. }
  \item{rep}{ the number of repetitions for the neural network's training for every threshold. }
  \item{weights.mean}{ the mean of the normal distribution, the initial weights are drawn from. }
  \item{weights.variance}{ the variance of the normal distribution, the initial weights are drawn from. }
  \item{startweights}{ a vector containing starting values for the weights. The weights will not be randomly initialized.}
  \item{learningrate.limit}{ a vector or a list containing the lowest and highest limit for the learning rate. }
  \item{learningrate.factor}{ a vector or a list containing the multiplication factors for the upper and lower learning rate. }
  \item{lifesign}{ a string specifying how much the function will print during the calculation of the neural network. 'none', 'minimal' or 'full'. }
  \item{lifesign.step}{ an integer specifying the stepsize to print the minimal threshold in full lifesign mode.  }
  \item{algorithm}{ a string containing the algorithm type to calculate the neural network. The  following types are possible: 'rprop+', 'rprop-', 'sag', or 'slr'. 'rprop+' and 'rprop-' refer to  the Resilient Backpropagation with and without weightbacktracking,
 while 'sag' and 'slr' induce the usage of the modified globally convergent algorithm (grprop). See Details for more information.}
  \item{err.fct}{ a differentiable function that is used for the calculation of the error. Alternatively, the strings 'sse' and 'ce' which stand for the sum of squared errors and the cross-entropy can be used.}
  \item{act.fct}{ a differentiable function that is used for smoothing the result of the cross product of the covariate or neurons and the weights. Additionally the strings, 'logistic' and 'tanh' are possible for the logistic function and tangent hyperbolicus. }
  \item{linear.output}{ logical. If act.fct should not be applied to the output neurons set linear output to TRUE, otherwise to FALSE. }
  \item{family}{ a description of the error distribution and link function to be used only in the glm model. This can be a character string naming a family function, a family function or the result of a call to a family function.}
}
\value{
  \code{neuralnet} returns an object of class \code{nn}.
  An object of class \code{nn} is a list containing at most the following components:

  \item{ call }{ the matched call. }
  \item{ response }{ extracted from the \code{data argument}.  }
  \item{ covariate }{ the variables extracted from the \code{data argument}. }
  \item{ model.list }{ a list containing the covariates and the response variables extracted from the \code{formula argument}. }
  \item{ err.fct }{ the error function. }
  \item{ act.fct }{ the activation function. }
  \item{ data }{ the \code{data argument}.}
  \item{ net.result }{ a list containing the overall result of the neural network for every repetition.}
  \item{ weights }{ a list containing the fitted weights of the neural network for every repetition. }
  \item{ gw }{ a list containing the generalized weights of the neural network for every repetition. }
  \item{ result.matrix }{ a matrix containing the threshold, reached threshold, steps, error, aic (if computed) and weights for every repetition. Each column represents one repetition. }
  \item{ list.glm }{ a list of glm objects. It will be set to NULL if there is more than one response or family is not stated.}
  \item{ predictions }{ a list of the predictions of the repetitions, the data and the glm models. It will not be computed if at least one of the covariates has more than 50 factors.}
  \item{ data.error }{ the error of the data. This is 0, if no row in covariate has a duplicate. It will not be computed if at least one of the covariates has more than 50 factors.}
}
\references{ 
    Riedmiller M. (1994) 
    \emph{Rprop - Description and Implementation Details.}
    Technical Report. University of Karlsruhe.

    Riedmiller M. and Braun H. (1993) 
    \emph{A direct adaptive method for faster backpropagation learning: The RPROP algorithm.}
    Proceedings of the IEEE International Conference on Neural Networks (ICNN), pages 586-591.
    San Francisco.

    Anastasiadis A. et. al. (2005) 
    \emph{New globally convergent training scheme based on the resilient propagation algorithm.} 
    Neurocomputing 64, pages 253-270. 

    Intrator O. and Intrator N. (1993)
    \emph{Using Neural Nets for Interpretation of Nonlinear Models.}  
    Proceedings of the Statistical Computing Section, 244-249 
    San Francisco: American Statistical Society (eds). 
}
\author{ Stefan Fritsch \email{fritsch@bips.uni-bremen.de} }

\seealso{\code{\link{plot.nn}} for plotting of the neural network.
 \code{\link{gwplot}} for plotting of the generalized weights.
 \code{\link{compute}} for computation of the calculated network.
}
\examples{
AND <- c(rep(0,7),1)
OR <- c(0,rep(1,7))
SUM <- c(0,1,1,2,1,2,2,3)
binary.data <- data.frame(expand.grid(c(0,1), c(0,1), c(0,1)), AND, OR, SUM)
print(net <- neuralnet( AND+OR~Var1+Var2+Var3,  binary.data, hidden=0, rep=10,
	     err.fct="ce", linear.output=FALSE))
print(net.sum <- neuralnet( SUM~Var1+Var2+Var3,  binary.data, hidden=0, 
		 linear.output=TRUE))
net.sum$predictions

data(infert, package="datasets")
print(net.infert <- neuralnet( case~parity+induced+spontaneous,  infert, 
		    err.fct="ce", linear.output=FALSE, family=binomial()))
}
\keyword{ neural }
