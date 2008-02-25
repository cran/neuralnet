`compute` <-
function (x, covariate, rep = 1) 
{
    nn <- x
    weights <- nn$weights[[rep]]
    linear.output <- nn$linear.output
    length.weights <- length(weights)
    covariate <- as.matrix(cbind(1, covariate))
    act.fct <- nn$act.fct
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    if (linear.output) 
        net.result <- temp
    else net.result <- act.fct(temp)
    list(neurons = neurons, net.result = net.result)
}
