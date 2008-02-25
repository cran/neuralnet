`calculate.data.result` <-
function (response, covariate, model.list) 
{
    duplicated <- duplicated(covariate)
    if (!any(duplicated)) {
        return(response)
    }
    which.duplicated <- seq_along(duplicated)[duplicated]
    which.not.duplicated <- seq_along(duplicated)[!duplicated]
    if (ncol(response) == 1) {
        for (each in which.not.duplicated) {
            out <- NULL
            if (any(which.duplicated)) 
                for (j in 1:length(which.duplicated)) {
                  if (all(covariate[which.duplicated[j], ] == 
                    covariate[each, ])) {
                    out <- c(out, j)
                  }
                }
            if (!is.null(out)) {
                rows <- c(each, which.duplicated[out])
                response[rows] = mean(response[rows])
                which.duplicated <- which.duplicated[-out]
            }
        }
    }
    else {
        for (each in which.not.duplicated) {
            out <- NULL
            if (any(which.duplicated)) 
                for (j in 1:length(which.duplicated)) {
                  if (all(covariate[which.duplicated[j], ] == 
                    covariate[each, ])) {
                    out <- c(out, j)
                  }
                }
            if (!is.null(out)) {
                response[each, ] = colMeans(response[c(each, 
                  which.duplicated[out]), ])
                for (k in 1:length(out)) response[which.duplicated[k], 
                  ] = response[each, ]
                which.duplicated <- which.duplicated[-out]
            }
        }
    }
    response
}
`calculate.gradients` <-
function (weights, length.weights, neurons, neuron.deriv, err.deriv) 
{
    if (any(is.na(err.deriv))) 
        stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
            call. = FALSE)
    delta <- neuron.deriv[[length.weights]] * err.deriv
    gradients <- crossprod(neurons[[length.weights]], delta)
    if (length.weights > 1) 
        for (w in (length.weights - 1):1) {
            delta <- neuron.deriv[[w]] * tcrossprod(delta, remove.intercept(weights[[w + 
                1]]))
            gradients <- c(crossprod(neurons[[w]], delta), gradients)
        }
    gradients
}
`calculate.gradients.linear.output` <-
function (weights, length.weights, neurons, neuron.deriv, err.deriv) 
{
    if (any(is.na(err.deriv))) 
        stop("the error derivative contains a NA; varify that the derivative function does not divide by 0 (e.g. cross entropy)", 
            call. = FALSE)
    delta <- err.deriv
    gradients <- crossprod(neurons[[length.weights]], delta)
    if (length.weights > 1) 
        for (w in (length.weights - 1):1) {
            delta <- neuron.deriv[[w]] * tcrossprod(delta, remove.intercept(weights[[w + 
                1]]))
            gradients <- c(crossprod(neurons[[w]], delta), gradients)
        }
    gradients
}
`calculate.gw` <-
function (weights, neuron.deriv, net.result) 
{
    for (w in 1:length(weights)) {
        weights[[w]] <- remove.intercept(weights[[w]])
    }
    gw <- NULL
    for (k in 1:ncol(net.result)) {
        for (w in length(weights):1) {
            if (w == length(weights)) {
                temp <- neuron.deriv[[length(weights)]][, k] * 
                  1/(net.result[, k] * (1 - (net.result[, k])))
                delta <- tcrossprod(temp, weights[[w]][, k])
            }
            else {
                delta <- tcrossprod(delta * neuron.deriv[[w]], 
                  weights[[w]])
            }
        }
        gw <- cbind(gw, delta)
    }
    return(gw)
}
`calculate.neuralnet` <-
function (data, model.list, hidden, stepmax, rep, threshold, 
    weights.mean, weights.variance, learningrate.limit, learningrate.factor, 
    lifesign, covariate, response, lifesign.step, startweights, 
    algorithm, act.fct, act.deriv.fct, err.fct, err.deriv.fct, 
    linear.output) 
{
    time.start.local <- Sys.time()
    weights <- generate.initial.weights(model.list, hidden, startweights, 
        rep, weights.mean, weights.variance)
    result <- rprop(weights = weights, threshold = threshold, 
        response = response, covariate = covariate, learningrate.limit = learningrate.limit, 
        learningrate.factor = learningrate.factor, stepmax = stepmax, 
        lifesign = lifesign, lifesign.step = lifesign.step, act.fct = act.fct, 
        act.deriv.fct = act.deriv.fct, err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
        algorithm = algorithm, linear.output = linear.output)
    weights <- result$weights
    step <- result$step
    reached.threshold <- result$reached.threshold
    net.result <- result$net.result
    aic <- NULL
    if (type(err.fct) == "ce" && all((response == 1 | response == 
        0))) {
        if (all(net.result <= 1, net.result >= 0)) {
            synapse.count <- sum(sapply(weights, length))
            error <- sum(err.fct(net.result, response), na.rm = T)
            aic <- 2 * error + (2 * synapse.count)
        }
        else {
            aic <- NA
            error <- sum(err.fct(net.result, response))
        }
    }
    else error <- sum(err.fct(net.result, response))
    if (is.na(error)) 
        warning("'err.fct' does not fit 'data' or 'act.fct'", 
            call. = F)
    if (lifesign != "none") {
        if (step < stepmax) {
            cat(rep(" ", (max(nchar(stepmax), nchar("stepmax")) - 
                nchar(step))), step, sep = "")
            cat("\terror: ", round(error, 5), rep(" ", 6 - (nchar(round(error, 
                5)) - nchar(round(error, 0)))), sep = "")
            if (!is.null(aic)) {
                cat("\taic: ", round(aic, 5), rep(" ", 6 - (nchar(round(aic, 
                  5)) - nchar(round(aic, 0)))), sep = "")
            }
            cat("\ttime: ", difftime(Sys.time(), time.start.local), 
                sep = "")
            cat("\n")
        }
    }
    if (step == stepmax) 
        return(result = list(output.vector = NULL, weights = NULL))
    output.vector <- c(threshold = threshold, reached.threshold = reached.threshold, 
        steps = step, error = error)
    if (!is.null(aic)) {
        output.vector <- c(output.vector, aic = aic)
    }
    for (w in 1:length(weights)) output.vector <- c(output.vector, 
        as.vector(weights[[w]]))
    generalized.weights <- calculate.gw(weights, neuron.deriv = result$neuron.deriv, 
        net.result = net.result)
    return(list(generalized.weights = generalized.weights, weights = weights, 
        net.result = result$net.result, output.vector = output.vector))
}
`calculate.predictions` <-
function (covariate, data.result, list.glm = NULL, matrix, list.net.result, 
    model.list, act.fct) 
{
    not.duplicated <- !duplicated(covariate)
    nrow.notdupl <- sum(not.duplicated)
    covariate.mod <- matrix(covariate[not.duplicated, ], nrow = nrow.notdupl)
    predictions <- list(data = cbind(covariate.mod, matrix(data.result[not.duplicated, 
        ], nrow = nrow.notdupl)))
    if (!is.null(matrix)) {
        for (i in length(list.net.result):1) {
            pred.temp <- cbind(covariate.mod, matrix(list.net.result[[i]][not.duplicated, 
                ], nrow = nrow.notdupl))
            predictions <- eval(parse(text = paste("c(list(rep", 
                i, "=pred.temp), predictions)", sep = "")))
        }
    }
    if (!is.null(list.glm)) {
        for (i in 1:length(list.glm)) {
            pred.temp <- cbind(covariate.mod, matrix(act.fct(list.glm[[i]]$linear.predictors)[not.duplicated], 
                nrow = nrow.notdupl))
            text <- paste("c(predictions, list(list.glm.", names(list.glm[i]), 
                "=pred.temp))", sep = "")
            predictions <- eval(parse(text = text))
        }
    }
    for (i in 1:length(predictions)) {
        colnames(predictions[[i]]) <- c(model.list$variables, 
            model.list$response)
        if (nrow(covariate) > 1) 
            for (j in (1:ncol(covariate))) predictions[[i]] <- predictions[[i]][order(predictions[[i]][, 
                j]), ]
        rownames(predictions[[i]]) <- 1:nrow(predictions[[i]])
    }
    predictions
}
`compute.net` <-
function (weights, length.weights, covariate, act.fct, act.deriv.fct, 
    output.act.fct, output.act.deriv.fct) 
{
    neuron.deriv <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neuron.deriv[[i]] <- act.deriv.fct(temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    if (!is.list(neuron.deriv)) 
        neuron.deriv <- list(neuron.deriv)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    neuron.deriv[[length.weights]] <- output.act.deriv.fct(temp)
    if (any(is.na(neuron.deriv))) 
        stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
            call. = FALSE)
    list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result)
}
`compute.net.special` <-
function (weights, length.weights, covariate, act.fct, act.deriv.fct, 
    output.act.fct, output.act.deriv.fct) 
{
    neuron.deriv <- NULL
    neurons <- list(covariate)
    if (length.weights > 1) 
        for (i in 1:(length.weights - 1)) {
            temp <- neurons[[i]] %*% weights[[i]]
            act.temp <- act.fct(temp)
            neuron.deriv[[i]] <- act.deriv.fct(act.temp)
            neurons[[i + 1]] <- cbind(1, act.temp)
        }
    if (!is.list(neuron.deriv)) 
        neuron.deriv <- list(neuron.deriv)
    temp <- neurons[[length.weights]] %*% weights[[length.weights]]
    net.result <- output.act.fct(temp)
    neuron.deriv[[length.weights]] <- output.act.deriv.fct(net.result)
    if (any(is.na(neuron.deriv))) 
        stop("neuron derivatives contain a NA; varify that the derivative function does not divide by 0", 
            call. = FALSE)
    list(neurons = neurons, neuron.deriv = neuron.deriv, net.result = net.result)
}
`differentiate` <-
function (orig.fct) 
{
    body.fct <- deparse(body(orig.fct))
    if (body.fct[1] == "{") 
        body.fct <- body.fct[2]
    text <- paste("y~", body.fct, sep = "")
    text2 <- paste(deparse(orig.fct)[1], "{}")
    temp <- deriv(eval(parse(text = text)), "x", func = eval(parse(text = text2)))
    temp <- deparse(temp)
    derivative <- NULL
    for (i in 1:length(temp)) {
        if (!any(grep("value", temp[i]))) 
            derivative <- c(derivative, temp[i])
    }
    derivative[length(derivative) - 1] <- unlist(strsplit(derivative[length(derivative) - 
        1], "<-"))[2]
    derivative <- eval(parse(text = derivative))
    return(derivative)
}
`display` <-
function (hidden, threshold, i.thr, rep, i.rep, lifesign) 
{
    text <- paste("%", max(nchar(threshold)) - nchar(threshold[i.thr]), 
        "s    rep: %", nchar(rep), "s", sep = "")
    cat("hidden: ", paste(hidden, collapse = ", "), "    thresh: ", 
        threshold[i.thr], sprintf(eval(expression(text)), "", 
            i.rep), "/", rep, "    steps: ", sep = "")
    if (lifesign == "full") 
        lifesign <- sum(nchar(hidden)) + 2 * length(hidden) - 
            2 + max(nchar(threshold)) + 2 * nchar(rep) + 41
    return(lifesign)
}
`generate.initial.variables` <-
function (data, model.list, hidden, act.fct, err.fct, algorithm, 
    linear.output) 
{
    covariate <- NULL
    input.count <- length(model.list$variables)
    output.count <- length(model.list$response)
    response <- matrix(0, nrow(data), output.count)
    for (i in 1:output.count) {
        if (!any(colnames(data) == model.list$response[[i]])) {
            message <- sprintf("column %s does not exist in 'data'", 
                sQuote(model.list$response[[i]]))
            stop(message, call. = FALSE)
        }
        response[, i] <- eval(parse(text = paste("data$", model.list$response[[i]], 
            sep = "")))
    }
    names <- c(0)
    covariate <- matrix(0, nrow(data), (input.count + 1))
    covariate[, 1] <- 1
    for (i in 1:input.count) {
        if (!any(colnames(data) == model.list$variables[[i]])) {
            message <- sprintf("column %s does not exist in 'data'", 
                sQuote(model.list$variables[[i]]))
            stop(message, call. = FALSE)
        }
        covariate[, i + 1] <- eval(parse(text = paste("data$", 
            model.list$variables[[i]], sep = "")))
        if (is.factor(eval(parse(text = paste("data$", model.list$variables[[i]], 
            sep = ""))))) 
            warning(sprintf("factor %s will be interpreted as a numeric variable", 
                sQuote(model.list$variables[[i]])), call. = FALSE)
    }
    names <- c(0)
    names[1] <- "Intercept"
    for (i in 2:(input.count + 1)) names[i] <- paste("covariate", 
        i - 1, sep = "")
    colnames(covariate) <- names
    pred <- T
    for (i in 2:ncol(covariate)) {
        if (length(levels(as.factor(covariate[, i]))) > 50) 
            pred <- F
    }
    if (!pred) 
        warning("'predictions' will not be calculated, as at least one covariate contains more than 50 different values", 
            call. = FALSE)
    if (is.function(act.fct)) {
        act.deriv.fct <- differentiate(act.fct)
        attr(act.fct, "type") <- "function"
        if (length(act.deriv.fct(c(1, 1))) == 1) {
            act.deriv.fct <- eval(parse(text = paste("function(x){matrix(", 
                act.deriv.fct(1), ", nrow(x), ncol(x))}")))
        }
    }
    else {
        if (act.fct == "tanh") {
            act.fct <- function(x) {
                tanh(x)
            }
            attr(act.fct, "type") <- "tanh"
            act.deriv.fct <- function(x) {
                1 - x^2
            }
        }
        else if (act.fct == "logistic") {
            act.fct <- function(x) {
                1/(1 + exp(-x))
            }
            attr(act.fct, "type") <- "logistic"
            act.deriv.fct <- function(x) {
                x * (1 - x)
            }
        }
    }
    if (is.function(err.fct)) {
        err.deriv.fct <- differentiate(err.fct)
        attr(err.fct, "type") <- "function"
    }
    else {
        if (err.fct == "ce") {
            if (all(response == 0 | response == 1)) {
                err.fct <- function(x, y) {
                  -(y * log(x) + (1 - y) * log(1 - x))
                }
                attr(err.fct, "type") <- "ce"
                err.deriv.fct <- function(x, y) {
                  (1 - y)/(1 - x) - y/x
                }
            }
            else {
                err.fct <- function(x, y) {
                  1/2 * (x - y)^2
                }
                attr(err.fct, "type") <- "sse"
                err.deriv.fct <- function(x, y) {
                  x - y
                }
                warning("'err.fct' was automatically set to sum of squared error (sse), because the response is not binary", 
                  call. = F)
            }
        }
        else if (err.fct == "sse") {
            err.fct <- function(x, y) {
                1/2 * (x - y)^2
            }
            attr(err.fct, "type") <- "sse"
            err.deriv.fct <- function(x, y) {
                x - y
            }
        }
    }
    return(list(covariate = covariate, response = response, pred = pred, 
        err.fct = err.fct, err.deriv.fct = err.deriv.fct, act.fct = act.fct, 
        act.deriv.fct = act.deriv.fct, algorithm = algorithm))
}
`generate.initial.weights` <-
function (model.list, hidden, startweights, rep, weights.mean, 
    weights.variance) 
{
    input.count <- length(model.list$variables)
    output.count <- length(model.list$response)
    if (!(length(hidden) == 1 && hidden == 0)) {
        length.weights <- length(hidden) + 1
        nrow.weights <- array(0, dim = c(length.weights))
        ncol.weights <- array(0, dim = c(length.weights))
        nrow.weights[1] <- (input.count + 1)
        ncol.weights[1] <- hidden[1]
        if (length(hidden) > 1) 
            for (i in 2:length(hidden)) {
                nrow.weights[i] <- hidden[i - 1] + 1
                ncol.weights[i] <- hidden[i]
            }
        nrow.weights[length.weights] <- hidden[length.weights - 
            1] + 1
        ncol.weights[length.weights] <- output.count
    }
    else {
        length.weights <- 1
        nrow.weights <- array((input.count + 1), dim = c(1))
        ncol.weights <- array(output.count, dim = c(1))
    }
    length <- sum(ncol.weights * nrow.weights)
    if (is.null(startweights) || length(startweights) < (length * 
        rep)) 
        vector <- rnorm(length, weights.mean, weights.variance)
    else vector <- startweights[((rep - 1) * length + 1):(length * 
        rep)]
    weights <- relist(vector, nrow.weights, ncol.weights)
    return(weights)
}
`generate.output` <-
function (covariate, call, rep, threshold, matrix, startweights, 
    model.list, response, err.fct, act.fct, data, family, pred, 
    list.result, linear.output) 
{
    covariate <- t(remove.intercept(t(covariate)))
    nn <- list(call = call)
    class(nn) <- c("nn")
    nn$response <- response
    nn$covariate <- covariate
    nn$model.list <- model.list
    nn$err.fct <- err.fct
    nn$act.fct <- act.fct
    nn$linear.output <- linear.output
    nn$data <- data
    if (!is.null(matrix)) {
        nn$net.result <- NULL
        nn$weights <- NULL
        nn$gw <- NULL
        for (i in 1:length(list.result)) {
            nn$net.result <- c(nn$net.result, list(list.result[[i]]$net.result))
            nn$weights <- c(nn$weights, list(list.result[[i]]$weights))
            nn$gw <- c(nn$gw, list(list.result[[i]]$generalized.weights))
        }
        nn$result.matrix <- generate.rownames(matrix, nn$weights[[1]], 
            model.list)
    }
    if (length(model.list$response) == 1) {
        if (!is.null(family)) {
            nn$list.glm <- list.glm(data = data, model.list = model.list, 
                family = family)
        }
    }
    if (pred) {
        data.result <- calculate.data.result(response = response, 
            model.list = model.list, covariate = covariate)
        nn$predictions <- calculate.predictions(covariate = covariate, 
            data.result = data.result, list.glm = nn$list.glm, 
            matrix = matrix, list.net.result = nn$net.result, 
            model.list = model.list, act.fct = act.fct)
        if (type(err.fct) == "ce" && all(data.result >= 0) && 
            all(data.result <= 1)) 
            nn$data.error <- sum(err.fct(data.result, response), 
                na.rm = T)
        else nn$data.error <- sum(err.fct(data.result, response))
        if (is.na(nn$data.error)) 
            nn$data.error <- NULL
    }
    return(nn)
}
`generate.rownames` <-
function (matrix, weights, model.list) 
{
    rownames <- rownames(matrix)[rownames(matrix) != ""]
    for (w in 1:length(weights)) {
        for (j in 1:ncol(weights[[w]])) {
            for (i in 1:nrow(weights[[w]])) {
                if (i == 1) {
                  if (w == length(weights)) {
                    rownames <- c(rownames, paste("Intercept.to.", 
                      model.list$response[j], sep = ""))
                  }
                  else {
                    rownames <- c(rownames, paste("Intercept.to.", 
                      w, "layhid", j, sep = ""))
                  }
                }
                else {
                  if (w == 1) {
                    if (w == length(weights)) {
                      rownames <- c(rownames, paste(model.list$variables[i - 
                        1], ".to.", model.list$response[j], sep = ""))
                    }
                    else {
                      rownames <- c(rownames, paste(model.list$variables[i - 
                        1], ".to.1layhid", j, sep = ""))
                    }
                  }
                  else {
                    if (w == length(weights)) {
                      rownames <- c(rownames, paste(w - 1, "layhid.", 
                        i - 1, ".to.", model.list$response[j], 
                        sep = ""))
                    }
                    else {
                      rownames <- c(rownames, paste(w - 1, "layhid.", 
                        i - 1, ".to.", w, "layhid", j, sep = ""))
                    }
                  }
                }
            }
        }
    }
    rownames(matrix) <- rownames
    colnames(matrix) <- 1:(ncol(matrix))
    return(matrix)
}
`list.glm` <-
function (data, model.list, family) 
{
    output <- NULL
    text <- paste("c(output, list(null.model", "=glm(formula=", 
        model.list$response[1], "~0,family=family, data=data)))", 
        sep = "")
    output <- eval(parse(text = text))
    text <- paste("c(output, list(intercept.model", "=glm(formula=", 
        model.list$response[1], "~1,family=family, data=data)))", 
        sep = "")
    output <- eval(parse(text = text))
    if (length(model.list$variables) > 1) {
        full.model <- paste(model.list$response[1], "~", model.list$variable[1], 
            sep = "")
        maineffects.model <- paste(model.list$response[1], "~", 
            model.list$variable[1], sep = "")
        for (i in 1:length(model.list$variables)) {
            if (i != 1) {
                full.model <- paste(full.model, "*", model.list$variable[i], 
                  sep = "")
                maineffects.model <- paste(maineffects.model, 
                  "+", model.list$variable[i], sep = "")
            }
            model <- paste(model.list$response[1], "~", model.list$variables[i], 
                sep = "")
            text <- paste("glm( formula=", model, ", family=family, data=data )", 
                sep = "")
            list.glm <- eval(parse(text = text))
            text <- paste("c(output, list(single.effect.", model.list$variables[i], 
                "=list.glm))", sep = "")
            output <- eval(parse(text = text))
        }
        text <- paste("c(output, list(main.effects", "=glm(formula=", 
            maineffects.model, ",family=family, data=data)))", 
            sep = "")
        output <- eval(parse(text = text))
    }
    else {
        full.model <- paste(model.list$response[1], "~", model.list$variable[1], 
            sep = "")
    }
    text <- paste("c(output, list(full.model", "=glm(formula=", 
        full.model, ",family=family, data=data)))", sep = "")
    output <- eval(parse(text = text))
    return(output)
}
`minus` <-
function (gradients, gradients.old, weights, length.weights, 
    nrow.weights, ncol.weights, learningrate, learningrate.factor, 
    learningrate.limit, algorithm) 
{
    weights <- unlist(weights)
    temp <- gradients.old * gradients
    positive <- temp > 0
    negative <- temp < 0
    if (any(positive)) 
        learningrate[positive] <- pmin.int(learningrate[positive] * 
            learningrate.factor$plus, learningrate.limit$max)
    if (any(negative)) 
        learningrate[negative] <- pmax.int(learningrate[negative] * 
            learningrate.factor$minus, learningrate.limit$min)
    if (algorithm != "rprop-") {
        delta <- 10^-6
        notzero <- gradients != 0
        gradients.notzero <- gradients[notzero]
        if (algorithm == "slr") {
            min <- which.min(learningrate[notzero])
        }
        else if (algorithm == "sag") {
            min <- which.min(abs(gradients.notzero))
        }
        if (length(min) != 0) {
            temp <- learningrate[notzero] * gradients.notzero
            sum <- sum(temp[-min]) + delta
            learningrate[notzero][min] <- min(max(-sum/gradients.notzero[min], 
                learningrate.limit$min), learningrate.limit$max)
        }
    }
    weights <- weights - sign(gradients) * learningrate
    list(gradients.old = gradients, weights = relist(weights, 
        nrow.weights, ncol.weights), learningrate = learningrate)
}
`neuralnet` <-
function (formula, data, hidden = 1, threshold = c(0.001), stepmax = 1e+05, 
    rep = 1, weights.mean = 0, weights.variance = 1, startweights = NULL, 
    learningrate.limit = NULL, learningrate.factor = list(minus = 0.5, 
        plus = 1.2), lifesign = "none", lifesign.step = 1000, 
    algorithm = "rprop+", err.fct = "sse", act.fct = "logistic", 
    linear.output = TRUE, family = NULL) 
{
    call <- match.call()
    options(scipen = 100, digits = 10)
    result <- varify.variables(data, formula, startweights, learningrate.limit, 
        learningrate.factor, lifesign, algorithm, threshold, 
        weights.mean, weights.variance, lifesign.step, hidden, 
        rep, stepmax, err.fct, act.fct)
    data <- result$data
    formula <- result$formula
    startweights <- result$startweights
    learningrate.limit <- result$learningrate.limit
    learningrate.factor <- result$learningrate.factor
    lifesign <- result$lifesign
    algorithm <- result$algorithm
    threshold <- result$threshold
    weights.mean <- result$weights.mean
    weights.variance <- result$weights.variance
    lifesign.step <- result$lifesign.step
    hidden <- result$hidden
    rep <- result$rep
    stepmax <- result$stepmax
    model.list <- result$model.list
    matrix <- NULL
    list.result <- NULL
    result <- generate.initial.variables(data, model.list, hidden, 
        act.fct, err.fct, algorithm, linear.output)
    covariate <- result$covariate
    response <- result$response
    pred <- result$pred
    err.fct <- result$err.fct
    err.deriv.fct <- result$err.deriv.fct
    act.fct <- result$act.fct
    act.deriv.fct <- result$act.deriv.fct
    algorithm <- result$algorithm
    for (i.thr in 1:length(threshold)) {
        for (i.rep in 1:rep) {
            if (lifesign != "none") {
                lifesign <- display(hidden, threshold, i.thr, 
                  rep, i.rep, lifesign)
            }
            flush.console()
            result <- calculate.neuralnet(learningrate.limit = learningrate.limit, 
                learningrate.factor = learningrate.factor, covariate = covariate, 
                response = response, data = data, model.list = model.list, 
                threshold = threshold[i.thr], lifesign.step = lifesign.step, 
                stepmax = stepmax, hidden = hidden, lifesign = lifesign, 
                startweights = startweights, algorithm = algorithm, 
                weights.mean = weights.mean, weights.variance = weights.variance, 
                err.fct = err.fct, err.deriv.fct = err.deriv.fct, 
                act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
                rep = i.rep, linear.output = linear.output)
            if (!is.null(result$output.vector)) {
                list.result <- c(list.result, list(result))
                matrix <- cbind(matrix, result$output.vector)
            }
        }
        flush.console()
    }
    rep.total <- rep * length(threshold)
    if (is.null(matrix)) {
        warning(sprintf("%s of %s repetition(s) reached the stepmax. net.result, weights, gw and the result.matrix were not calculated.", 
            rep.total, rep.total), call. = FALSE)
    }
    else {
        weight.count <- sum(sapply(list.result[[1]]$weights, 
            length))
        if (!is.null(startweights) && length(startweights) < 
            (rep.total * weight.count)) 
            warning("some synapse weights were randomly generated, because 'startweights' does not contain enough values", 
                call. = F)
        if (ncol(matrix) < rep.total) 
            warning(sprintf("%s of %s repetition(s) reached the stepmax", 
                (rep.total - ncol(matrix)), rep.total), call. = FALSE)
        if (any(rownames(matrix) == "aic") && any(is.na(matrix["aic", 
            ]))) {
            message <- sprintf("%s repetition(s) could not calculate the AIC; varify that 'linear.output' is FALSE and that 'act.fct' is bounded between 0 and 1", 
                sum(is.na(matrix["aic", ])))
            warning(message, call. = FALSE)
        }
    }
    nn <- generate.output(covariate, call, rep, threshold, matrix, 
        startweights, model.list, response, err.fct, act.fct, 
        data, family, pred, list.result, linear.output)
    return(nn)
}
`plus` <-
function (gradients, gradients.old, weights, nrow.weights, ncol.weights, 
    learningrate, learningrate.factor, learningrate.limit) 
{
    weights <- unlist(weights)
    sign.gradient <- sign(gradients)
    temp <- gradients.old * sign.gradient
    positive <- temp > 0
    negative <- temp < 0
    not.negative <- !negative
    if (any(positive)) {
        learningrate[positive] <- pmin.int(learningrate[positive] * 
            learningrate.factor$plus, learningrate.limit$max)
    }
    if (any(negative)) {
        weights[negative] <- weights[negative] + gradients.old[negative] * 
            learningrate[negative]
        learningrate[negative] <- pmax.int(learningrate[negative] * 
            learningrate.factor$minus, learningrate.limit$min)
        gradients.old[negative] <- 0
        if (any(not.negative)) {
            weights[not.negative] <- weights[not.negative] - 
                sign.gradient[not.negative] * learningrate[not.negative]
            gradients.old[not.negative] <- sign.gradient[not.negative]
        }
    }
    else {
        weights <- weights - sign.gradient * learningrate
        gradients.old <- sign.gradient
    }
    list(gradients.old = gradients.old, weights = relist(weights, 
        nrow.weights, ncol.weights), learningrate = learningrate)
}
`print.nn` <-
function (x, ...) 
{
    matrix <- x$result.matrix
    cat("Call: ", deparse(x$call), "\n\n", sep = "")
    if (!is.null(x$data.error)) {
        cat("Data Error:\t", x$data.error, ";\t", sep = "")
    }
    if (!is.null(matrix)) {
        if (ncol(matrix) > 1) {
            cat(ncol(matrix), " repetitions were calculated.\n\n", 
                sep = "")
            sorted.matrix <- matrix[, order(matrix["error", ])]
            if (any(rownames(sorted.matrix) == "aic")) {
                print(t(rbind(Error = sorted.matrix["error", 
                  ], AIC = sorted.matrix["aic", ], "Reached Threshold" = sorted.matrix["reached.threshold", 
                  ], Steps = sorted.matrix["steps", ])))
            }
            else {
                print(t(rbind(Error = sorted.matrix["error", 
                  ], "Reached Threshold" = sorted.matrix["reached.threshold", 
                  ], Steps = sorted.matrix["steps", ])))
            }
        }
        else {
            cat(ncol(matrix), " repetition was calculated.\n\n", 
                sep = "")
            if (any(rownames(matrix) == "aic")) {
                print(t(matrix(c(matrix["error", ], matrix["aic", 
                  ], matrix["reached.threshold", ], matrix["steps", 
                  ]), dimnames = list(c("Error", "AIC", "Reached Threshold", 
                  "Steps"), c(1)))))
            }
            else {
                print(t(matrix(c(matrix["error", ], matrix["reached.threshold", 
                  ], matrix["steps", ]), dimnames = list(c("Error", 
                  "Reached Threshold", "Steps"), c(1)))))
            }
        }
    }
    cat("\n")
    if (!is.null(x$list.glm)) {
        cat("\n")
        if (!is.null(x$list.glm$main.effect)) {
            k <- 2
            temp <- x$list.glm$main.effect
        }
        else {
            k <- 1
            temp <- x$list.glm$full.model
        }
        for (i in 1:k) {
            cat("\nCall: ", deparse(temp$call), "\n\n")
            cat("Residual Deviance:", temp$deviance, "\tAIC:", 
                temp$aic, "\n")
            temp <- x$list.glm$full.model
        }
        cat("\n")
    }
}
`relist` <-
function (x, nrow, ncol) 
{
    list.x <- NULL
    for (w in 1:length(nrow)) {
        length <- nrow[w] * ncol[w]
        list.x[[w]] <- matrix(x[1:length], nrow = nrow[w], ncol = ncol[w])
        x <- x[-(1:length)]
    }
    list.x
}
`remove.intercept` <-
function (matrix) 
{
    if (nrow(matrix) != 2 && ncol(matrix) != 1) 
        return(matrix[-1, ])
    if (nrow(matrix) == 2) 
        return(t(matrix[-1, ]))
    return(as.matrix(matrix[-1, ]))
}
`rprop` <-
function (weights, response, covariate, threshold, learningrate.limit, 
    learningrate.factor, stepmax, lifesign, lifesign.step, act.fct, 
    act.deriv.fct, err.fct, err.deriv.fct, algorithm, linear.output) 
{
    step <- 1
    nchar.stepmax <- max(nchar(stepmax), 7)
    length.weights <- length(weights)
    nrow.weights <- sapply(weights, nrow)
    ncol.weights <- sapply(weights, ncol)
    length.unlist <- length(unlist(weights))
    learningrate <- as.vector(matrix(0.1, nrow = 1, ncol = length.unlist))
    gradients.old <- as.vector(matrix(0, nrow = 1, ncol = length.unlist))
    if (type(act.fct) == "tanh" || type(act.fct) == "logistic") 
        compute.net <- compute.net.special
    if (linear.output) {
        calculate.gradients <- calculate.gradients.linear.output
        output.act.fct <- function(x) {
            x
        }
        output.act.deriv.fct <- function(x) {
            matrix(1, nrow(x), ncol(x))
        }
    }
    else {
        if (type(err.fct) == "ce" && type(act.fct) == "logistic") {
            err.deriv.fct <- function(x, y) {
                x * (1 - y) - y * (1 - x)
            }
            compute.net <- compute.net.special
            calculate.gradients <- calculate.gradients.linear.output
        }
        output.act.fct <- act.fct
        output.act.deriv.fct <- act.deriv.fct
    }
    result <- compute.net(weights, length.weights, covariate = covariate, 
        act.fct = act.fct, act.deriv.fct = act.deriv.fct, output.act.fct = output.act.fct, 
        output.act.deriv.fct = output.act.deriv.fct)
    err.deriv <- err.deriv.fct(result$net.result, response)
    gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
        neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
        err.deriv = err.deriv)
    reached.threshold <- max(abs(gradients))
    min.reached.threshold <- reached.threshold
    while (step < stepmax && reached.threshold > threshold) {
        if (!is.character(lifesign) && step%%lifesign.step == 
            0) {
            text <- paste("%", nchar.stepmax, "s", sep = "")
            cat(sprintf(eval(expression(text)), step), "\tmin thresh: ", 
                min.reached.threshold, "\n", rep(" ", lifesign), 
                sep = "")
            flush.console()
        }
        if (algorithm == "rprop+") 
            result <- plus(gradients, gradients.old, weights, 
                nrow.weights, ncol.weights, learningrate, learningrate.factor, 
                learningrate.limit)
        else result <- minus(gradients, gradients.old, weights, 
            length.weights, nrow.weights, ncol.weights, learningrate, 
            learningrate.factor, learningrate.limit, algorithm)
        gradients.old <- result$gradients.old
        weights <- result$weights
        learningrate <- result$learningrate
        result <- compute.net(weights, length.weights, covariate = covariate, 
            act.fct = act.fct, act.deriv.fct = act.deriv.fct, 
            output.act.fct = output.act.fct, output.act.deriv.fct = output.act.deriv.fct)
        err.deriv <- err.deriv.fct(result$net.result, response)
        gradients <- calculate.gradients(weights = weights, length.weights = length.weights, 
            neurons = result$neurons, neuron.deriv = result$neuron.deriv, 
            err.deriv = err.deriv)
        reached.threshold <- max(abs(gradients))
        if (reached.threshold < min.reached.threshold) {
            min.reached.threshold <- reached.threshold
        }
        step <- step + 1
    }
    if (lifesign != "none" && step == stepmax) {
        cat("stepmax\tmin thresh: ", min.reached.threshold, "\n", 
            sep = "")
    }
    return(list(weights = weights, step = as.integer(step), reached.threshold = reached.threshold, 
        net.result = result$net.result, neuron.deriv = result$neuron.deriv))
}
`type` <-
function (fct) 
{
    attr(fct, "type")
}
`varify.variables` <-
function (data, formula, startweights, learningrate.limit, learningrate.factor, 
    lifesign, algorithm, threshold, weights.mean, weights.variance, 
    lifesign.step, hidden, rep, stepmax, err.fct, act.fct) 
{
    if (is.null(data)) 
        stop("'data' is missing", call. = FALSE)
    if (is.null(formula)) 
        stop("'formula' is missing", call. = FALSE)
    if (!is.null(startweights)) 
        startweights <- c(startweights)
    data <- as.data.frame(data)
    formula <- as.formula(formula)
    model.vars <- as.character(formula[3])
    model.resp <- as.character(formula[2])
    model.vars <- gsub(" ", "", model.vars)
    model.resp <- gsub(" ", "", model.resp)
    if (!is.element(substring(model.vars, 1, 1), c("+", "-"))) 
        model.vars <- paste("+", model.vars, sep = "")
    model.vars <- unlist(strsplit(model.vars, "[\\+\\-]"))
    model.resp <- unlist(strsplit(model.resp, "[\\+\\-]"))
    model.vars <- model.vars[model.vars != ""]
    model.resp <- model.resp[model.resp != ""]
    model.list <- list(response = model.resp, variables = model.vars)
    if (!is.null(learningrate.limit)) {
        if (length(learningrate.limit) != 2) 
            stop("'learningrate.factor' must consist of two components", 
                call. = FALSE)
        learningrate.limit <- as.list(learningrate.limit)
        names(learningrate.limit) <- c("min", "max")
        learningrate.limit$min <- as.vector(as.numeric(learningrate.limit$min))
        learningrate.limit$max <- as.vector(as.numeric(learningrate.limit$max))
        if (is.na(learningrate.limit$min) || is.na(learningrate.limit$max)) 
            stop("'learningrate.limit' must be a numeric vector", 
                call. = FALSE)
    }
    if (!is.null(learningrate.factor)) {
        if (length(learningrate.factor) != 2) 
            stop("'learningrate.factor' must consist of two components", 
                call. = FALSE)
        learningrate.factor <- as.list(learningrate.factor)
        names(learningrate.factor) <- c("minus", "plus")
        learningrate.factor$minus <- as.vector(as.numeric(learningrate.factor$minus))
        learningrate.factor$plus <- as.vector(as.numeric(learningrate.factor$plus))
        if (is.na(learningrate.factor$minus) || is.na(learningrate.factor$plus)) 
            stop("'learningrate.factor' must be a numeric vector", 
                call. = FALSE)
    }
    else learningrate.factor <- list(minus = c(0.5), plus = c(1.2))
    if (is.null(lifesign)) 
        lifesign <- "none"
    lifesign <- as.character(lifesign)
    if (!((lifesign == "none") || (lifesign == "minimal") || 
        (lifesign == "full"))) 
        lifesign <- "minimal"
    if (is.na(lifesign)) 
        stop("'lifesign' must be a character", call. = FALSE)
    if (is.null(algorithm)) 
        algorithm <- "rprop+"
    algorithm <- as.character(algorithm)
    if (!((algorithm == "rprop+") || (algorithm == "rprop-") || 
        (algorithm == "slr") || (algorithm == "sag") || (algorithm == 
        "ran"))) 
        stop("'algorithm' is not known", call. = FALSE)
    if (is.null(threshold)) 
        threshold <- c(0.01)
    threshold <- as.vector(as.numeric(threshold))
    if (prod(!is.na(threshold)) == 0) 
        stop("'threshold' must be a numeric vector", call. = FALSE)
    if (is.null(weights.mean)) 
        weights.mean <- 0
    if (!is.numeric(weights.mean)) 
        stop("'weights.mean' must be numeric")
    weights.mean <- as.numeric(weights.mean)
    if (is.null(weights.variance)) 
        weights.variance <- 1
    if (!is.numeric(weights.variance)) 
        stop("'weights.variance' must be numeric")
    weights.variance <- as.numeric(weights.variance)
    if (is.null(lifesign.step)) 
        lifesign.step <- 1000
    lifesign.step <- as.integer(lifesign.step)
    if (is.na(lifesign.step)) 
        stop("'lifesign.step' must be an integer", call. = FALSE)
    if (lifesign.step < 1) 
        lifesign.step <- as.integer(100)
    if (is.null(hidden)) 
        hidden <- 0
    hidden <- as.vector(as.integer(hidden))
    if (prod(!is.na(hidden)) == 0) 
        stop("'hidden' must be an integer vector or a single integer", 
            call. = FALSE)
    if (length(hidden) > 1 && prod(hidden) == 0) 
        stop("'hidden' contains at least one 0", call. = FALSE)
    if (is.null(rep)) 
        rep <- 1
    rep <- as.integer(rep)
    if (is.na(rep)) 
        stop("'rep' must be an integer", call. = FALSE)
    if (is.null(stepmax)) 
        stepmax <- 10000
    stepmax <- as.integer(stepmax)
    if (is.na(stepmax)) 
        stop("'stepmax' must be an integer", call. = FALSE)
    if (stepmax < 1) 
        stepmax <- as.integer(1000)
    if (is.null(hidden)) {
        if (is.null(learningrate.limit)) 
            learningrate.limit <- list(min = c(1e-08), max = c(50))
    }
    else {
        if (is.null(learningrate.limit)) 
            learningrate.limit <- list(min = c(1e-10), max = c(0.1))
    }
    if (!is.function(act.fct) && act.fct != "logistic" && act.fct != 
        "tanh") 
        stop("''act.fct' is not known", call. = FALSE)
    if (!is.function(err.fct) && err.fct != "sse" && err.fct != 
        "ce") 
        stop("'err.fct' is not known", call. = FALSE)
    return(list(data = data, formula = formula, startweights = startweights, 
        learningrate.limit = learningrate.limit, learningrate.factor = learningrate.factor, 
        lifesign = lifesign, algorithm = algorithm, threshold = threshold, 
        weights.mean = weights.mean, weights.variance = weights.variance, 
        lifesign.step = lifesign.step, hidden = hidden, rep = rep, 
        stepmax = stepmax, model.list = model.list))
}
