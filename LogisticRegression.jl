function _mult(a::Array{Float64,1},b::Array{Float64,2})
    result = zeros(length(a))
    both_non_zero_indicator = ((a .!= 0) & (b .!= 0))
    result[both_non_zero_indicator[:]] = a[both_non_zero_indicator[:]] .* b[both_non_zero_indicator]
    return result
end


function goal_function(omega::Array{Float64,2}, beta::Float64, data::Array{Float64,2}, labels::Array{Float64,1})
    f_partial = 1.0 ./ (1.0 + exp(-data * omega  - beta))
    result = -sum(_mult(labels, log(f_partial)) + _mult((1.0 - labels), log(1.0 - f_partial)))
    return result
end

function convergence(omega::Array{Float64,2}, beta::Float64, data::Array{Float64,2}, labels::Array{Float64,1},  prevJ::Float64, epsilon::Float64)
     currJ = goal_function(omega, beta, data, labels)
     return abs(prevJ - currJ) < epsilon
end

function update_params(omega::Array{Float64,2}, beta::Float64,data::Array{Float64,2}, labels::Array{Float64,1}, alpha::Float64)
    partial_derivative = (1.0 ./ (1.0 + exp(-data * omega  - beta)))  - labels
    omega = omega - alpha *  (partial_derivative' * data)'
    beta = beta  - alpha * sum(partial_derivative)
    return omega,beta
end

function logistic_regression_learn(data::Array{Float64,2}, labels::Array{Float64,1}, params::Dict{Symbol,Float64})

    omega = zeros(Float64, size(data,2),1)
    beta = 0.0
    J = Inf
    current_iter = 0
    alpha_step, epsilon, max_iter = params[:alpha], params[:eps], params[:max_iter]
    J_values = zeros(0)
    while !convergence(omega, beta, data, labels, J, epsilon) && current_iter < max_iter
         J = goal_function(omega, beta, data, labels)
		 append!(J_values, [J])
         omega, beta = update_params(omega, beta, data, labels, alpha_step)
         current_iter += 1		 
    end
	
	model_params  = Dict();
	model_params[:omega] = omega; model_params[:beta] = beta;	
    return model_params, J_values
end

function predict(data, model_params) 
	1.0 ./ (1.0 + exp(-data * model_params[:omega]  - model_params[:beta]))
end 