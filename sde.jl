using Plots
using Statistics
using Distributions
using StatsPlots

import Random


function probF_SIP(S, I, P, params)

    r, k, lambda, mu, m, a, theta, delta = params

    dS_dW1 = sqrt(r * S)
    dS_dW2 = -sqrt(r / k * S * (S + 1))
    dS_dW3 = -sqrt(lambda * S * I)
    dI_dW3 = sqrt(lambda * S * I)
    dI_dW4 = -sqrt( m * P * I / (I + a) + mu * I)
    dP_dW5 = sqrt(theta * I * P / (I + a))
    dP_dW6 = -sqrt(delta * P)

    return dS_dW1, dS_dW2, dS_dW3, dI_dW3, dI_dW4, dP_dW5, dP_dW6

end

function F_SIP(S, I, P, params)

    r, k, lambda, mu, m, a, theta, delta = params

    dS = r * S * (1 - (S + I) / k) - lambda * S * I
    dI = lambda * S * I - mu * I - m * I * P / (I + a)
    dP = theta * I * P / (I + a) - delta * P

    return dS, dI, dP

end

function sde(params, T, dt0, sip0)
    # Initial values sip: [S0, I0, P0]
    # params: [r, k, lambda, mu, m, a, theta, delta]
    # T: terminal time
    # N: number of time steps
    
    # intialize everything
    dt = dt0
    S = [sip0[1]]
    I = [sip0[2]]
    P = [sip0[3]]
    t = [0.0]

    # take time-steps using Euler-Maruyama
    n = 1
    while t[end] < T
        dt = min(T - t[end], dt0)
        dW = sqrt(dt) .* randn(6)

        dW1, dW2, dW3, dW4, dW5, dW6 = dW

        # Compute S {n+1}, I {n+1}, P {n+1} using step size dt
        dS_dt, dI_dt, dP_dt = F_SIP(S[n], I[n], P[n], params)

        dS_dW1, dS_dW2, dS_dW3, dI_dW3, dI_dW4, dP_dW5, dP_dW6 = probF_SIP(S[n], I[n], P[n], params)

        # calculate change in sip
        dS = dS_dt * dt + (dS_dW1 * dW1 + dS_dW2 * dW2 + dS_dW3 * dW3)
        dI = dI_dt * dt + (dI_dW3 * dW3 + dI_dW4 * dW4)
        dP = dP_dt * dt + (dP_dW5 * dW5 + dP_dW6 * dW6)

        # calculate new sip
        push!(S, S[n] + dS)
        push!(I, I[n] + dI)
        push!(P, P[n] + dP)
    
            while min(S[n + 1], I[n + 1], P[n + 1]) < 0
            dt = dt / 2 
            dW = dW / sqrt(2)
            dW1, dW2, dW3, dW4, dW5, dW6 = dW
    
            # Recompute S {n+1}, I {n+1}, P {n+1} using new step size
            dS_dt, dI_dt, dP_dt = F_SIP(S[n], I[n], P[n], params)

            dS_dW1, dS_dW2, dS_dW3, dI_dW3, dI_dW4, dP_dW5, dP_dW6 = probF_SIP(S[n], I[n], P[n], params)

            # calculate change in sip
            dS = dS_dt * dt + (dS_dW1 * dW1 + dS_dW2 * dW2 + dS_dW3 * dW3)
            dI = dI_dt * dt + (dI_dW3 * dW3 + dI_dW4 * dW4)
            dP = dP_dt * dt + (dP_dW5 * dW5 + dP_dW6 * dW6)

            # calculate new sip
            S[n + 1] = S[n] + dS
            I[n + 1] = I[n] + dI
            P[n + 1] = P[n] + dP
            end
        push!(t, t[end] + dt)
        n = n + 1

    end

    return t, S, I, P

end


function run_simulation_x_times()
    # param = [r k lambda mu m a theta delta]
    param = [7.0, 400.0, 0.06, 3.4, 15.5, 15.0, 10.0, 8.3]

    # intial conditions S0, I0, P0
    sip = [100.0, 80.0, 20.0]

    # final time
    T = 10.0

    # intial time-step
    dt0 = 1e-4

    num_iterations = 100

    S_means = zeros(num_iterations)
    P_means = zeros(num_iterations)
    I_means = zeros(num_iterations)

    S_vars = zeros(num_iterations)
    P_vars = zeros(num_iterations)
    I_vars = zeros(num_iterations)

    for i = 1:num_iterations

        t, S, I, P = @time sde(param, T, dt0, sip)

        # display(plot(t, [S, I, P]))

        S_means[i] = mean(S)
        I_means[i] = mean(I)
        P_means[i] = mean(P)

        S_vars[i] = var(S)
        I_vars[i] = var(I)
        P_vars[i] = var(P)
    end
    
    display(histogram(S_means, bins = 20))
    display(histogram(I_means, bins = 20))
    display(histogram(P_means, bins = 20))
    display(histogram(S_vars, bins = 20))
    display(histogram(I_vars, bins = 20))
    display(histogram(P_vars, bins = 20))

end

function explore_parameter_a()
    # param = [r k lambda mu m a theta delta]
    param = [24.0, 400.0, 0.06, 3.4, 15.5, 1.0, 10.0, 8.3]

    # intial conditions S0, I0, P0
    sip = [100.0, 80.0, 20.0]

    # final time
    T = 10.0

    # intial time-step
    dt0 = 1e-4

    P_10 = zeros(30)

    for a = 1:30
        println(a)
        param[6] = float(a)

        P_means = zeros(400)

        for i = 1:400

            _, _, _, P = sde(param, T, dt0, sip)

            P_means[i] = mean(P)
        end

        P_10[a] = mean(P_means)
    end

    display(plot(1:30, P_10))
 
end

function main()
    explore_parameter_a()
end

main()