using DifferentiableTrajectoryOptimization: DifferentiableTrajectoryOptimization as Dito
using JuMP: JuMP
using Ipopt: Ipopt
using VegaLite: VegaLite

"""
Just a simple dummy problem to have something to work with.
"""
function get_primal_problem_data()
    A = [1 1; 0 1]
    B = [0 0.1]'

    function inequality_constraints(xs, us, params)
        params_reshaped = reshape(params, :, 2)
        lbx_violations = params_reshaped[:, 1]
        ubx_violations = params_reshaped[:, 2]
        # box constraints on states and inputs
        mapreduce(vcat, xs, us, Iterators.countfrom()) do x, u, t
            [
                x[1] + 1 + lbx_violations[t], # x1 >= -1
                -x[1] + 1 + ubx_violations[t], # x1 <= 1
                u[1] + 1, # u1 >= -1
                -u[1] + 1, # u1 <= 1
            ]
        end
    end

    state_dim = 2
    control_dim = 1
    parameter_dim = 100
    horizon = 50

    (; cost, dynamics, inequality_constraints, state_dim, control_dim, parameter_dim, horizon, A, B)
end

# TODO: there is probalby a way to set this up with Dito but currently cannot wrap my head around it
"""
In this optimization problem, we relax the constraints on x[1] and introduce slack variables whose
value we minimize. That is, we look for the minimum violation of that contraint if we still enforce
dynamics and input contraints.

If there is any input sequence that satisfies the constraints without relaxation, this problem will
return the unaltered constraints.

Note that in this setting, the primal objective does *note* appear*

Alson ote that we could use X and U of this problem as a starting point for the primal problem.
"""
function find_optimal_relaxation(; x0)
    (; state_dim, control_dim, horizon, A, B) = get_primal_problem_data()

    model = JuMP.Model(Ipopt.Optimizer)
    X = JuMP.@variable(model, [1:state_dim, 1:horizon])
    U = JuMP.@variable(model, [1:control_dim, 1:horizon])
    # slack variables to track the constraint violation
    lbx_violation = JuMP.@variable(model, [1:horizon], lower_bound = 0)
    ubx_violation = JuMP.@variable(model, [1:horizon], lower_bound = 0)

    # initial state constraint
    JuMP.@constraint(model, X[:, 1] .== x0)
    # dynamics constraints
    JuMP.@constraint(model, [t = 1:(horizon - 1)], X[:, t + 1] .== A * X[:, t] + B * U[:, t])
    # (unrelaxed) input constraints
    JuMP.@constraint(model, [t = 1:horizon], -1 .<= U[:, t] .<= 1)

    JuMP.@constraint(model, [t = 1:horizon], X[1, t] >= -1 - lbx_violation[t])
    JuMP.@constraint(model, [t = 1:horizon], X[1, t] <= 1 + ubx_violation[t])

    JuMP.@objective(model, Min, sum(lbx_violation) + sum(ubx_violation))

    JuMP.optimize!(model)

    (;
        X = JuMP.value.(X),
        U = JuMP.value.(U),
        lbx_violation = JuMP.value.(lbx_violation),
        ubx_violation = JuMP.value.(ubx_violation),
    )
end

"""
In this optimization problem, we minimize the primal objective and enforce only the relaxed
constraints.

Note that the objective of this primal problem is "pushing into the contraints" and thus, at the
optimal solution, the constraints are active. This is to demonstrate that the constraints will still
be enforced when feasible.
"""
function solve_primal_problem(; x0, lbx_violation, ubx_violation)
    (; state_dim, control_dim, horizon, A, B) = get_primal_problem_data()

    model = JuMP.Model(Ipopt.Optimizer)
    X = JuMP.@variable(model, [1:state_dim, 1:horizon])
    U = JuMP.@variable(model, [1:control_dim, 1:horizon])

    # initial state constraint
    JuMP.@constraint(model, X[:, 1] .== x0)
    # dynamics constraints
    JuMP.@constraint(model, [t = 1:(horizon - 1)], X[:, t + 1] .== A * X[:, t] + B * U[:, t])
    # (unrelaxed) input constraints
    JuMP.@constraint(model, [t = 1:horizon], -1 .<= U[:, t] .<= 1)

    JuMP.@constraint(model, [t = 1:horizon], X[1, t] >= -1 - lbx_violation[t])
    JuMP.@constraint(model, [t = 1:horizon], X[1, t] <= 1 + ubx_violation[t])

    JuMP.@objective(model, Min, sum(X[1, :] .+ 3) + sum(X[2, :] .^ 2) + sum(U .^ 2))

    JuMP.optimize!(model)

    xs = reshape(JuMP.value.(X), state_dim, horizon) |> eachcol |> collect
    us = reshape(JuMP.value.(U), control_dim, horizon) |> eachcol |> collect

    (; xs, us)
end

function main()
    x0 = [0.0, 1.0]
    # find the minimal constraint relaxation which renders the problem feasible
    (; lbx_violation, ubx_violation) = find_optimal_relaxation(; x0)
    (; xs, us) = solve_primal_problem(; x0, lbx_violation, ubx_violation)

    # plot the solution
    plot_data = [
        (; t, x1 = x[1], x2 = x[2], u = only(u), lbx1 = -1 - lbx1v, ubx1 = 1 + ubx1v) for
        (t, (x, u, lbx1v, ubx1v)) in enumerate(zip(xs, us, lbx_violation, ubx_violation))
    ]

    x1_plot =
        plot_data |>
        VegaLite.@vlplot(mark = :line, x = :t, width = 500) +
        VegaLite.@vlplot(:line, y = "x1") +
        VegaLite.@vlplot(:line, y = "lbx1", color = {value = "red"}) +
        VegaLite.@vlplot(:line, y = "ubx1", color = {value = "red"})
    x2_plot = plot_data |> VegaLite.@vlplot(:line, x = :t, y = :x2, width = 500)
    u_plot = plot_data |> VegaLite.@vlplot(:line, x = :t, y = :u, width = 500)

    [x1_plot; x2_plot; u_plot]
end
