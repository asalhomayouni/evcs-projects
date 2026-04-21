include("network_design.jl")
include("src/charger_CLI.jl")
using Statistics


# Utility used repeatedly for CLI defaults.
function arg_or(args::Dict{Symbol,Any}, key::Symbol, default)
    if !haskey(args, key)
        return default
    end
    v = args[key]
    return isnothing(v) ? default : v
end


# Utilities used repeatedly while building the PWL waiting approximation.
function erlang_c(s::Int, lam::Float64, mu::Float64)
    rho = lam / (s * mu)
    if rho >= 1.0
        return 1.0
    end
    a = lam / mu
    sum_terms = 0.0
    for k in 0:(s - 1)
        sum_terms += (a^k) / factorial(big(k))
    end
    last_term = (a^s) / (factorial(big(s)) * (1.0 - rho))
    return Float64(last_term / (sum_terms + last_term))
end


function waiting_time_mms(s::Int, mu::Float64, lam::Float64)
    cap = s * mu
    if lam >= cap
        return 1.0e12
    end
    return erlang_c(s, lam, mu) / (cap - lam)
end

# ============================================================
# 1. Config
# ============================================================
const ARGS_OVERRIDE = [
    "--city", "Milano",
    "--k", "30",
    "--timelimit", "30.0",
    "--alpha_wait", "1.0",
    "--breakpoints", "100",
    "--servers_max", "6",
    "--charger_budget", "8",
    "--station_capacity_per_server", "4.0",
    "--lambda_scale", "1.0",
    "--demand_scale", "1.0",
    "--auto_normalize_demand",
    "--noopt_max_range", "30.0",
]

arg_parser = make_charger_arg_parser()
parsed_args = parse_charger_commandline(arg_parser)
parsed_args = parse_charger_commandline(arg_parser, ARGS_OVERRIDE)

time_limit = parsed_args[:timelimit]

# ============================================================
# 2. Load data and preprocess
# ============================================================
I, H = generate_full_instance(parsed_args)
print_instance(I, H)

# --- Basic sets and core data ---
n = size(H.coordinates, 1)
I_set = 1:n
J_set = 1:n

# d_raw: original demand at each clustered node (arrival-rate proxy).
d_raw = copy(H.assigned_demand)
demand_scale = Float64(arg_or(parsed_args, :demand_scale, 1.0))

# t[i,j]: travel-time proxy from i to charger j (Euclidean in current data pipeline).
t = copy(H.tau)
for i in I_set
    t[i, i] = 0.0
end

# Station technology/cost parameters.
s_default = 4
s_max_arg = arg_or(parsed_args, :servers_max, nothing)
s_max = isnothing(s_max_arg) ? s_default : Int(round(Float64(s_max_arg)))
s_max = max(1, s_max)
S_set = 1:s_max

mu_default = 1.2
c_default = 1.0
c = fill(max(1.0e-6, c_default), n)

alpha_wait = Float64(arg_or(parsed_args, :alpha_wait, 1.0))
eps = 1.0e-4

station_cap_arg = arg_or(parsed_args, :station_capacity_per_server, nothing)
if isnothing(station_cap_arg)
    station_cap_per_server = max(1.0e-6, mu_default - eps)
    mu = fill(max(1.0e-6, mu_default), n)
else
    station_cap_per_server = max(1.0e-6, Float64(station_cap_arg))
    # Keep waiting-time and stability capacity internally consistent.
    mu = fill(station_cap_per_server + eps, n)
end

gamma_arg = arg_or(parsed_args, :charger_budget, nothing)
gamma = isnothing(gamma_arg) ? max(1.0, floor(n / 3)) : Float64(gamma_arg)

# cap[j,s_idx]: max stable load for station j with s_idx installed servers.
cap = zeros(Float64, n, s_max)
for j in J_set, s_idx in S_set
    cap[j, s_idx] = s_idx * station_cap_per_server
end

# Optional demand normalization to target average demand per node near 1.
auto_norm = Bool(arg_or(parsed_args, :auto_normalize_demand, false))
avg_d_raw = mean(d_raw)
if auto_norm && avg_d_raw > 0.0
    demand_scale *= 1.0 / avg_d_raw
end

# d: effective demand used in optimization after scaling.
d = d_raw .* demand_scale

# No-opt generalized cost per origin node.
noopt_const = Float64(arg_or(parsed_args, :noopt_max_range, 30.0))
noopt_cost = fill(noopt_const, n)

# --- PWL waiting approximation data ---
N_bp = Int(round(Float64(arg_or(parsed_args, :breakpoints, 100))))
N_bp = max(2, N_bp)
lambda_scale = Float64(arg_or(parsed_args, :lambda_scale, 1.0))
lambda_max_global = max(0.0, lambda_scale * sum(d) - eps)

# lambda_hat[j,s_idx,n]: breakpoint n for station j and server choice s_idx.
# w[j,s_idx,n]: waiting-time derivative evaluated at each breakpoint.
# F[j,s_idx,n]: accumulated integral approximation up to breakpoint.
lambda_hat = zeros(Float64, n, s_max, N_bp)
w = zeros(Float64, n, s_max, N_bp)
F = zeros(Float64, n, s_max, N_bp)
for j in J_set, s_idx in S_set
    local_max = min(lambda_max_global, cap[j, s_idx])
    grid = collect(range(0.0, local_max; length=N_bp))
    lambda_hat[j, s_idx, :] .= grid

    for k in 1:N_bp
        w[j, s_idx, k] = waiting_time_mms(s_idx, mu[j], grid[k])
    end

    F[j, s_idx, 1] = 0.0
    for k in 2:N_bp
        h = grid[k] - grid[k - 1]
        F[j, s_idx, k] = F[j, s_idx, k - 1] + 0.5 * (w[j, s_idx, k - 1] + w[j, s_idx, k]) * h
    end
end

println("================ Full Formulation Setup ================")
println("|I| = $(n), |J| = $(n), breakpoints = $(N_bp)")
println("Raw total demand: $(sum(d_raw))")
println("Scaled total demand: $(sum(d))")
println("Demand scale: $(demand_scale)")
println("Average scaled demand per node: $(mean(d))")
println("Server options per location: 1:$(s_max)")
println("Station capacity per server: $(station_cap_per_server)")
println("No-opt constant (range threshold): $(noopt_const)")
println("No-opt range: [$(minimum(noopt_cost)), $(maximum(noopt_cost))]")
println("========================================================")

if n == 0
    error("No candidate stations were generated.")
end

# ============================================================
# 3. Build and solve full single-level MILP
# ============================================================
model = GurobiFactory.gurobi_model(
    "TimeLimit" => time_limit,
    "Threads" => 1,
)

# Variables (leader and follower primal + follower dual embedding).
# x[j,s_idx] = 1 means location j installs exactly s_idx servers.
@variable(model, x[j in J_set, s_idx in S_set], Bin)
@variable(model, xj[j in J_set], Bin)
@variable(model, y[i in I_set, j in J_set] >= 0)
@variable(model, y_noopt[i in I_set] >= 0)
@variable(model, lambda[j in J_set] >= 0)
@variable(model, phi[j in J_set] >= 0)

@variable(model, pi_d[i in I_set])
@variable(model, pi_lambda[j in J_set])
@variable(model, pi_x[i in I_set, j in J_set] >= 0)
@variable(model, pi_s[j in J_set, s_idx in S_set] >= 0)
@variable(model, pi_w[j in J_set, s_idx in S_set, n_bp in 1:N_bp] >= 0)
@variable(model, Pi[i in I_set, j in J_set] >= 0)

# Upper-level objective: maximize load served by opened chargers.
@objective(model, Max, sum(lambda[j] for j in J_set))

# Leader budget and follower primal feasibility.
@constraint(model, budget, sum(c[j] * s_idx * x[j, s_idx] for j in J_set, s_idx in S_set) <= gamma)
@constraint(model, one_facility_per_location[j in J_set], sum(x[j, s_idx] for s_idx in S_set) <= 1)
@constraint(model, location_open[j in J_set], xj[j] == sum(x[j, s_idx] for s_idx in S_set))
@constraint(model, demand[i in I_set], sum(y[i, j] for j in J_set) + y_noopt[i] == d[i])
@constraint(model, access[i in I_set, j in J_set], y[i, j] <= d[i] * xj[j])
@constraint(model, load[j in J_set], lambda[j] == sum(y[i, j] for i in I_set))
@constraint(model, stability[j in J_set], lambda[j] <= sum(cap[j, s_idx] * x[j, s_idx] for s_idx in S_set))

# Waiting epigraph is activated only for selected server choices at each location.
for j in J_set, s_idx in S_set, n_bp in 1:N_bp
    @constraint(model, x[j, s_idx] => {phi[j] >= F[j, s_idx, n_bp] + w[j, s_idx, n_bp] * (lambda[j] - lambda_hat[j, s_idx, n_bp])})
end
for j in J_set
    @constraint(model, !xj[j] => {phi[j] == 0})
end

# Follower dual feasibility.
@constraint(model, dual_y[i in I_set, j in J_set], t[i, j] + pi_d[i] + pi_x[i, j] - pi_lambda[j] >= 0)
@constraint(model, dual_y_noopt[i in I_set], noopt_cost[i] + pi_d[i] >= 0)
@constraint(model, dual_lambda[j in J_set], pi_lambda[j] + sum(pi_s[j, s_idx] for s_idx in S_set) + sum(w[j, s_idx, n_bp] * pi_w[j, s_idx, n_bp] for s_idx in S_set, n_bp in 1:N_bp) >= 0)
@constraint(model, dual_phi[j in J_set], alpha_wait - sum(pi_w[j, s_idx, n_bp] for s_idx in S_set, n_bp in 1:N_bp) >= 0)

# Strong duality enforces lower-level optimality in the single-level embedding.
@constraint(model,
    strong_duality,
    sum(t[i, j] * y[i, j] for i in I_set, j in J_set) +
    sum(noopt_cost[i] * y_noopt[i] for i in I_set) +
    alpha_wait * sum(phi[j] for j in J_set)
    ==
    -sum(d[i] * pi_d[i] for i in I_set) -
    sum(d[i] * Pi[i, j] for i in I_set, j in J_set) -
    sum(cap[j, s_idx] * pi_s[j, s_idx] for j in J_set, s_idx in S_set) +
    sum((F[j, s_idx, n_bp] - w[j, s_idx, n_bp] * lambda_hat[j, s_idx, n_bp]) * pi_w[j, s_idx, n_bp] for j in J_set, s_idx in S_set, n_bp in 1:N_bp)
)

# Linearization of Pi = x * pi_x via indicators (no big-M).
for i in I_set, j in J_set, s_idx in S_set
    @constraint(model, x[j, s_idx] => {Pi[i, j] - pi_x[i, j] == 0})
end
for i in I_set, j in J_set
    @constraint(model, !xj[j] => {Pi[i, j] == 0})
end

# Store references for REPL inspection and plotting helpers.
model[:x] = x
model[:xj] = xj
model[:y] = y
model[:y_noopt] = y_noopt
model[:lambda] = lambda
model[:phi] = phi
model[:pi_d] = pi_d
model[:pi_lambda] = pi_lambda
model[:pi_x] = pi_x
model[:pi_s] = pi_s
model[:pi_w] = pi_w
model[:Pi] = Pi

println("Solving full single-level charger MILP...")
optimize!(model)

println("Status: ", termination_status(model))

if has_values(model)
    x_val = value.(model[:x])
    lambda_val = value.(model[:lambda])
    y_noopt_val = value.(model[:y_noopt])

    # Served load equals objective value in this formulation.
    served_load = objective_value(model)
    open_set = [(j, s_idx) for j in J_set, s_idx in S_set if x_val[j, s_idx] >= 0.5]
    total_lambda = sum(lambda_val[j] for j in J_set)
    total_noopt_flow = sum(y_noopt_val[i] for i in I_set)
    total_installed_servers = sum(s_idx * x_val[j, s_idx] for j in J_set, s_idx in S_set)

    println("Objective (max served load): ", served_load)
    println("Selected facilities count: ", length(open_set))
    println("Selected (location, servers): ", open_set)
    println("Total installed servers: ", total_installed_servers)
    println("Total lambda: ", total_lambda)
    println("Total no-opt flow: ", total_noopt_flow)

    # Aggregate model values to location level for existing plotting helper.
    x_plot = zeros(Float64, n)
    y_plot = zeros(Float64, n, n)
    for j in J_set
        x_plot[j] = sum(x_val[j, s_idx] for s_idx in S_set)
        for i in I_set
            y_plot[i, j] = value(model[:y][i, j])
        end
    end
    model[:x_plot] = x_plot
    model[:y_plot] = y_plot

    # Plot uses t, demand, and x/y values to show assignment geometry.
    plot_data = (
        n=n,
        I_set=I_set,
        J_set=J_set,
        d=d,
        t=t,
        noopt_cost=noopt_cost,
    )
    allocation_plot = plot_charger_solution(H, plot_data, model)
    display(allocation_plot)
    println("Rendered full-formulation allocation plot (variable: allocation_plot).")
else
    println("No primal values returned.")
end