include("network_design.jl")
include("src/charger_CLI.jl")
using Statistics
using Random


function arg_or(args::Dict{Symbol,Any}, key::Symbol, default)
    if !haskey(args, key)
        return default
    end
    v = args[key]
    return isnothing(v) ? default : v
end


# Randomly open chargers within budget.
function random_open_plan(n::Int, s_max::Int, budget::Float64, unit_cost::Float64, seed::Int)
    Random.seed!(seed)

    servers = zeros(Int, n)
    remaining = budget

    perm = randperm(n)
    for j in perm
        max_affordable = min(s_max, Int(floor(remaining / unit_cost)))
        if max_affordable < 1
            break
        end

        # Random server count at this site, respecting remaining budget.
        s_j = rand(1:max_affordable)
        servers[j] = s_j
        remaining -= unit_cost * s_j
    end

    return servers, remaining
end


# Assign demand greedily to nearest open stations that are within no-opt cost
# threshold, while respecting station capacity.
function assign_demand_with_capacity(
    d::Vector{Float64},
    t::Matrix{Float64},
    open_servers::Vector{Int},
    station_cap_per_server::Float64,
    noopt_const::Float64,
)
    n = length(d)
    y = zeros(Float64, n, n)
    y_noopt = zeros(Float64, n)

    cap_remaining = [open_servers[j] * station_cap_per_server for j in 1:n]
    open_idx = [j for j in 1:n if open_servers[j] > 0]

    for i in 1:n
        rem = d[i]

        if !isempty(open_idx)
            sorted_open = sort(open_idx, by=j -> t[i, j])
            for j in sorted_open
                if t[i, j] > noopt_const
                    break
                end
                if cap_remaining[j] <= 0.0
                    continue
                end

                flow = min(rem, cap_remaining[j])
                if flow > 0.0
                    y[i, j] += flow
                    cap_remaining[j] -= flow
                    rem -= flow
                end

                if rem <= 1.0e-9
                    rem = 0.0
                    break
                end
            end
        end

        y_noopt[i] = rem
    end

    captured = sum(d) - sum(y_noopt)
    return y, y_noopt, cap_remaining, captured
end


# ============================================================
# 1. Config
# ============================================================
const ARGS_OVERRIDE = [
    "--city", "Milano",
    "--k", "100",
    "--demand_scale", "1.0",
    "--auto_normalize_demand",
    "--servers_max", "6",
    "--charger_budget", "30",
    "--station_capacity_per_server", "4.0",
    "--noopt_max_range", "30.0",
    "--seed", "42",
]

arg_parser = make_charger_arg_parser()
parsed_args = parse_charger_commandline(arg_parser)
parsed_args = parse_charger_commandline(arg_parser, ARGS_OVERRIDE)

# ============================================================
# 2. Load data
# ============================================================
I, H = generate_full_instance(parsed_args)
print_instance(I, H)

n = size(H.coordinates, 1)
I_set = 1:n
J_set = 1:n

if n == 0
    error("No candidate stations were generated.")
end

# Demand scaling
raw_d = copy(H.assigned_demand)
demand_scale = Float64(arg_or(parsed_args, :demand_scale, 1.0))
auto_norm = Bool(arg_or(parsed_args, :auto_normalize_demand, false))
avg_raw_d = mean(raw_d)
if auto_norm && avg_raw_d > 0.0
    demand_scale *= 1.0 / avg_raw_d
end
d = raw_d .* demand_scale

# Travel matrix
t = copy(H.tau)
for i in I_set
    t[i, i] = 0.0
end

# Model defaults aligned with the current full script choices.
s_default = 4
s_max_arg = arg_or(parsed_args, :servers_max, nothing)
s_max = isnothing(s_max_arg) ? s_default : Int(round(Float64(s_max_arg)))
s_max = max(1, s_max)

mu_default = 1.2
eps = 1.0e-4
station_cap_arg = arg_or(parsed_args, :station_capacity_per_server, nothing)
if isnothing(station_cap_arg)
    station_cap_per_server = max(1.0e-6, mu_default - eps)
else
    station_cap_per_server = max(1.0e-6, Float64(station_cap_arg))
end

# Budget and no-opt constant
c_default = 1.0
gamma_arg = arg_or(parsed_args, :charger_budget, nothing)
gamma = isnothing(gamma_arg) ? max(1.0, floor(n / 3)) : Float64(gamma_arg)
noopt_const = Float64(arg_or(parsed_args, :noopt_max_range, 30.0))
seed = Int(round(Float64(arg_or(parsed_args, :seed, 42))))

# ============================================================
# 3. Random placement + captured demand
# ============================================================
open_servers, remaining_budget = random_open_plan(n, s_max, gamma, c_default, seed)

open_idx = [j for j in J_set if open_servers[j] > 0]
installed_servers = sum(open_servers)
used_budget = installed_servers * c_default

y, y_noopt, cap_remaining, captured = assign_demand_with_capacity(
    d,
    t,
    open_servers,
    station_cap_per_server,
    noopt_const,
)

captured_ratio = (sum(d) > 0.0) ? (captured / sum(d)) : 0.0

println("================ Random Charger Baseline ================")
println("|I| = $(n), |J| = $(n)")
println("Seed: $(seed)")
println("Raw total demand: $(sum(raw_d))")
println("Scaled total demand: $(sum(d))")
println("Demand scale: $(demand_scale)")
println("Budget gamma: $(gamma)")
println("Budget used: $(used_budget)")
println("Budget remaining: $(remaining_budget)")
println("Opened locations: $(length(open_idx))")
println("Installed servers: $(installed_servers)")
println("Station capacity per server: $(station_cap_per_server)")
println("No-opt constant threshold: $(noopt_const)")
println("Captured demand: $(captured)")
println("No-opt demand: $(sum(y_noopt))")
println("Captured ratio: $(round(captured_ratio * 100, digits=2))%")
println("=========================================================")

# Reuse existing plotting helper by attaching random-baseline values.
x_plot = [open_servers[j] > 0 ? 1.0 : 0.0 for j in J_set]
plot_model = JuMP.Model()
plot_model[:x_plot] = x_plot
plot_model[:y_plot] = y

plot_data = (
    n=n,
    I_set=I_set,
    J_set=J_set,
    d=d,
    t=t,
    noopt_cost=fill(noopt_const, n),
)
allocation_plot = plot_charger_solution(H, plot_data, plot_model)
display(allocation_plot)
println("Rendered random-baseline allocation plot (variable: allocation_plot).")

# Expose arrays for REPL inspection.
open_servers_result = open_servers
assignment_result = y
noopt_result = y_noopt
remaining_capacity_result = cap_remaining
