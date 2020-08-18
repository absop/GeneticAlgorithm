using Random
using Statistics
using Pkg
Pkg.add("Plots")
using Plots

theme(:vibrant)


ω = 10
σ = 2

pc = 0.6    # the probability of crossover
pm = 0.01   # the probability of mutation

lfc = 22            # full length of chromosome
lhc = div(lfc, 2)   # half length of chromosome
scale = 2^lhc / 10  # 10 => [-5.0, 5.0]
mapper = Dict('0' => '1', '1' => '0')


function fitness(x, y)
    sin(ω * x)^2 * sin(ω * y)^2 * ℯ^((x + y) / σ)
end

#2d plot#
gr()
contour(-5.0:0.01:5.0, -5.0:0.01:5.0, fitness, fill=:true, levels=:8,
    size=[450,450], color=:blues; legend=:none)

#3d plot#
plotly()
surface(-5.0:0.01:5.0, -5.0:0.01:5.0, fitness, fill=:true, levels=:8,
    size=[450,450], color=:blues; legend=:none)


function decode(s::String)
    integer = parse(Int, s, base=2)
    numerical = (integer / scale) - 5.0
    return numerical
end

function encode(numerical::Float64)
    integer = Int(floor((numerical + 5.0) * scale))
    binary = bitstring(integer)[64-lhc+1:64]
    return binary
end

function encode(population::Array{Array{Float64,1},1})
    genes = [
        [encode(population[i][1]), encode(population[i][2])]
        for i in 1:length(population)
    ]
end

function update_fitness(fitnesses, genes)
    for i in 1:length(genes)
        x1 = decode(genes[i][1])
        x2 = decode(genes[i][2])
        fitnesses[i] = fitness(x1, x2)
    end
end

function eliminate!(fitnesses)
    for i in 1:length(fitnesses)
        if fitnesses[i] < 0
            fitnesses[i] = 0.0
        end
    end
end


function select!(genes, fitnesses)
    total = sum(fitnesses)
    start = 0
    old_genes = copy(genes)
    distribution = []
    for i in 1:length(old_genes)
        temp = fitnesses[i]/total + start
        append!(distribution, temp)
        start = temp
    end

    for i in 1:length(old_genes)
        randx = rand()
        index = 0
        for j in 1:length(old_genes)
            if distribution[j] < randx
                continue
            else
                index = j
                break
            end
        end

        if index == 0
            index = length(old_genes)
        end
        genes[index] = old_genes[index]
    end
end


function crossover!(genes)
    for i in 1:2:length(genes) - 1
        j = i + 1
        if rand() < pc
            cross_pos = rand(2:lfc-1)
            i_xy_str = string(genes[i][1], genes[i][2])
            j_xy_str = string(genes[j][1], genes[j][2])

            str1_part1 = i_xy_str[1:cross_pos]
            str1_part2 = i_xy_str[cross_pos+1:lfc]
            str2_part1 = j_xy_str[1:cross_pos]
            str2_part2 = j_xy_str[cross_pos+1:lfc]
            str1 = string(str1_part1, str2_part2)
            str2 = string(str2_part1, str1_part2)
            genes[i][1] = str1[1:lhc]
            genes[i][2] = str1[lhc+1:lfc]
            genes[j][1] = str2[1:lhc]
            genes[j][2] = str2[lhc+1:lfc]
        end
    end
end


function mutation!(genes)
    for i in 1:length(genes)
        if rand() < pm
            r = rand(1:lfc)
            j = cld(r, lhc)
            k = rem(r, lhc)
            if k == 0
                k = lhc
            end
            g = genes[i][j]
            genes[i][j] = string(g[1:k-1], mapper[g[k]], g[k+1:lhc])
        end
    end
end


function getbest(xarray, yarray, zarray, fitnesses, genes)
    value = -Inf
    index = 1
    for i in 1:length(fitnesses)
        if fitnesses[i] > value
            value = fitnesses[i]
            index = i
        end
    end
    x = decode(genes[index][1])
    y = decode(genes[index][2])
    z = value
    append!(xarray, x)
    append!(yarray, y)
    append!(zarray, z)
end


function solve(n, generation)
    population = [
        [rand(-5.0:0.01:5.0), rand(-5.0:0.01:5.0)]
        for i in 1:n
    ]
    fitnesses = [
        fitness(population[i][1], population[i][2])
        for i in 1:n
    ]
    genes = encode(population)

    xarray = []
    yarray = []
    zarray = []

    getbest(xarray, yarray, zarray, fitnesses, genes)

    for i in 1:generation
        eliminate!(fitnesses)
        select!(genes, fitnesses)
        crossover!(genes)
        mutation!(genes)
        update_fitness(fitnesses, genes)
        getbest(xarray, yarray, zarray, fitnesses, genes)
    end

    value = -Inf
    index = 0
    for i in 1:length(zarray)
        if zarray[i] > value
            value = zarray[i]
            index = i
        end
    end
    scatter!(xarray, yarray, zarray, color=:red, label="chromosome #1")
    scatter!(xarray, yarray, zarray, color=:red, label="flying chromosome")
    xarray[index], yarray[index], zarray[index]
end



point3d = solve(500, 100)
println(point3d)
