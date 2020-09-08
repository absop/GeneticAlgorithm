using Random
using Statistics
using Pkg

ω = 10
σ = 2

pc = 0.6    # the probability of crossover
pm = 0.01   # the probability of mutation

lfc = 22            # full length of chromosome
lhc = div(lfc, 2)   # half length of chromosome
scale = 2^lhc / 10  # 10 => [-5.0, 5.0]

function decode(integer::Int64)
    numerical = (integer / scale) - 5.0
    return numerical
end

function encode(numerical::Float64)
    integer = Int(floor((numerical + 5.0) * scale))
    integer = integer & ((1 << lhc) - 1)
    return integer
end

function encode(population::Array{Array{Float64,1},1})
    genes = [
        [encode(population[i][1]), encode(population[i][2])]
        for i in 1:length(population)
    ]
end

function fitness(x, y)
    sin(ω * x)^2 * sin(ω * y)^2 * ℯ^((x + y) / σ)
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
    start = 0.0
    old_genes = copy(genes)
    parray = []
    for i in 1:length(old_genes)
        start += fitnesses[i] / total
        append!(parray, start)
    end

    for i in 1:length(old_genes)
        randx = rand()
        index = 0
        for j in 1:length(old_genes)
            if parray[j] < randx
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

function invbit(n, i, total)
    n ⊻ (1 << (total - i))
end

function crossover!(genes)
    for i in 1:2:length(genes) - 1
        j = i + 1
        if rand() < pc
            cross_pos = rand(2:lfc-1)

            if cross_pos <= lhc
                genes[i][2], genes[j][2] = genes[j][2], genes[i][2]
                index = 1
            else
                cross_pos -= lhc
                index = 2
            end
            bit_mask_lo = (1 << (lhc - cross_pos)) - 1
            bit_mask_hi = ~bit_mask_lo
            i_hi = bit_mask_hi & genes[i][index]
            i_lo = bit_mask_lo & genes[i][index]
            j_hi = bit_mask_hi & genes[j][index]
            j_lo = bit_mask_lo & genes[j][index]

            genes[i][index] = i_hi | j_lo
            genes[j][index] = j_hi | i_lo
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
            genes[i][j] = invbit(genes[i][j], k, lhc)
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


function evolute(n, generation)
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
    foreach(index->println([xarray[index], yarray[index], zarray[index]]), 1:length(zarray))
    xarray[index], yarray[index], zarray[index]
end



point3d = evolute(500, 1000)
println(point3d)
