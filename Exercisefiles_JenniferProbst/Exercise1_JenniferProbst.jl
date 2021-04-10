#Exercises 1:
function stringreverser(s::String)
    rev=""
    if length(s)==1
        return s
    else
        return ( rev * s[end] * stringreverser(s[1:length(s)-1]))
    end 
end
stringreverser("hallo")

#Exercises 2: 
function palindrom(s::String)
    if length(s)==0 || length(s)==1
        return "true"
    elseif s[1]==s[end]
        return palindrom(s[2:length(s)-1])
    else 
        return "false"
    end 
end
palindrom("anna")
palindrom("annma")
palindrom("blulb")

#Exercises 3:
function counter(tocount)
    sum = 0
    for i in tocount
        sum+=1
    end
end

function dublicate(inp::String)
    len = counter(inp)
    ret = inp * inp
    return ret
end

function dublicate(inp::AbstractArray)
    len = counter(inp)
    ret = Array{eltype(inp)}(undef, 2*len)
    ret[1:len] .= inp
    ret[len+1:end] .= inp
    return ret
end

println(dublicate("weregrz"))
println(dublicate([2,4,5.6]))

#Exercise 4
t = [[1, 2], [3], [4, 5, 6]]
function nestedsum(tocount)
    counter=0
    for element in tocount
        if length(element)==1
            counter+=element[1]
        else 
            counter+=nestedsum(element)
        end 
    end
    return counter
end
nestedsum(t)

#Exercise 5
arr=[2,3,6,7,8,9,1,3]
arr2=[2,3,6,7,8,9,1]
function dublicate(in)
    b=Set(in)
    if length(b)==length(in)
        println("no dublicates")
    else
        println("dublicates")
    end
end
dublicate(arr)
dublicate(arr2)

#Exercise 6
mutable struct Point
    x::Float64
    y::Float64
end

mutable struct Circle
    center::Point
    radius::Real
end

function area(cen, rad)
    return pi*rad^2
end
circ = Circle(Point(1.0, 2.0), 2.4)
area(circ.center, circ.radius)

#Exercise 7
using RDatasets
using Statistics
using CSV
df = dataset("datasets","anscombe")
function subset(df)
    df2=df[df.X1 .< mean(df[:,1]), :]
    return df2
end
sub = subset(df)
CSV.write("try1.txt", sub)

#Exercise 8
using CSV
dataf = CSV.read("try1.txt")
dataf == sub

#Exercise 9
using RDatasets
df = dataset("car","Chile")
diffed= Set(df.Education)
for element in diffed
    b=[]
    for entry in 1:size(df, 1)
        #println(element, entry)
        if ismissing(df.Education[entry])
            if ismissing(element)
                push!(b,1)
            else 
                push!(b,0)
            end
        elseif ismissing(element)
            push!(b,0)
        elseif df.Education[entry]==element
            push!(b,1)
        else
            push!(b,0)
        end
    end
    df[!, Symbol(element)] = b
end


#Exercise 9 with general function
using RDatasets

function subset_df(df, col)
    diffed= Set(df[col])
    for element in diffed
        b=[]
        for entry in 1:size(df, 1)
            if ismissing(df[entry,col])
                if ismissing(element)
                    push!(b,1)
                else    
                    push!(b,0)
                end
            elseif ismissing(element)
                push!(b,0)
            elseif df[entry,col]==element
                push!(b,1)
            else
                push!(b,0)
            end
        end
        df[!, Symbol(element)] = b
    end
    return df
end

#run
df = dataset("car","Chile")
col=5
subset_df(df,col)

#Exercise 10
using Plots
scatter(df.Age, df.StatusQuo, title= "Plot of Age vs. StatusQuo", xlab="Age", ylab="StatusQuo")