#plotting function for datasets 
import Plots
using Plots
using VegaDatasets
using DataFrames

function plotting(x, y, a, b, title="Plot", xlab="xlab", ylab="ylab")
    equation(d)=a*d+b
    plot(x, y, seriestype=:scatter, title=title, xlabel=xlab, ylabel=ylab)
    plot!(equation, minimum(x):maximum(x))
end

car = DataFrame(VegaDatasets.dataset("cars"))
plotting(car.Weight_in_lbs, car.Displacement, 0.1, -100, "Scatterplot of Cars", "Weight in lbs", "Displacement")



#projections exercise
r = [1,2,3,4] 
s = [5,6,7,8]

function sizevec(vec)
    sum=0
    for i in 1:length(vec)
        sum+=vec[i]^2
    end
    ss=sqrt(sum)
    return ss
end

function innerproduct(vec1, vec2)
    sum=0
    for i in 1:length(vec1)
        sum+=vec1[i]*vec2[i]
    end
    return sum
end

function projection(vec1, vec2)
    return (innerproduct(vec1, vec2) / sizevec(vec2))
end

sizevec(r)
sizevec(s)
innerproduct(r, s)
projection(r,s)
projection(s,r)



#loss function exercise from linear regression
using Random
using Plots
function lossfunction_L1(x, y, a, b)
    ssq=0
    for i in 1:length(x)
        y_line=a*x[i]+b
        ssq+=abs(y_line-y[i])
    end
    return ssq/length(y)
end

function lossfunction_L2(x, y, a, b) #Atrans*x
    ssq=0
    for i in 1:length(x)
        y_line=a*x[i]+b
        ssq+=(y_line-y[i])^2
    end
    return ssq/length(y)
end

function optimize(x,y,arange,brange)
    bestssq1=10000
    bestssq2=10000
    besta1=arange[1]
    bestb1=brange[1]
    besta2=arange[1]
    bestb2=brange[1]
    for a in arange
        for b in brange
            ssq_L1 = lossfunction_L1(x, y, a, b)
            ssq_L2 = lossfunction_L2(x, y, a, b)
            if ssq_L1<bestssq1
                besta1=a
                bestb1=b
                bestssq1=ssq_L1
            end
            if ssq_L2<bestssq2
                besta2=a
                bestb2=b
                bestssq2=ssq_L2
            end
        end
    end
    return (bestssq1, besta1, bestb1, bestssq2, besta2, bestb2)
end

#optimization and plot for random generated points
x = collect(1:10) .+ randn(10)
y = collect(1:10) .+ randn(10)
a = collect(range(0, length=1000, stop=2))
b = collect(range(-1, length=1000, stop=1))
bestssq1, besta1, bestb1, bestssq2, besta2, bestb2 = optimize(x, y, a, b)
plotting(x, y, besta1, bestb1, "Random generated plot - L1 as lossfunction")
plotting(x, y, besta2, bestb2, "Random generated plot - L2 as lossfunction")

#optimization and plot for the car dataset 
a = -50:0.01:50
b = -2000:10
bestssq1, besta1, bestb1, bestssq2, besta2, bestb2= optimize(car.Weight_in_lbs, car.Displacement, a, b)
plotting(car.Weight_in_lbs, car.Displacement, besta1, bestb1, "Scatterplot of Cars - L1 as lossfunction", "Weight in lbs", "Displacement") #a=0.11, b=-136
plotting(car.Weight_in_lbs, car.Displacement, besta2, bestb2, "Scatterplot of Cars - L2 as lossfunction", "Weight in lbs", "Displacement") #a=0.12, b=-163 