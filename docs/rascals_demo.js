var margin = {
    top: 50,
    right: 50, 
    bottom: 50,
    left: 50
};

var width = 800 - margin.left - margin.right;
var height = 600 - margin.top - margin.bottom;

d3.csv("../notebooks/rascals_viz2.csv").then(function(data){

    data.forEach(function(d){
        d.RA_deg = +d.RA_deg
        d.Dec_deg = +d.Dec_deg
        d.naive_ra = +d.naive_ra
        d.naive_dec = +d.naive_dec
        d.theta_x_deg = +d.theta_x_deg
        d.theta_y_deg = +d.theta_y_deg
    })

    var bodies = d3.nest()
        .key(function(d){
            return d.name;
        })
        .entries(data);

    var exposures = d3.nest()
        .key(function(d){
            return d.exp_mjd;
        })
        .entries(data);

    var colors = d3.scaleOrdinal()
        .domain(function(d){
            return bodies.key;
        })

    console.log(colors)

    var svg = d3.select(".chart-area")
        .append("svg")
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom);

    var g = svg.append("g")
        .attr("transform", "translate(" + margin.left + ", " + margin.top + ")")

    var x = d3.scaleLinear()
        .domain([
            d3.min(data, function(d){
                return d.RA_deg;
            }),
            d3.max(data, function(d){
                return d.RA_deg;
            })])
        .range([0, width]);

    var y = d3.scaleLinear()
        .domain([
            d3.min(data, function(d){
                return d.Dec_deg;
            }),
            d3.max(data, function(d){
                return d.Dec_deg;
            })])
        .range([height, 0]);

    var leftAxis = d3.axisLeft(y);
    g.append("g")
        .attr("class", "left axis")
        .call(leftAxis);

    var rightAxis = d3.axisRight(y);
    g.append("g")
        .attr("class", "right axis")
        .attr("transform", "translate(" + width + ", 0)")
        .call(rightAxis);

    var bottomAxis = d3.axisBottom(x);
    g.append("g")
        .attr("class", "bottom axis")
        .attr("transform", "translate(0, " + height + ")")
        .call(bottomAxis);

    var topAxis = d3.axisTop(x);
    g.append("g")
        .attr("class", "top axis")
        .call(topAxis);


    var plotAll = function(){

        var circles = g.selectAll("circle")
            .data(data)
            
        circles.enter()
            .append("circle")
            .attr("cx", function(d){
                return x(d.RA_deg);
            })
            .attr("cy", function(d){
                return y(d.Dec_deg);
            })
            .attr("r", 1)
            .attr("fill", "blue");
    }

    var resetPlot = function(){
        x.domain([
            d3.min(data, function(d){
                return d.RA_deg;
            }),
            d3.max(data, function(d){
                return d.RA_deg;
            })])

        y.domain([
            d3.min(data, function(d){
                return d.Dec_deg;
            }),
            d3.max(data, function(d){
                return d.Dec_deg;
            })])

        g.select(".bottom")
            .transition()
            .duration(4000)
                .call(bottomAxis);
        g.select(".top")
            .transition()
            .duration(4000)
                .call(topAxis);
        g.select(".right")
            .transition()
            .duration(4000)
                .call(rightAxis);
        g.select(".left")
            .transition()
            .duration(4000)
                .call(leftAxis);

        var circles = g.selectAll("circle")

        circles
            .exit()
            .remove()
            
        circles
            .enter()
            .append("circle")

        circles
            .transition()
            .duration(5000)
            .attr("cx", function(d){
                return x(d.RA_deg);
            })
            .attr("cy", function(d){
                return y(d.Dec_deg);
            })
            .attr("r", 1)
            .attr("fill", "blue");
        
    }

    var plotNaive = function(){

        x.domain([
            d3.min(data, function(d){
                return d.naive_ra;
            }),
            d3.max(data, function(d){
                return d.naive_ra;
            })])

        y.domain([
            d3.min(data, function(d){
                return d.naive_dec;
            }),
            d3.max(data, function(d){
                return d.naive_dec;
            })])

        g.select(".bottom")
            .transition()
            .duration(4000)
                .call(bottomAxis);
        g.select(".top")
            .transition()
            .duration(4000)
                .call(topAxis);
        g.select(".right")
            .transition()
            .duration(4000)
                .call(rightAxis);
        g.select(".left")
            .transition()
            .duration(4000)
                .call(leftAxis);

        var circles = g.selectAll("circle")

        circles
            .exit()
            .remove()
            
        circles
            .enter()
            .append("circle")

        circles
            .transition()
            .duration(5000)
            .attr("cx", function(d){
                return x(d.naive_ra);
            })
            .attr("cy", function(d){
                return y(d.naive_dec);
            })
            .attr("r", 1)
        
    }

    var plotRobust = function(){

        x.domain([
            d3.min(data, function(d){
                return d.theta_x_deg;
            }),
            d3.max(data, function(d){
                return d.theta_x_deg;
            })])

        y.domain([
            d3.min(data, function(d){
                return d.theta_y_deg;
            }),
            d3.max(data, function(d){
                return d.theta_y_deg;
            })])
    

        g.select(".bottom")
            .transition()
            .duration(4000)
                .call(bottomAxis);
        g.select(".top")
            .transition()
            .duration(4000)
                .call(topAxis);
        g.select(".right")
            .transition()
            .duration(4000)
                .call(rightAxis);
        g.select(".left")
            .transition()
            .duration(4000)
                .call(leftAxis);

        var circles = g.selectAll("circle")

        circles
            .exit()
            .remove()
            
        circles
            .enter()
            .append("circle")

        circles
            .transition()
            .duration(5000)
            .attr("cx", function(d){
                return x(d.theta_x_deg);
            })
            .attr("cy", function(d){
                return y(d.theta_y_deg);
            })
            .attr("r", 1)
        
    }

    var updateColors = function(){
        var circles = g.selectAll("circle")

        circles
            .attr("fill", colors)
    }

    naiveButton = d3.select("#naive")
    naiveButton.on("click", function(){
        plotNaive()

    });

    robustButton = d3.select("#robust")
    robustButton.on("click", function(){
        plotRobust()

    });

    resetButton = d3.select("#reset")
    resetButton.on("click", function(){
        resetPlot()
    });

    colorButton = d3.select("#colorby")
    colorButton.on("click", function(){
        updateColors()
    });

    plotAll()
    
})