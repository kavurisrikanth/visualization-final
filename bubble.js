/**
 * Method to generate a bubble chart.
 *
 * 
 */
var bubbleChart = function () {
	// Basic variables.
    var width = 960,
		height = 960,
		marginTop = 96,
		minRadius = 6,
		maxRadius = 20,
		columnForTitle = "Name",
		columnForRadius = "Value",
		columnForColors = "Color",
		forceApart = -70,
		unitName="Value",
		customRange,
		customDomain,
		chartSelection,
		chartSVG;

    /**
	 * This method is the nuts and bolts of the bubble chart generation algorithm.
    */
    function chart(selection){
        var data = selection.datum();
		chartSelection=selection;
		var div = selection,
			svg = div.selectAll('svg');
		svg.attr('width', width).attr('height', height);
		chartSVG=svg;
		var title = "Sample Bubble Chart";

		// Set up a basic div to put the SVG chart in.
		var tooltip = selection
		.append("div")
		.style("position", "absolute")
		.style("visibility", "hidden")
		.style("color", "white")
		.style("padding", "8px")
		.style("background-color", "#626D71")
		.style("border-radius", "6px")
		.style("text-align", "center")
		.style("font-family", "monospace")
		.style("width", "400px")
		.text("");

		// Set up animations for the bubbles.
		var simulation = d3.forceSimulation(data)
		.force("charge", d3.forceManyBody().strength([forceApart]))
		.force("x", d3.forceX())
		.force("y", d3.forceY())
		.on("tick", ticked);

		/*
		 * This method defines the movement of the bubble after it has been formed.
		*/
		function ticked(e) {
			node.attr("transform",function(d) {
				return "translate(" + [d.x+(width / 2), d.y+((height+marginTop) / 2)] +")";
			});
		}

		// Set up bubble color
		var colorCircles = d3.scaleOrdinal(d3.schemeCategory10);
		
		// Set up min and max bubble radius
		var minRadiusDomain = d3.min(data, function(d) {
			return +d[columnForRadius];
		});
		var maxRadiusDomain = d3.max(data, function(d) {
			return +d[columnForRadius];
		});

		// Set up scaled bubble radius
		var scaleRadius = d3.scaleLinear()
		.domain([minRadiusDomain, maxRadiusDomain])
		.range([minRadius, maxRadius])

		// Add data for each circle (bubble)
		var node = svg.selectAll("circle")
		.data(data)
		.enter()
		.append("g")
		.attr('transform', 'translate(' + [width / 2, height / 2] + ')')
		.style('opacity',1);
			
		// Set up the visual attributes for each bubble.
		// The attributes are: ID, Radius, Color, Mouse-over action, Mouse movement action,
		// and Mouse remove action
		node.append("circle")
		.attr("id",function(d,i) {
			return i;
		})
		.attr('r', function(d) {
			return scaleRadius(d[columnForRadius]);
		})
		.style("fill", function(d) {
			return colorCircles(d[columnForColors]);
		})
		.on("mouseover", function(d) {
			tooltip.html(d[columnForTitle] + "<br/>" + d[columnForColors] + "<br/>" + d[columnForRadius] + " "+ unitName);
			return tooltip.style("visibility", "visible");
		})
		.on("mousemove", function() {
			return tooltip.style("top", (d3.event.pageY - 10) + "px").style("left", (d3.event.pageX + 10) + "px");
		})
		.on("mouseout", function() {
			return tooltip.style("visibility", "hidden");
		});

		// Set up the clip path (paintable area) for the bubbles. 
		node.append("clipPath")
		.attr("id",function(d,i) {
			return "clip-"+i;
		})
		.append("use")
		.attr("xlink:href",function(d,i) {
			return "#" + i;
		});

		// Add title for entire chart.
		svg.append('text')
			.attr('x',width/2).attr('y',marginTop)
			.attr("text-anchor", "middle")
			.attr("font-size","1.8em")
			.text(title);
    }

	// Set up methods for attributes
    chart.width = chartWidth;
	chart.height = chartHeight;
	chart.title = chartTitle;
	chart.columnForColors = chartColForColors;
	chart.columnForRadius = chartColForRadius;
	chart.columnForTitle = chartColForTitle;
	chart.minRadius = chartMinRadius;
	chart.maxRadius = chartMaxRadius;
	chart.forceApart = chartForceApart;
	chart.unitName = chartUnitName;
	
	/*
	 * Chart width method.
	*/
    function chartWidth(value) {
		if (!arguments.length) {
			return width;
		}
		width = value;
		return chart;
	};

	/*
	 * Chart height method.
	*/
	function chartHeight(value) {
		if (!arguments.length) {
			return height;
		}
		height = value;
		marginTop=0.05*height;
		return chart;
	};

	/*
	 * Chart color method.
	*/
	function chartColForColors(value) {
		if (!arguments.length) {
			return columnForColors;
		}
		columnForColors = value;
		return chart;
	};

	/*
	 * Chart title method.
	*/
	function chartColForTitle(value) {
		if (!arguments.length) {
			return columnForTitle;
		}
		columnForTitle = value;
		return chart;
	};

	/*
	 * Chart radius method.
	*/
	function chartColForRadius (value) {
		if (!arguments.length) {
			return columnForRadius;
		}
		columnForRadius = value;
		return chart;
	};

	/*
	 * Chart minimum radius method.
	*/
	function chartMinRadius(value) {
		if (!arguments.length) {
			return minRadius;
		}
		minRadius = value;
		return chart;
	};

	/*
	 * Chart maximum radius method.
	*/
	function chartMaxRadius(value) {
		if (!arguments.length) {
			return maxRadius;
		}
		maxRadius = value;
		return chart;
	};

	/*
	 * Chart unit name method.
	*/
	function chartUnitName(value) {
		if (!arguments.length) {
			return unitName;
		}
		unitName = value;
		return chart;
	};

	/*
	 * Chart animation distance method.
	*/
	function chartForceApart(value) {
		if (!arguments.length) {
			return forceApart;
		}
		forceApart = value;
		return chart;
	};

	/*
	 * Chart title method.
	*/
	function chartTitle(value) {
		if (!arguments.length) {
			return title;
		}
		title = value;
		return chart;
	}

    return chart;
}