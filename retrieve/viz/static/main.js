
$(document).ready(function() {

  var socket = io.connect('http://' + document.domain + ':' + location.port);

  socket.on('heatmap', function(data) {

    console.log(data);

    const maxWidth = 1000;
    const maxHeight = 400;
    const cellSizeMin = 3.5;

    const nrow = data.nrow;
    const ncol = data.ncol;

    var cellSize = Math.min(maxWidth / ncol, maxHeight / nrow);
    var cellSizeDisp = Math.max(cellSize, cellSizeMin);    

    var margin = {top: 50, right: 100, bottom: 20, left: 100},
	width = (ncol * cellSize), // - margin.left - margin.right,
	height = (nrow * cellSize); // - margin.top - margin.bottom;

    console.log(width, height, cellSize, cellSizeDisp);

    // clear up previous svg
    d3.select("#heatmap-viz").select("svg").remove();

    // append the svg object to the body of the page
    var heatmap = d3.select("#heatmap-viz")
	.append("svg")
	.attr("width", width + margin.left + margin.right)
	.attr("height", height + margin.top + margin.bottom)
	.append("g")
	.attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // create scales
    var xScale = d3.scaleBand().range([0, width])
	.domain(d3.range(0, ncol));
    var yScale = d3.scaleBand().range([height, 0])
	.domain(d3.range(0, nrow));

    // Create axes
    var xTicks = [],
    	yTicks = [];
    data.points.forEach(function(p){
      xTicks[p.col] = p.col_id;
      yTicks[p.row] = p.row_id;
    });
    var xAxis = d3.axisTop().scale(xScale)
    	.tickValues(xScale.domain().filter((p, i) => i in xTicks))
    	.tickFormat(i => xTicks[i])
	.tickPadding(10)
    	.tickSize(0);
    var yAxis = d3.axisLeft().scale(yScale)
    	.tickValues(yScale.domain().filter((p, i) => i in yTicks))
    	.tickFormat(i => yTicks[i])
	.tickPadding(10)
	.tickSize(0);
    heatmap.append("g")
      .style("font-size", 35)
      .attr("id", "xAxis")
      .call(xAxis)
      .select(".domain").remove();
    heatmap.append("g")
      .style("font-size", 35)
      .attr("id", "yAxis")
      .call(yAxis)
      .select(".domain").remove();

    /** add xAxis label */
    heatmap.append("text")
      .attr('x', width / 2)
      .attr('y', 20 - margin.top)
      .style("text-anchor", "middle")
      .text(data.colName);

    /** add yAxis label */
    heatmap.append("text")
      .attr('transform', 'rotate(-90)')
      .attr('y', 0 - margin.left)
      .attr('x', 0 - (height / 2))
      .attr('dy', '1em')
      .style('text-anchor', 'middle')
      .text(data.rowName);

    /** hide ticks */
    heatmap.select('#xAxis').selectAll('g.tick').style('opacity', 0);
    heatmap.select('#yAxis').selectAll('g.tick').style('opacity', 0);

    /** add ids to ticks for selectability */
    heatmap.select('#xAxis').selectAll('.tick').attr('id', d => 'xAxis-' + d);
    heatmap.select('#yAxis').selectAll('.tick').attr('id', d => 'yAxis-' + d);

    // build color scale
    var myColor = d3.scalePow()
	.exponent(1) // TODO: nonlinear scale
	.range(["white", "#BD262B"])
	.domain([0, 1]);

    // tooltip
    var tooltip = d3.select("body")
	.append('div')
	.attr('class', 'tooltip')
	.style("position", "absolute")
	.style("z-index", "10")
	.style('opacity', 0)
	.style('font-size', 35)
	.style("background-color", "white")
	.style("border", "solid")
	.style("border-width", "2px")
	.style("border-radius", "5px")
	.style("padding", "5px");

    // vertical-horizontal lines to highlight coordinate
    var vLine = heatmap
	.append('g')
	.append('line')
	.style('stroke', 'black')
	.style('opacity', 0);

    var hLine = heatmap
	.append('g')
	.append('line')
	.style('stroke', 'black')
	.style('opacity', 0);

    var mouseover = function(p) {
      // highlight rect
      d3.select(this)
	.style("stroke", "black")
	.style("opacity", 1);
      // tooltip
      tooltip.style('opacity', 1);
      // add opacity to lines
      vLine.style('opacity', 1);
      hLine.style('opacity', 1);
      // add visiblity to tick
      d3.select('#xAxis-' + p.col).style('opacity', 1);
      d3.select('#yAxis-' + p.row).style('opacity', 1);      
    };

    var mousemove = function(p) {
      // tooltip
      tooltip.html('Sim: ' + String(round(p.sim)))
	.style("top", (d3.event.pageY + 10) + "px")
        .style("left", (d3.event.pageX + 10) + "px");
      // reposition line
      vLine
	.attr("x1", xScale(p.col) + cellSizeDisp / 2)
	.attr("y1", yScale(p.row))
	.attr("x2", xScale(p.col) + cellSizeDisp / 2)
	.attr("y2", 0);
      hLine
	.attr("x1", xScale(p.col))
	.attr("y1", yScale(p.row) + cellSizeDisp / 2)
	.attr("x2", 0)
	.attr("y2", yScale(p.row) + cellSizeDisp / 2);
    };

    var mouseleave = function(p) {
      // reset opacity of rect
      d3.select(this)
	.style("stroke", "none")
	.style("opacity", 0.8);
      // tooltip
      tooltip.style('opacity', 0);
      // remove opacity from lines
      vLine.style('opacity', 0);
      hLine.style('opacity', 0);
      // remove opacity from ticks
      d3.select('#xAxis-' + p.col).style('opacity', 0);
      d3.select('#yAxis-' + p.row).style('opacity', 0);
    };

    heatmap.selectAll()
      .data(data.points)     /** no "if-function" needed in this case */
      .enter()
      .append("rect")
      .attr("x", p => xScale(p.col))
      .attr("y", p => yScale(p.row))
      .attr("rx", 1)
      .attr("ry", 1)
      .attr("width", cellSizeDisp)
      .attr("height", cellSizeDisp)
      .style("fill", p => myColor(p.sim))
      .style("stroke-width", 1)
      .style("stroke", "none")
      .style("opacity", 0.75)
      .on("mouseover", mouseover)
      .on("mousemove", mousemove)
      .on("mouseleave", mouseleave);
  });

});
