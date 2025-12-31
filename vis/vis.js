// initialize global variables
var allNodes;
var allEdges;

var nodeColors;
var originalNodes;
var network;

var filter = {
    "item": "",
    "property": "",
    "value": [],
};

var options = {
    "configure": {
        "enabled": true,
        "filter": [
	    "physics"
        ]
    },
    "edges": {
        "color": {
	    "inherit": true
        },
        "smooth": {
	    "enabled": true,
	    "type": "dynamic"
        }
    },
    "interaction": {
        "dragNodes": true,
        "hideEdgesOnDrag": false,
        "hideNodesOnDrag": false
    },
    "physics": {
        "enabled": true,
        "stabilization": {
	    "enabled": true,
	    "fit": true,
	    "iterations": 1000,
	    "onlyDynamicEdges": false,
	    "updateInterval": 50
        }
    }
};


function neighbourhoodHighlight (params) {
    allNodes = nodes.get({ returnType: "Object" });

    // if something is selected:
    if (params.nodes.length > 0) {
	var highlightActive = true;

	var i, j;
	var selectedNode = params.nodes[0];
	var degrees = 2;

	// mark all nodes as hard to read.
	for (let nodeId in allNodes) {
	    // nodeColors[nodeId] = allNodes[nodeId].color;
	    allNodes[nodeId].color = "rgba(200,200,200,0.5)";

	    if (allNodes[nodeId].hiddenLabel === undefined) {
		allNodes[nodeId].hiddenLabel = allNodes[nodeId].label;
		allNodes[nodeId].label = undefined;
	    }
	}

	var connectedNodes = network.getConnectedNodes(selectedNode);
	var allConnectedNodes = [];

	// get the second degree nodes
	for (i = 1; i < degrees; i++) {
	    for (j = 0; j < connectedNodes.length; j++) {
		allConnectedNodes = allConnectedNodes.concat(
		    network.getConnectedNodes(connectedNodes[j])
		);
	    }
	}

	// all second degree nodes get a different color and their label back
	for (i = 0; i < allConnectedNodes.length; i++) {
	    allNodes[allConnectedNodes[i]].color = "rgba(150,150,150,0.75)";

	    if (allNodes[allConnectedNodes[i]].hiddenLabel !== undefined) {
		allNodes[allConnectedNodes[i]].label = allNodes[allConnectedNodes[i]].hiddenLabel;
		allNodes[allConnectedNodes[i]].hiddenLabel = undefined;
	    }
	}

	// all first degree nodes get their own color and their label back
	for (i = 0; i < connectedNodes.length; i++) {
	    allNodes[connectedNodes[i]].color = nodeColors[connectedNodes[i]];

	    if (allNodes[connectedNodes[i]].hiddenLabel !== undefined) {
		allNodes[connectedNodes[i]].label = allNodes[connectedNodes[i]].hiddenLabel;
		allNodes[connectedNodes[i]].hiddenLabel = undefined;
	    }
	}

	// the main node gets its own color and its label back.
	// allNodes[selectedNode].color = undefined;
	allNodes[selectedNode].color = nodeColors[selectedNode];

	if (allNodes[selectedNode].hiddenLabel !== undefined) {
	    allNodes[selectedNode].label = allNodes[selectedNode].hiddenLabel;
	    allNodes[selectedNode].hiddenLabel = undefined;
	}
    }
    else if (highlightActive === true) {
	// reset all nodes
	for (let nodeId in allNodes) {
	    allNodes[nodeId].color = nodeColors[nodeId];
	    // delete allNodes[nodeId].color;

	    if (allNodes[nodeId].hiddenLabel !== undefined) {
		allNodes[nodeId].label = allNodes[nodeId].hiddenLabel;
		allNodes[nodeId].hiddenLabel = undefined;
	    }
	}

	highlightActive = false;
    }

    // transform the object into an array
    var updateArray = [];

    if (params.nodes.length > 0) {
	for (let nodeId in allNodes) {
	    if (allNodes.hasOwnProperty(nodeId)) {
		updateArray.push(allNodes[nodeId]);
	    }
	}

	nodes.update(updateArray);
    }
    else {
	for (let nodeId in allNodes) {
	    if (allNodes.hasOwnProperty(nodeId)) {
		updateArray.push(allNodes[nodeId]);
	    }
	}

	nodes.update(updateArray);
    }
}


function filterHighlight (params) {
    allNodes = nodes.get({ returnType: "Object" });

    // if something is selected:
    if (params.nodes.length > 0) {
	var filterActive = true;
	let selectedNodes = params.nodes;

	// hiding all nodes and saving the label
	for (let nodeId in allNodes) {
	    allNodes[nodeId].hidden = true;

	    if (allNodes[nodeId].savedLabel === undefined) {
		allNodes[nodeId].savedLabel = allNodes[nodeId].label;
		allNodes[nodeId].label = undefined;
	    }
	}

	for (let i = 0; i < selectedNodes.length; i++) {
	    allNodes[selectedNodes[i]].hidden = false;

	    if (allNodes[selectedNodes[i]].savedLabel !== undefined) {
		allNodes[selectedNodes[i]].label = allNodes[selectedNodes[i]].savedLabel;
		allNodes[selectedNodes[i]].savedLabel = undefined;
	    }
	}

    }
    else if (filterActive === true) {
	// reset all nodes
	for (let nodeId in allNodes) {
	    allNodes[nodeId].hidden = false;

	    if (allNodes[nodeId].savedLabel !== undefined) {
		allNodes[nodeId].label = allNodes[nodeId].savedLabel;
		allNodes[nodeId].savedLabel = undefined;
	    }
	}

	filterActive = false;
    }

    // transform the object into an array
    var updateArray = [];

    if (params.nodes.length > 0) {
	for (let nodeId in allNodes) {
	    if (allNodes.hasOwnProperty(nodeId)) {
		updateArray.push(allNodes[nodeId]);
	    }
	}

	nodes.update(updateArray);
    }
    else {
	for (let nodeId in allNodes) {
	    if (allNodes.hasOwnProperty(nodeId)) {
		updateArray.push(allNodes[nodeId]);
	    }
	}

	nodes.update(updateArray);
    }
}


function selectNode (nodes) {
    network.selectNodes(nodes);
    neighbourhoodHighlight({ nodes: nodes });

    return nodes;
}


function selectNodes (nodes) {
    network.selectNodes(nodes);
    filterHighlight({nodes: nodes});

    return nodes;
}


function highlightFilter (filter) {
    let selectedNodes = []
    let selectedProp = filter['property']

    if (filter['item'] === 'node') {
	let allNodes = nodes.get({ returnType: "Object" });

	for (let nodeId in allNodes) {
	    if (allNodes[nodeId][selectedProp] && filter['value'].includes((allNodes[nodeId][selectedProp]).toString())) {
		selectedNodes.push(nodeId)
	    }
	}
    }
    else if (filter['item'] === 'edge'){
	let allEdges = edges.get({returnType: 'object'});

	// check if the selected property exists for selected edge and
	// select the nodes connected to the edge
	for (let edge in allEdges) {
	    if (allEdges[edge][selectedProp] && filter['value'].includes((allEdges[edge][selectedProp]).toString())) {
		selectedNodes.push(allEdges[edge]['from'])
		selectedNodes.push(allEdges[edge]['to'])
	    }
	}
    }

    selectNodes(selectedNodes)
}


// this method is responsible for drawing the graph, returns the drawn network
function drawGraph () {
    var container = document.getElementById('mynetwork');

    // parse the nodes and edges defined in Python
    var nodes = new vis.DataSet(NODES);
    var edges = new vis.DataSet(EDGES);
		  
    nodeColors = {};
    allNodes = nodes.get({ returnType: "Object" });

    for (nodeId in allNodes) {
        nodeColors[nodeId] = allNodes[nodeId].color;
    }

    allEdges = edges.get({ returnType: "Object" });

    // load the nodes and edges into the graph
    var data = {
	nodes: nodes,
	edges: edges,
    };
                
    // if the network needs to display a configure window, add its
    // <div/>
    options.configure["container"] = document.getElementById("config");

    network = new vis.Network(container, data, options);
                  
    network.on("stabilizationProgress", function(params) {
        document.getElementById('loadingBar').removeAttribute("style");

        var maxWidth = 496;
        var minWidth = 20;
        var widthFactor = params.iterations/params.total;
        var width = Math.max(minWidth,maxWidth * widthFactor);

        document.getElementById('bar').style.width = width + 'px';
        document.getElementById('text').innerHTML = Math.round(widthFactor*100) + '%';
    });

    network.once("stabilizationIterationsDone", function() {
        document.getElementById('text').innerHTML = '100%';
        document.getElementById('bar').style.width = '496px';
        document.getElementById('loadingBar').style.opacity = 0;

        // really clean the DOM element
        setTimeout(
	    function () {
		document.getElementById('loadingBar').style.display = 'none';
	    },
	    500
	);
    });

    return network;
}


drawGraph();
