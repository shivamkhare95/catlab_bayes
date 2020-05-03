module TestGraphviz

using Test
import LightGraphs, MetaGraphs
using LightGraphs: add_vertex!, add_vertices!, add_edge!
using MetaGraphs: MetaGraph, MetaDiGraph

using Catlab.Graphics.Graphviz

# MetaGraphs
############

# Undirected simple graph
g = MetaGraph()
add_vertex!(g, :label, "v")
add_vertices!(g, 2)
add_edge!(g, 1, 2, :xlabel, "e")
gv = to_graphviz(g)::Graph
@test !gv.directed
nodes = filter(s -> isa(s,Node), gv.stmts)
edges = filter(s -> isa(s,Edge), gv.stmts)
@test length(nodes) == 3
@test length(edges) == 1
@test nodes[1].attrs[:label] == "v"
@test edges[1].attrs[:xlabel] == "e"

# Directed simple graph
g = MetaDiGraph()
add_vertices!(g, 3); add_edge!(g, 1, 2); add_edge!(g, 2, 3)
gv = to_graphviz(g)::Graph
@test gv.directed
@test length(filter(s -> isa(s,Node), gv.stmts)) == 3
@test length(filter(s -> isa(s,Edge), gv.stmts)) == 2

# Directed multigraph
g = MetaDiGraph()
add_vertices!(g, 2)
add_edge!(g, 1, 2, :edges, [Dict(:label => "e1"), Dict(:label => "e2")])
gv = to_graphviz(g; multigraph=true)::Graph
@test gv.directed
nodes = filter(s -> isa(s,Node), gv.stmts)
edges = filter(s -> isa(s,Edge), gv.stmts)
@test length(nodes) == 2
@test length(edges) == 2
@test edges[1].attrs[:label] == "e1"
@test edges[2].attrs[:label] == "e2"

# Pretty-print
##############

spprint(expr::Expression) = sprint(pprint, expr)

# Node statement
@test spprint(Node("n")) == "n;"
@test spprint(Node("n"; label="foo")) == "n [label=\"foo\"];"
@test spprint(Node("n"; shape="box", style="filled")) ==
  "n [shape=\"box\",style=\"filled\"];"

# Edge statement
@test spprint(Edge("n1","n2")) == "n1 -- n2;"
@test spprint(Edge("n1","n2"; label="foo")) == "n1 -- n2 [label=\"foo\"];"
@test spprint(Edge("n1","n2"; style="dotted", weight="10")) ==
  "n1 -- n2 [style=\"dotted\",weight=\"10\"];"
@test spprint(Edge(NodeID("n1","p1"), NodeID("n2","p2"))) == "n1:p1 -- n2:p2;"
@test spprint(Edge(NodeID("n1","p1"), NodeID("n2","p2"); label="bar")) ==
  "n1:p1 -- n2:p2 [label=\"bar\"];"
@test spprint(Edge(NodeID("n1","p1","w"), NodeID("n2", "p2", "e"))) ==
  "n1:p1:w -- n2:p2:e;"
@test spprint(Edge("n1","n2","n3")) == "n1 -- n2 -- n3;"
@test spprint(Edge(NodeID("n1"), NodeID("n2"), NodeID("n3"))) ==
  "n1 -- n2 -- n3;"

# Graph statement
graph = Graph("G",
  Node("n1"),
  Node("n2"),
  Edge("n1","n2")
)
@test spprint(graph) == """
graph G {
  n1;
  n2;
  n1 -- n2;
}
"""

graph = Digraph("G",
  Node("n1"),
  Node("n2"),
  Edge("n1","n2")
)
@test spprint(graph) == """
digraph G {
  n1;
  n2;
  n1 -> n2;
}
"""

graph = Digraph("G",
  Node("n1"; label="foo"),
  Node("n2"; label="bar"),
  Edge("n1","n2");
  graph_attrs = Attributes(:rankdir => "LR"),
  node_attrs = Attributes(:shape => "box", :style => "filled"),
  edge_attrs = Attributes(:style => "dotted")
)
@test spprint(graph) == """
digraph G {
  graph [rankdir="LR"];
  node [shape="box",style="filled"];
  edge [style="dotted"];
  n1 [label="foo"];
  n2 [label="bar"];
  n1 -> n2;
}
"""

# Subgraph statement
subgraph = Subgraph("sub", 
  Node("n1"),
  Node("n2"),
  Edge("n1","n2")
)
@test spprint(subgraph) == """
subgraph sub {
  n1;
  n2;
  n1 -- n2;
}"""

subgraph = Subgraph(
  Node("n1"),
  Node("n2"),
  graph_attrs = Attributes(:rank => "same")
)
@test spprint(subgraph) == """
{
  graph [rank="same"];
  n1;
  n2;
}"""

end
