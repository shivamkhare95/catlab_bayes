module TestGraphvizWiringDiagrams

using Test

using Catlab.WiringDiagrams, Catlab.Graphics
import Catlab.Graphics: Graphviz

function stmts(graph::Graphviz.Graph, type::Type)
  [ stmt for stmt in graph.stmts if stmt isa type ]
end
function stmts(graph::Graphviz.Graph, type::Type, attr::Symbol)
  [ stmt.attrs[attr] for stmt in graph.stmts
    if stmt isa type && haskey(stmt.attrs, attr) ]
end

f = singleton_diagram(Box(:f, [:A], [:B]))
g = singleton_diagram(Box(:g, [:B], [:A]))

# Simple wiring diagrams, with default settings.
graph = to_graphviz(f)
@test graph isa Graphviz.Graph && graph.directed
@test stmts(graph, Graphviz.Node, :comment) == ["f"]
@test stmts(graph, Graphviz.Edge, :comment) == ["A","B"]
@test stmts(graph, Graphviz.Edge, :label) == []
@test stmts(graph, Graphviz.Edge, :xlabel) == []
@test length(stmts(graph, Graphviz.Subgraph)) == 2

graph = to_graphviz(singleton_diagram(Box(:h, Symbol[], [:A])))
@test stmts(graph, Graphviz.Node, :comment) == ["h"]
@test length(stmts(graph, Graphviz.Subgraph)) == 1

graph = to_graphviz(singleton_diagram(Box(:h, Symbol[], Symbol[])))
@test stmts(graph, Graphviz.Node, :comment) == ["h"]
@test length(stmts(graph, Graphviz.Subgraph)) == 0

graph = to_graphviz(compose(f,g))
@test stmts(graph, Graphviz.Node, :comment) == ["f","g"]

# Layout orientation.
for orientation in instances(LayoutOrientation)
  graph = to_graphviz(compose(f,g); orientation=orientation)
  @test stmts(graph, Graphviz.Node, :comment) == ["f","g"]
end

# Junction nodes.
graph = to_graphviz(add_junctions!(compose(f, mcopy(Ports([:B])))))
@test stmts(graph, Graphviz.Node, :comment) == ["f","junction"]

# Edge labels.
graph = to_graphviz(f; labels=true)
@test stmts(graph, Graphviz.Edge, :label) == ["A","B"]
@test stmts(graph, Graphviz.Edge, :xlabel) == []

graph = to_graphviz(f; labels=true, label_attr=:xlabel)
@test stmts(graph, Graphviz.Edge, :label) == []
@test stmts(graph, Graphviz.Edge, :xlabel) == ["A","B"]

# Anchoring of outer ports.
graph = to_graphviz(otimes(f,g); anchor_outer_ports=true)
@test stmts(graph, Graphviz.Node, :comment) == ["f","g"]
graph = to_graphviz(otimes(f,g); anchor_outer_ports=false)
@test stmts(graph, Graphviz.Node, :comment) == ["f","g"]

end
