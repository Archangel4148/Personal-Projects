from transitions.extensions import GraphMachine


class Model:
    pass


model = Model()

machine = GraphMachine(
    model=model,
    states=["A", "B", "C"],
    initial="A",
    graph_engine="graphviz",
)

machine.add_transition("go", "A", "B")
machine.add_transition("finish", "B", "C")

graph = model.get_graph()  # type: ignore[attr-defined]

graph.draw(
    "test.png",
    prog="dot",
)

print("Generated test.png")