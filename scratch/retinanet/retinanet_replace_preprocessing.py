import onnx
from onnx.helper import make_node

# Load the ONNX model
input_path = "../../RetinaNet_pruned.onnx"  # Replace with your input model path
output_path = "../../RetinaNet_pruned.onnx"  # Replace with your desired output path

model = onnx.load(input_path)
graph = model.graph

# Identify the range of nodes to replace
start_index = 1  # Replace with the actual start index (i)
end_index = 16   # Replace with the actual end index (j)

if start_index < 0 or end_index >= len(graph.node) or start_index > end_index:
    raise ValueError("Invalid indices specified for the range of nodes to replace.")

# Get the input of the start node and the output of the end node
input_of_start_node = graph.node[start_index].input
output_of_end_node = graph.node[end_index].output

# Create a new Identity node
identity_node = make_node(
    "Identity",
    inputs=input_of_start_node,
    outputs=output_of_end_node,
    name="Identity_Replacement"
)

# Create a new list of nodes, excluding the range to be replaced
new_nodes = (
    graph.node[:start_index] +  # Nodes before the range
    [identity_node] +           # Add the new Identity node
    graph.node[end_index + 1:]  # Nodes after the range
)

graph.ClearField("node")  # Clear the current nodes
graph.node.extend(new_nodes)  # Add back the updated nodes

# Save the modified model
onnx.save(model, output_path)

print(f"Modified model saved to {output_path}")