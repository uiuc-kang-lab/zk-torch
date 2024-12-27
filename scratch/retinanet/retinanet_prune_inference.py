import onnx
from onnx.helper import make_model
from onnx import NodeProto

# Load the ONNX model
input_path = "../../RetinaNet.onnx"  # Replace with your input model path
output_path = "../../RetinaNet_pruned.onnx"  # Replace with your desired output path

model = onnx.load(input_path)
graph = model.graph

# Identify the index of the node to remove and its successors
node_to_remove_after_this = "Concat_900"  # Replace with the node name to remove
node_index = None

for i, node in enumerate(graph.node):
    if node.name == node_to_remove_after_this:
        node_index = i+1
        break

if node_index is None:
    raise ValueError(f"Node with name '{node_to_remove_after_this}' not found in the model.")

# Create a new list of nodes, excluding the ones after the specified node
new_nodes = graph.node[:node_index]  # Exclude the specified node and all after it
graph.ClearField("node")  # Clear the current nodes
graph.node.extend(new_nodes)  # Add back only the nodes we want to keep

# Save the modified model
onnx.save(model, output_path)

print(f"Modified model saved to {output_path}")
