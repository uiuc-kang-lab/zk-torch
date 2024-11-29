import onnx
from onnx import shape_inference
from onnx import helper, TensorProto

# Load the ONNX model
model = onnx.load('resnet50_fixed.onnx')

# Perform shape inference
inferred_model = shape_inference.infer_shapes(model)

# Access the graph
graph = inferred_model.graph

# Create a dictionary to map tensor names to their shapes
tensor_shapes = {}

# Helper function to extract shape from a tensor type
def get_shape(tensor_type):
    return [dim.dim_value if (dim.dim_value > 0) else 'None' for dim in tensor_type.shape.dim]

# Collect shapes from value_info
for value_info in graph.value_info:
    tensor_name = value_info.name
    tensor_type = value_info.type.tensor_type
    tensor_shapes[tensor_name] = get_shape(tensor_type)

# Include input and output shapes
for input_value in graph.input:
    tensor_name = input_value.name
    tensor_type = input_value.type.tensor_type
    tensor_shapes[tensor_name] = get_shape(tensor_type)

for output_value in graph.output:
    tensor_name = output_value.name
    tensor_type = output_value.type.tensor_type
    tensor_shapes[tensor_name] = get_shape(tensor_type)

# Now, for each node, get the input shapes
shapes = []
for node in graph.node:
    print(f"Node Name: {node.name}, Op Type: {node.op_type}")
    for input_name in node.input:
        shape = tensor_shapes.get(input_name)
        if shape is not None:
            print(f"  Input Name: {input_name}, Shape: {shape}")
            shapes.append(shape)
    print()

# Define the new node type you want to use instead of Conv, for example, "Relu"
new_node_type = "Conv2D"
new_node_type_conv1 = "LargeConv2D"
new_node_type1 = "MaxPool2D"
insert_index, reshape_input, reshape_output = None, None, None
# Replace each Conv node
for i, node in enumerate(graph.node):
    if node.op_type == "Conv":
        # Create a new node with the same inputs, outputs, and attributes as the Conv node
        new_node = None
        new_node = helper.make_node(
            new_node_type,
            inputs=node.input,
            outputs=node.output,
            name=node.name
        )

        #attr_shape = helper.make_attribute("orig_input_shape", shapes[i])
        #
        # Copy all attributes from the original Conv node to the new node
        for attr in node.attribute:
            new_node.attribute.extend([attr])
        #new_node.attribute.extend([attr_shape])

        # Replace the node in the graph
        graph.node.remove(node)
        graph.node.insert(i, new_node)
    
    if node.op_type == "MaxPool":
        # Create a new node with the same inputs, outputs, and attributes as the Conv node
        new_node = helper.make_node(
            new_node_type1,
            inputs=node.input,
            outputs=node.output,
            name=node.name
        )

        attr_shape = helper.make_attribute("in_channel", 64)
        #
        # Copy all attributes from the original Conv node to the new node
        for attr in node.attribute:
            new_node.attribute.extend([attr])
        new_node.attribute.extend([attr_shape])

        # Replace the node in the graph
        graph.node.remove(node)
        graph.node.insert(i, new_node)
    
    if node.op_type == "ReduceMean" or node.op_type == "Squeeze":
        if node.op_type == "ReduceMean":
            insert_index = i
            reshape_input = node.input[0]
            reshape_output = node.output[0]
            node.input[0] = node.output[0]
            node.output[0] = "new_" + node.output[0]
        
        if node.op_type == "Squeeze":
            node.input[0] = "new_" + node.input[0]

        for attr in node.attribute:
            if attr.name == "axes":
                node.attribute.remove(attr)
                break
        attr = helper.make_attribute("axes", [2])
        node.attribute.extend([attr])


# Define the shape tensor for the reshape operation
reshape_shape_name = "reshape_shape_tensor"
new_shape = [1, 2048, 49]
reshape_shape_tensor = helper.make_tensor(
    name=reshape_shape_name,
    data_type=TensorProto.INT64,
    dims=[len(new_shape)],
    vals=new_shape,
)
# Add the shape tensor to the graph's initializer
graph.initializer.append(reshape_shape_tensor)
# Create the Reshape node
reshape_node = helper.make_node(
    "Reshape",
    inputs=[reshape_input, reshape_shape_name],
    outputs=[reshape_output],
    name="Reshape_Node"
)
# Add the Reshape node to the graph
graph.node.insert(insert_index, reshape_node)

# Access the model's input (assuming it’s the first input here)
input_tensor = inferred_model.graph.input[0]

# Update the input shape from [N, C, H, W] to [N, C, HW]
# For example, if the original shape was [1, 1, 5, 5], change it to [1, 1, 25]
input_tensor.type.tensor_type.shape.dim[1].dim_value = input_tensor.type.tensor_type.shape.dim[1].dim_value * input_tensor.type.tensor_type.shape.dim[2].dim_value * input_tensor.type.tensor_type.shape.dim[3].dim_value
# Remove the last dimension (W) as it's now flattened into a single dimension
del input_tensor.type.tensor_type.shape.dim[2]
del input_tensor.type.tensor_type.shape.dim[2]

for node in graph.node:
    print(f"Node Name: {node.name}, Op Type: {node.op_type}")
# Save the modified model
onnx.save(inferred_model, "modified_resnet.onnx")
