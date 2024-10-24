import onnx
from onnx import helper, shape_inference

def replace_reshape_transpose(model):
    graph = model.graph
    nodes = graph.node

    # Build mappings from tensor names to nodes
    tensor_producers = {}
    tensor_consumers = {}
    for node in nodes:
        for output_name in node.output:
            tensor_producers[output_name] = node
        for input_name in node.input:
            tensor_consumers.setdefault(input_name, []).append(node)

    nodes_to_remove = []
    nodes_to_add = []

    for idx, transpose_node in enumerate(list(nodes)):
        print("node")
        if transpose_node.op_type != 'Transpose':
            continue

        transpose_input = transpose_node.input[0]
        transpose_output = transpose_node.output[0]

        # Check if the input comes from a Reshape node
        reshape_node = tensor_producers.get(transpose_input)
        if reshape_node is None or reshape_node.op_type != 'Reshape':
            continue

        # Ensure that the Reshape node's output is only consumed by this Transpose node
        consumers_of_reshape_output = tensor_consumers.get(reshape_node.output[0], [])
        if len(consumers_of_reshape_output) != 1:
            continue  # Can't replace if Reshape output has other consumers

        # Collect inputs for the custom node
        reshape_inputs = reshape_node.input
        custom_node_output = transpose_output

        # Get attributes from the Transpose node
        transpose_perm = None
        for attr in transpose_node.attribute:
            if attr.name == 'perm':
                transpose_perm = attr.ints

        # Create attributes for the custom node
        custom_attrs = {}
        if transpose_perm is not None:
            custom_attrs['perm'] = transpose_perm

        # Create the custom ReshapeTrans node
        custom_node = helper.make_node(
            'ReshapeTrans',
            inputs=reshape_inputs,
            outputs=[custom_node_output],
            name='ReshapeTrans_' + str(idx),
            **custom_attrs
        )

        # Record nodes to remove
        nodes_to_remove.extend([reshape_node, transpose_node])

        # Insert the new node at the position of the Transpose node
        nodes_to_add.append((idx, custom_node))

        # Update the mappings
        # Remove old producer mappings
        tensor_producers.pop(reshape_node.output[0], None)
        tensor_producers.pop(transpose_node.output[0], None)
        # Add new producer mapping
        tensor_producers[custom_node_output] = custom_node

        # Update consumer mappings
        for input_name in reshape_node.input:
            consumers = tensor_consumers.get(input_name, [])
            if reshape_node in consumers:
                consumers.remove(reshape_node)
        for input_name in transpose_node.input:
            consumers = tensor_consumers.get(input_name, [])
            if transpose_node in consumers:
                consumers.remove(transpose_node)
        for input_name in custom_node.input:
            tensor_consumers.setdefault(input_name, []).append(custom_node)

    # Insert new nodes into the graph
    c = 0
    for idx, custom_node in nodes_to_add:
        graph.node.insert(idx + c, custom_node)
        c += 1

    # Remove old nodes from the graph
    for node in nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)

    # (Optional) Infer shapes to ensure consistency
    #model = shape_inference.infer_shapes(model)

    return model

# Load the original model
model = onnx.load('Bert.onnx')

# Apply the pattern replacement
model = replace_reshape_transpose(model)

# Check the model for correctness
#onnx.checker.check_model(model)

# Save the modified model
onnx.save(model, 'Bert_replaced.onnx')
