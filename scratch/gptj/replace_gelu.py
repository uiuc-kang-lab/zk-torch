import onnx
from onnx import helper, shape_inference

def replace_gelu(model):
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

    final_nodes_to_remove = []
    nodes_to_add = []

    for idx, mul_node in enumerate(list(nodes)):
        mul_node_0 = None
        mul_node_1 = None
        add_node_0 = None
        tanh_node_0 = None
        mul_node_2 = None
        add_node_1 = None
        mul_node_3 = None
        pow_node_0 = None

        nodes_to_remove = []
        if mul_node.op_type != 'Mul':
            nodes_to_remove = []
            continue
        mul_node_0 = mul_node
        nodes_to_remove.append(mul_node)
    
        mul_input = mul_node.input
        gelu_output = mul_node.output[0]
        gelu_inputs = []

        # Check if the input comes from a Reshape node
        input_name_0 = mul_input[0]
        input_node = tensor_producers.get(input_name_0)
        
        if input_node.op_type != 'Mul':
            nodes_to_remove = []
            continue
        else:
            mul_node = input_node
            mul_node_1 = mul_node
            nodes_to_remove.append(mul_node)
            gelu_inputs.append(mul_node.input[0])

        input_name_1 = mul_input[1]
        input_node = tensor_producers.get(input_name_1)
        
        if input_node.op_type != 'Add':
            nodes_to_remove = []
            continue
        else:
            add_node = input_node
            add_node_0 = add_node
            nodes_to_remove.append(add_node)
            add_input = add_node.input[0]
            input_node = tensor_producers.get(add_input)
            if input_node.op_type != 'Tanh':
                nodes_to_remove = []
                continue
            else:
                tanh_node = input_node
                tanh_node_0 = tanh_node
                nodes_to_remove.append(tanh_node)
                tanh_input = tanh_node.input[0]
                input_node = tensor_producers.get(tanh_input)
                if input_node.op_type != 'Mul':
                    nodes_to_remove = []
                    continue
                else:
                    mul_node = input_node
                    mul_node_2 = mul_node
                    nodes_to_remove.append(mul_node)
                    mul_input = mul_node.input[0]
                    input_node = tensor_producers.get(mul_input)
                    if input_node.op_type != 'Add':
                        nodes_to_remove = []
                        continue
                    else:
                        add_node = input_node
                        add_node_1 = add_node
                        nodes_to_remove.append(add_node)
                        add_input = add_node.input[1]
                        gelu_inputs.append(add_node.input[0])
                        input_node = tensor_producers.get(add_input)
                        if input_node.op_type != 'Mul':
                            nodes_to_remove = []
                            continue
                        else:
                            mul_node = input_node
                            mul_node_3 = mul_node
                            nodes_to_remove.append(mul_node)
                            mul_input = mul_node.input[0]
                            input_node = tensor_producers.get(mul_input)
                            if input_node.op_type != 'Pow':
                                nodes_to_remove = []
                                continue
                            else:
                                pow_node = input_node
                                pow_node_0 = pow_node
                                nodes_to_remove.append(pow_node)
                                gelu_inputs.append(pow_node.input[0])

        # all of them should be the same
        if len(gelu_inputs) != 3 or gelu_inputs[0] != gelu_inputs[1] or gelu_inputs[0] != gelu_inputs[2]:
            nodes_to_remove = []
            continue

        custom_gelu_input = gelu_inputs[0]
        custom_gelu_output = gelu_output

        custom_node = helper.make_node(
            'Gelu',
            inputs=[custom_gelu_input],
            outputs=[custom_gelu_output],
            name='Gelu_' + mul_node.name
        )

        final_nodes_to_remove.extend(nodes_to_remove)
        nodes_to_add.append((idx, custom_node))

        update_node_list = [mul_node_0, mul_node_1, add_node_0, tanh_node_0, mul_node_2, add_node_1, mul_node_3, pow_node_0]

        # Update the mappings
        # Remove old producer mappings
        for node in update_node_list:
            tensor_producers.pop(node.output[0], None)
            for input_name in node.input:
                consumers = tensor_consumers.get(input_name, [])
                if node in consumers:
                    consumers.remove(node)

        # Add new producer mapping
        tensor_producers[custom_gelu_output] = custom_node
        for input_name in custom_node.input:
            tensor_consumers.setdefault(input_name, []).append(custom_node)

    # Insert new nodes into the graph
    c = 0
    for idx, custom_node in nodes_to_add:
        graph.node.insert(idx + c, custom_node)
        c += 1

    # Remove old nodes from the graph
    for node in final_nodes_to_remove:
        if node in graph.node:
            graph.node.remove(node)
    
    # (Optional) Infer shapes to ensure consistency
    #model = shape_inference.infer_shapes(model)
    return model

# Load the original model
model = onnx.load('GPTj.onnx', load_external_data=False)

# Apply the pattern replacement
model = replace_gelu(model)

# Save the modified model
onnx.save(model, 'GPTj_gelu.onnx')
