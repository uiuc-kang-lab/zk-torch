import onnx
from onnx import helper, shape_inference

def replace_for_rope(model):
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

    for idx, node in enumerate(list(nodes)):
        nodes_to_remove = []
        if node.op_type != 'Reshape':
            nodes_to_remove = []
            continue
        reshape_node_0 = node
        nodes_to_remove.append(reshape_node_0)
    
        reshape_input = reshape_node_0.input
        rope_output = reshape_node_0.output[0]
        rope_inputs = []

        # Check if the input 1 comes from a Concat node
        input_name_0 = reshape_input[0]
        input_name_1 = reshape_input[1]
        input_node_0 = tensor_producers.get(input_name_0)
        input_node_1 = tensor_producers.get(input_name_1)
        
        shape_input = None
        if input_node_1.op_type != 'Concat':
            nodes_to_remove = []
            continue
        else:
            concat_node = input_node_1
            nodes_to_remove.append(concat_node)
            concat_input = concat_node.input[0]
            input_node = tensor_producers.get(concat_input)
            if input_node == None or input_node.op_type != 'Slice':
                nodes_to_remove = []
                continue
            else:
                slice_node = input_node
                nodes_to_remove.append(slice_node)
                slice_input = slice_node.input[0]
                input_node = tensor_producers.get(slice_input)
                if input_node.op_type != 'Shape':
                    nodes_to_remove = []
                    continue
                else:
                    shape_node = input_node
                    nodes_to_remove.append(shape_node)
                    shape_input = shape_node.input[0]
        
        if input_node_0.op_type != 'Concat' or shape_input != input_name_0:
            nodes_to_remove = []
            continue
        else:
            concat_node = input_node_0
            nodes_to_remove.append(concat_node)
            concat_input_0 = concat_node.input[0]
            concat_input_1 = concat_node.input[1]
            input_node_0 = tensor_producers.get(concat_input_0)
            input_node_1 = tensor_producers.get(concat_input_1)

            if input_node_0.op_type != 'Unsqueeze':
                nodes_to_remove = []
                continue
            else:
                unsqueeze_node = input_node_0
                nodes_to_remove.append(unsqueeze_node)
                unsqueeze_input = unsqueeze_node.input[0]
                input_node = tensor_producers.get(unsqueeze_input)
                if input_node.op_type != 'Neg':
                    nodes_to_remove = []
                    continue
                else:
                    neg_node = input_node
                    nodes_to_remove.append(neg_node)
                    neg_input = neg_node.input[0]
                    input_node = tensor_producers.get(neg_input)
                    if input_node.op_type != 'Slice':
                        nodes_to_remove = []
                        continue
                    else:
                        slice_node = input_node
                        nodes_to_remove.append(slice_node)
                        slice_input = slice_node.input[0]
                        rope_inputs.append(slice_input)
            
            if input_node_1.op_type != 'Unsqueeze':
                nodes_to_remove = []
                continue
            else:
                unsqueeze_node = input_node_1
                nodes_to_remove.append(unsqueeze_node)
                unsqueeze_input = unsqueeze_node.input[0]
                input_node = tensor_producers.get(unsqueeze_input)
                if input_node.op_type != 'Slice':
                    nodes_to_remove = []
                    continue
                else:
                    slice_node = input_node
                    nodes_to_remove.append(slice_node)
                    slice_input = slice_node.input[0]
                    rope_inputs.append(slice_input)


        # all of them should be the same
        if len(rope_inputs) != 2 or rope_inputs[0] != rope_inputs[1]:
            nodes_to_remove = []
            continue

        custom_rope_input = rope_inputs[0]
        custom_rope_output = rope_output

        custom_node = helper.make_node(
            'RopeRotate',
            inputs=[custom_rope_input],
            outputs=[custom_rope_output],
            name='RopeRotate_' + reshape_node_0.name.split('_')[-1]
        )

        final_nodes_to_remove.extend(nodes_to_remove)
        nodes_to_add.append((idx, custom_node))

        update_node_list = nodes_to_remove

        # Update the mappings
        # Remove old producer mappings
        for node in update_node_list:
            tensor_producers.pop(node.output[0], None)
            for input_name in node.input:
                consumers = tensor_consumers.get(input_name, [])
                if node in consumers:
                    consumers.remove(node)

        # Add new producer mapping
        tensor_producers[custom_rope_output] = custom_node
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

def replace_for_rope_mul(model):
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

    for idx, node in enumerate(list(nodes)):

        if node.op_type != 'Mul':
            nodes_to_remove = []
            continue
        else:
            mul_node = node
            nodes_to_remove.append(mul_node)
            mul_input_0 = mul_node.input[0]
            input_node_0 = tensor_producers.get(mul_input_0)
            if input_node_0.op_type != 'RopeRotate':
                nodes_to_remove = []
                continue

            mul_input_1 = mul_node.input[1]
            input_node = tensor_producers.get(mul_input_1)
            if input_node.op_type != 'Unsqueeze':
                nodes_to_remove = []
                continue
            else:
                shape_node = helper.make_node(
                    'Shape',
                    inputs=[mul_input_1],
                    outputs=['Shape_' + mul_input_1],
                    name='Shape_R_' + mul_input_1
                )
                ropeconst_node = helper.make_node(
                    "RopeConst",
                    inputs=['Shape_' + mul_input_1],
                    outputs=['RopeConst_' + mul_input_1],
                    name='RopeConst_R_' + mul_input_1
                )
                mul_node_1 = helper.make_node(
                    'Mul',
                    inputs=['RopeConst_' + mul_input_1, mul_input_1],
                    outputs=['Mul_RopeConst_' + mul_input_1],
                    name='Mul_RopeConst_R_' + mul_input_1
                )
                mul_node_2 = helper.make_node(
                    'Mul',
                    inputs=['Mul_RopeConst_' + mul_input_1, mul_input_0],
                    outputs=[node.output[0]],
                    name='Mul_RopeConst_R1_' + mul_input_1
                )
                nodes_to_add.append((idx, shape_node))
                nodes_to_add.append((idx, ropeconst_node))
                nodes_to_add.append((idx, mul_node_1))
                nodes_to_add.append((idx, mul_node_2))

        final_nodes_to_remove.extend(nodes_to_remove)
        update_node_list = nodes_to_remove

        # Update the mappings
        # Remove old producer mappings
        for node in update_node_list:
            tensor_producers.pop(node.output[0], None)
            for input_name in node.input:
                consumers = tensor_consumers.get(input_name, [])
                if node in consumers:
                    consumers.remove(node)

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
model = onnx.load('GPTj_gelu.onnx', load_external_data=False)

# Apply the pattern replacement
model = replace_for_rope(model)
model = replace_for_rope_mul(model)

# Save the modified model
onnx.save(model, 'GPTj_gelu_rope.onnx')
