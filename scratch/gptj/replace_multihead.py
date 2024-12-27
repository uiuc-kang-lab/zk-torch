import onnx
from onnx import helper, shape_inference

def replace_for_multihead(model):
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
    shape_dict = {}

    for idx, node in enumerate(list(nodes)):
        shapes = []
        nodes_to_remove = []
        if node.op_type != 'Reshape':
            nodes_to_remove = []
            continue
        reshape_node_0 = node
        nodes_to_remove.append(reshape_node_0)
    
        reshape_input = reshape_node_0.input
        original_matmul_inputs = None
        original_matmul_outputs = []
        multihead_output = reshape_node_0.output[0]
        multihead_input = reshape_input[1]

        # Check if the input 1 comes from a Concat node
        input_name_0 = reshape_input[0]
        original_matmul_outputs.append(input_name_0)
        input_name_1 = reshape_input[1]
        input_node_0 = tensor_producers.get(input_name_0)
        input_node_1 = tensor_producers.get(input_name_1)
        
        shape_input = None
        if input_node_1.op_type != 'Concat':
            nodes_to_remove = []
            continue
        else:
            concat_node = input_node_1
            concat_input_0 = concat_node.input[0]
            concat_input_1 = concat_node.input[1]
            input_node = tensor_producers.get(concat_input_0)
            if input_node == None or input_node.op_type != 'Unsqueeze':
                nodes_to_remove = []
                continue
            else:
                unsqueeze_input = input_node.input[0]
                input_node = tensor_producers.get(unsqueeze_input)
                if input_node.op_type != 'Gather':
                    nodes_to_remove = []
                    continue
                else:
                    gather_input = input_node.input[0]
                    input_node = tensor_producers.get(gather_input)
                    if input_node.op_type != 'Shape':
                        nodes_to_remove = []
                        continue
                    else:
                        shape_input = input_node.input[0]
                        original_matmul_outputs.append(shape_input)
                        shapes.append(input_node.name)
            
            input_node = tensor_producers.get(concat_input_1)
            if input_node == None or input_node.op_type != 'Unsqueeze':
                nodes_to_remove = []
                continue
            else:
                unsqueeze_input = input_node.input[0]
                input_node = tensor_producers.get(unsqueeze_input)
                if input_node.op_type != 'Gather':
                    nodes_to_remove = []
                    continue
                else:
                    gather_input = input_node.input[0]
                    input_node = tensor_producers.get(gather_input)
                    if input_node.op_type != 'Shape':
                        nodes_to_remove = []
                        continue
                    else:
                        shape_input = input_node.input[0]
                        original_matmul_outputs.append(shape_input)
                        shapes.append(input_node.name)
        
        if original_matmul_outputs[0] != original_matmul_outputs[1] or original_matmul_outputs[0] != original_matmul_outputs[2]:
            nodes_to_remove = []
            continue

        
        if input_node_0.op_type != 'MatMul':
            nodes_to_remove = []
            continue
        else:
            nodes_to_remove.append(input_node_0)
            original_matmul_inputs = input_node_0.input
            multihead_input = [original_matmul_inputs[0], original_matmul_inputs[1], multihead_input]
            for shape in shapes:
                shape_dict[shape] = multihead_input[0]

        custom_node = helper.make_node(
            'MultiHeadMatMul',
            inputs=multihead_input,
            outputs=[multihead_output],
            name='MultiHeadMatMul_' + reshape_node_0.name.split('_')[-1]
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
        tensor_producers[multihead_output] = custom_node
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

    # update the shape_dict
    for node in graph.node:
        if node.op_type == 'Shape' and node.name in shape_dict:
            node.input[0] = shape_dict[node.name]
    
    # (Optional) Infer shapes to ensure consistency
    #model = shape_inference.infer_shapes(model)
    return model

# Load the original model
model = onnx.load('model_gelu_rope.onnx', load_external_data=False)

# Apply the pattern replacement
model = replace_for_multihead(model)

# Save the modified model
onnx.save(model, 'model_gelu_rope_multi.onnx')
