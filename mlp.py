from typing import List

from math_functions import Vector, sigmoid, scalar_product_of_vectors, gradient_step


def neuron_output(weights: Vector, inputs: Vector) -> float:
    return sigmoid(scalar_product_of_vectors(weights, inputs))


def feed_forward(neural_network: List[List[Vector]], input_vector: Vector) -> List[Vector]:
    outputs: List[Vector] = []

    for layer in neural_network:
        input_with_bias = input_vector + [1]
        output = [neuron_output(neuron, input_with_bias) for neuron in layer]
        outputs.append(output)
        input_vector = output

    return outputs


def sqerror_gradients(network: List[List[Vector]], input_vector: Vector, target_vector: Vector) -> List[List[Vector]]:
    hidden_outputs, outputs = feed_forward(network, input_vector)
    output_deltas = [output * (1 - output) * (output - target) for output, target in zip(outputs, target_vector)]
    output_grads = [[output_deltas[i] * hidden_output for hidden_output in hidden_outputs + [1]]
                    for i, output_neuron in enumerate(network[-1])]
    hidden_deltas = [[hidden_output * (1 - hidden_output) *
                      scalar_product_of_vectors(output_deltas, [n[i] for n in network[-1]])
                      for i, hidden_output in enumerate(hidden_outputs)]]
    hidden_grads = [[hidden_deltas[i] * input for input in (input_vector + [1])]
                    for i, hidden_neuron in enumerate(network[0])]

    return [hidden_grads, output_grads]


def main():
    import random
    random.seed(0)

    # training data
    xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
    ys = [[0.], [1.], [1.], [0.]]

    # start with random weights
    network = [  # hidden layer: 2 inputs -> 2 outputs
        [[random.random() for _ in range(2 + 1)],  # 1st hidden neuron
         [random.random() for _ in range(2 + 1)]],  # 2nd hidden neuron
        # output layer: 2 inputs -> 1 output
        [[random.random() for _ in range(2 + 1)]]  # 1st output neuron
    ]

    import tqdm

    learning_rate = 1.0

    for epoch in tqdm.trange(20000, desc="neural net for xor"):
        for x, y in zip(xs, ys):
            gradients = sqerror_gradients(network, x, y)

            # Take a gradient step for each neuron in each layer
            network = [[gradient_step(neuron, grad, -learning_rate)
                        for neuron, grad in zip(layer, layer_grad)]
                       for layer, layer_grad in zip(network, gradients)]

    # check that it learned XOR
    assert feed_forward(network, [0, 0])[-1][0] < 0.01
    assert feed_forward(network, [0, 1])[-1][0] > 0.99
    assert feed_forward(network, [1, 0])[-1][0] > 0.99
    assert feed_forward(network, [1, 1])[-1][0] < 0.01


if __name__ == "__main__": main()
