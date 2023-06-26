from typing import List
import numpy as np

Vector = List[float]


def scalar_product_of_vectors(first_vector: Vector, second_vector: Vector) -> float:
    assert len(first_vector) == len(second_vector)

    return sum(first_vector_i * second_vector_i for first_vector_i, second_vector_i in zip(first_vector, second_vector))


def add_vectors(first_vector: Vector, second_vector: Vector) -> Vector:
    assert len(first_vector) == len(second_vector)

    return [first_vector_i + second_vector_i for first_vector_i, second_vector_i in zip(first_vector, second_vector)]


def multiply_vector_by_scalar(scalar: float, vector: Vector) -> Vector:
    return [scalar * vector_i for vector_i in vector]


def gradient_step(vector: Vector, gradient: Vector, step_size: float) -> Vector:
    assert len(vector) == len(gradient)
    step = multiply_vector_by_scalar(step_size, gradient)

    return add_vectors(vector, step)


def sigmoid(t: float) -> float:
    return 1 / (1 + np.exp(-t))

