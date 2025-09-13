from vinagent.register import primary_function


@primary_function
def weight_of_bulldog():
    """The weight of a bulldog"""
    return 25


@primary_function
def weight_of_husky():
    """The weight of a husky"""
    return 20


@primary_function
def average_weight_of_two_dogs(weight1: float, weight2: float):
    """The average weight of two dogs"""
    return (weight1 + weight2) / 2
