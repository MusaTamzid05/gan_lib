from simple_gan.const import LAYER_WIDTH

def is_valid_shape(width):

    # Poor mans valid tester :<
    # TODO : Improve this stupid shit !!
    current_value = LAYER_WIDTH
    valids= []

    for i in range(10):
        valids.append(int(current_value))
        current_value = current_value * 2

    if width in valids:
        return True
    return False





