import numpy as np


conversions = {
    'uint8': np.uint8,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex_float': np.complex64,
    'complex_double': np.complex128
}


def decide_type(type1, type2):

    if type1 == type2:
        return conversions[type1]

    #uint8 conversion rules
    if (type1 == 'uint8' and type2 == 'int') or (type1 == 'int' and type2 == 'uint8'):
        return conversions['int']
    if (type1 == 'uint8' and type2 == 'float') or (type1 == 'float' and type2 == 'uint8'):
        return conversions['float']
    if (type1 == 'uint8' and type2 == 'double') or (type1 == 'double' and type2 == 'uint8'):
        return conversions['double']
    if (type1 == 'uint8' and type2 == 'complex_float') or (type1 == 'complex_float' and type2 == 'uint8'):
        return conversions['complex_float']
    if (type1 == 'uint8' and type2 == 'complex_double') or (type1 == 'complex_double' and type2 == 'uint8'):
        return conversions['complex_double']

    #int conversion rules
    if (type1 == 'int' and type2 == 'float') or (type1 == 'float' and type2 == 'int'):
        return conversions['float']
    if (type1 == 'int' and type2 == 'double') or (type1 == 'double' and type2 == 'int'):
        return conversions['double']
    if (type1 == 'int' and type2 == 'complex_float') or (type1 == 'complex_float' and type2 == 'int'):
        return conversions['complex_float']
    if (type1 == 'int' and type2 == 'complex_double') or (type1 == 'complex_double' and type2 == 'int'):
        return conversions['complex_double']

    # float conversion rules
    if (type1 == 'float' and type2 == 'double') or (type1 == 'double' and type2 == 'float'):
        return conversions['double']
    if (type1 == 'float' and type2 == 'complex_float') or (type1 == 'complex_float' and type2 == 'float'):
        return conversions['complex_float']
    if (type1 == 'float' and type2 == 'complex_double') or (type1 == 'complex_double' and type2 == 'float'):
        return conversions['complex_double']

    # double conversion rules
    if (type1 == 'double' and type2 == 'complex_float') or (type1 == 'complex_float' and type2 == 'double'):
        return conversions['complex_float']
    if (type1 == 'double' and type2 == 'complex_double') or (type1 == 'complex_double' and type2 == 'double'):
        return conversions['complex_double']

    # complex_float conversion rules
    if (type1 == 'complex_float' and type2 == 'complex_double') or \
            (type1 == 'complex_double' and type2 == 'complex_float'):
        return conversions['complex_double']
