#pragma once

#define MAKE_STR(x) #x
#define CONCAT(x, y) x ## y
#define EVALUATE(fName, ...) fName(__VA_ARGS__)



