# Miscellaneous classes

Although many of the classes in this library follow a class hierarchy, there are a few classes which have no relation to each other 
but can be quite useful for the users and overall functionality.

## Utility

The Utility class provides assorted static functions that help support other classes. These functions are quite small but are called often 
in the core classes. These functions help with checking parameters, array/matrix manipulations, statistics calculations, randomization, and more.
This class is unit tested.

## LinearAlgebra

The LinearAlgebra class provides static functions that implement Linear Algebra mathematical operations. These include things like matrix multiplication, 
 addition, initialization and other helper functions.
This class is unit tested.

## CSVWriter

The CSVWriter class is used extensively in the [examples](Examples.md) as a way of writing data to disk so that plots can be made. The class
allows users to create a csv file buffer with specified file path and column titles and then add rows to the buffer one row at a time. This 
class has no dependencies to any other classes, so it can be used in separate projects as well.