# rmpc

_To do: Write a proper readme_

## 32-bit floats

When compiling for embedded devices, it may be beneficial to use 32-bit floats instead of the default 64-bit floats inside the solver. This reduces the accuracy but can improve the computation time on devices without support for 64-bit floats in hardware. This can be done by disabling the default `f64` feature. Note that this also changes the default values of the tolerance settings.
