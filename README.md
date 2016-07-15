# Cudalicious

C++ header library intended to reduce the amount of CUDA boilerplate code — and hopefully less error-prone.

## Huh?
OK, let's assume we have a `std::vector<float> foo` with some arbitrary data. If we want to use this data in a CUDA kernel,
we could write some code like this in order to copy the dat from the host to the device;

``` c++
const auto foo_size = sizeof(float) * foo.size();
float* d_foo;
cudaMalloc(&d_foo, foo_size);
cudaMemcpy(d_foo, foo.data(), foo_size, cudaMemcpyHostToDevice);
```

That's lot of boilerplate code simply to copy a bit of data to the GPU. The observant reader might also have noticed this
code does not perform any error checking on those two CUDA functions.

However, if we were to use Cudalicious, the code above could magically be written as

``` c++
auto* d_foo = cuda::copy_to_device(foo);
```

This is equivalent to the original code, with the added benefit that it also checks for errors.

## Usage
Simply include the `cudalicious.h` header file in your `.cu` files, and you are good to go.

See the accompanying examples to get a sense of what your code can look like if you end up giving this library a chance.

## Disclaimer
This project is in an early alpha phase and is probably not ready for real production use.
Any and all help is very much appreciated.
