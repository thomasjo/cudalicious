# Cudalicious

This is a header-only C++ template library intended to make any code using the
CUDA runtime API easier to read and write, and hopefully less error prone.

## Huh?
OK, let's assume we have a `std::vector<float> foo` with some arbitrary data.
If we want to use some CUDA kernel, then we might write some code like this

``` c++
const auto foo_size = sizeof(float) * foo.size();
float* d_foo;
cudaMalloc(&d_foo, foo_size);
cudaMemcpy(d_foo, foo.data(), foo_size, cudaMemcpyHostToDevice);
```

That's lot of boilerplate code simply to copy a bit of data to the GPU device.
The observant reader might also have noticed this code does not perform any
error checking on those two CUDA functions.

However, if we were to use Cudalicious, the code above would magically become

``` c++
auto* d_foo = cuda::copy_to_device(foo);
```

which is equivalent to the original code — but it also checks for errors.

## Usage
Simply include the `cudalicious.h` header file in your `.cu` files, and you are
good to go. See the accompanying examples to get a sense of what your code can
look like if you end up giving this library a chance.

## Disclaimer
This project is in an early alpha phase and is probably not ready for real
production use. Any and all help is very much appreciated.
