# Cudalicious

C++ header library intended to reduce the amount of CUDA boilerplate code — and
hopefully less error-prone.

## Huh?
OK, let's assume we have a container with some arbitrary data,

``` c++
const std::vector<float> foo // ...
```

If we want to use this data in a CUDA kernel, we can do something like this in
order to copy the data from the host to the device

``` c++
const auto foo_size = sizeof(float) * foo.size();
float* d_foo;
cudaMalloc(&d_foo, foo_size);
cudaMemcpy(d_foo, foo.data(), foo_size, cudaMemcpyHostToDevice);
```

That's lot of boilerplate code simply to copy some data to the GPU. The
observant reader might also have noticed this code does not perform any error
checking on those two CUDA functions...

However, if we use Cudalicious, the code above can "magically" be written as

``` c++
auto d_foo = cuda::copy_to_device(foo);
```

This is equivalent to the original code, with the added benefit that it also
checks for errors :zap:

## Usage
Simply grab the header files from the `include` directory, include the ones you
need, and you should be good to go.

See the accompanying examples to get a sense of what your code can look like if
you end up giving this library a chance.

## Disclaimer
This project is in an early alpha phase and is probably not ready for real
production use.

Any and all help is very much appreciated.
