After building the Rust package and moving the library up to this directory:

```
gcc -c jacobi.c
gcc -fopenmp -lm jacobi.o libjacobi.a -o jacobi -ldl

./jacobi
```
