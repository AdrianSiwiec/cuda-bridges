To see the example, I recommend to download this branch only:

```
git clone --single-branch --branch gunrock_example https://github.com/AdrianSiwiec/cuda-bridges.git --recurse-submodules
```

How to run:

1. Build Gunrock
  ```
  cd 3rdparty/gunrock
  mkdir build
  cd build
  cmake ..
  make
  cd ../..
  ```

2. make prepare-tests (downloads tests)

3. make run-tests
