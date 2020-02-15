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

2. Download tests
```
make prepare-tests
```

3. Add gunrock library to your path, for example:
```
LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:<path_to_your_repository>/3rdparty/gunrock/build/lib/
export LD_LIBRARY_PATH
```

4. Run the example
```
make run-tests
```
