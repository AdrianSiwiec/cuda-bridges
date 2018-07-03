cuda-bridges
============

Source code of Bachelor's Thesis related work.

Example:
```shell
# Prepare
wget http://nrvis.com/download/data/dimacs10/kron_g500-logn20.zip
unzip kron_g500-logn20.zip *.mtx

# Execute (please make sure that ulimit -s is large enough)
make
./networkrepository-parser.e kron_g500-logn20.mtx kron_g500-logn20.mtx.bin
./runner.e kron_g500-logn20.mtx.bin
```

If you would like to export statistics printed on `stdout` you can use `export_results.py` parser:
```shell
./runner.e kron_g500-logn20.mtx.bin > stats.out
python3 export_results.py stats.out stats.csv
```

If you prefer to download & run all tests included in work you can use some helper scripts (please note that whole directory can even take up to 5GB of disk space):
```shell
cd test/
python3 test-downloader.py
python3 test-parser.py
cd ..
./simple_test_loop.sh
```
