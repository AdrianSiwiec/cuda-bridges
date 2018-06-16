#!/bin/bash

./simple_test_loop.sh > stats.out
./simple_test_and_export_stats.sh stats.out stats.csv

