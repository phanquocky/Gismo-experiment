# Run experiment GISMO

## tools
### encode network tool
```
./web-gcnf/identifying-codes/scripts/encoding/encode_network.py -n "network_file" --out_dir "output_directory" --out_file "output_filename" --encoding gis --two_step k # k is the maximum identifiable set size
```

### gismo tool
to run gismo tool we need to exec in docker container and run the following command:
```
$ docker exec -it acca5e59ce88 /bin/bash
$ ./gismo/build/gismo --maxc 5000 ./gismo/example/example.gcnf (here is gismo tool)
```
### parse gismo output
we have function `parse_sensor_set_from_gismo_output(gismo_text: str, gcnf_path: str) -> List[int]:` in `parse_gismo_output.py` to parse gismo output and return the sensor set.

## TODO list

### Task1: Run experiment gismo tool of all datasets
- [ ] Traversal all the datasets in `datasets` folder.
- [ ] For each dataset, run encode network tool to convert the network file to gcnf format (with each k = 1,2,3,4,6,8,10,12,16). I mean for each dataset, we will have 9 gcnf files with different k values.
- [ ] Run gismo tool with --maxc 5000 with respective gcnf file to generate gismo output.
- [ ] Use `parse_sensor_set_from_gismo_output` function to parse gismo output and get the sensor set.
- [ ] Save the result in a file named `gismo_result.txt` in the same folder as the dataset file, with the following information:
    - Run time: time take run "only" gismo tool (exclude time take run encode network tool)
    - Is Solvable: whether gismo tool can find a solution within the maxc limit (True, False)
    - Sensor Set Size: the size of the sensor set obtained from gismo output
- [ ] write a report to summarize the results of gismo tool on all datasets, including the run time, whether it is solvable, and the sensor set size for each dataset and each k value. The report can be in a tabular format for easy comparison.