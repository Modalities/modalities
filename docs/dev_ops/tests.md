# Testing Modalities

Modalities has a threefold setup for testing, namely

* Main tests <br> 
The main tests comprise CPU, single GPU and multi-GPU tests. The latter ones create a distributed environment internally and allow end2end testing of Modalities. 
Each of these tests defines its requirements (typically the number of GPUs) in the test and the test will be skipped if the requirements are not met.

* Torchrun tests <br>
These tests are run from a shell script using torchrun and are typically end2end or at least integration tests. Since we implemented distributed testing using multiprocessing within Modalities, these tests will be integrated into the main tests in the long term. Note that some of the torchrun tests have been already migrated to the main tests. 

* Example / Tutorial tests <br>
These tests take an example config (e.g., training config or a warmstart config) and execute it. The test makes sure that the config can be executed without errors. The test does not check the results of the execution, but only that the execution can be completed without errors. The user has to check manually for errors in the output.

## Testing Entry Points
There is a single entrypoint to run all test types specified above. 
For a full specification of the test API run

```bash
cd modalities
python tests/tests.py --help
```

in your command line. 

