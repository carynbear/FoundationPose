# Instructions to run with IPD

Setup your code to be like this:

```
FoundationPose/
    ...
    run_ipd.py
    setup_fp4ipd.sh
    debug/
        ipd/
            ...
            [saved results].yml
    ...
ipd/
    ...
    datasets/
        [dataset]/
            test/
                ...
            dataset_info.json
    results_foundation_pose.ipynb
    ...
```

1. Follow instructions to run docker container interactively
2. Inside container, run `./FoundationPose/setup_fp4ipd.sh` to install x11 and `ipd` package
3. Test x11 with `xeyes`
4. `cd FoundationPose`
5. `python run_ipd.py`

To see results, launch the `ipd/results_foundation_pose.ipynb` notebook. Make sure the kernel you use has `ipd` installed