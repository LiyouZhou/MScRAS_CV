# Computer Vision Coursework Code

## How to run

Install dependencies

```sh
pip3 install -r requirements. txt
```

Run the scripts

```sh
python3 ./task1.py
```

Will produce a single plot with all masked images.

```sh
python3 ./task2.py
```

Will produced plots showing distribution of the shape and texture features.

```sh
python3 ./task3.py
```

Will print the RMS error of the filter.

```sh
python3 ./task3.py --optimize
```

Will optimize the measurement and motion noise starting from default values. `--help` will give you further usage.

```sh
python3 ./task3.py --help
NAME
    task3.py

SYNOPSIS
    task3.py <flags>

FLAGS
    --measurement_noise=MEASUREMENT_NOISE
        Default: [0.25, 0.25]
    --motion_noise=MOTION_NOISE
        Default: [0.16, 0.36, 0.16, 0.36]
    -o, --optimize=OPTIMIZE
        Default: False
```

There are more code relating to producing the plots and analysis used in the report. See [assessment.ipynb](assessment.ipynb) for detail.
