# TMK KEDV Models

A collection of machine learning models for analyzing earthquake-related social media content, developed for the Oxfam project.

## Setup

1. Install Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download)
   > Note: Any recent version of Anaconda/Miniconda will work, as long as it can create environments with Python 3.10.4

2. Create and activate environment:
    ```bash
    # Create environment and install dependencies in one go
    conda create -n kedv python=3.10.4
    conda activate kedv
    pip install -r transformers torch numpy huggingface_hub
    ```
    **Tested library versions for the development environment:**
    - transformers==4.49.0
    - torch==2.1.0+cu121
    - numpy==1.26.3
    - huggingface_hub==0.26.2

    **Note:**
    - We have tested with cuda version of 12.2.
    - Although we tested with these versions, latest libraries should also work as well.


## Model Pipeline

The model pipeline works in the following order:
1. **Earthquake Detection**: Detect earthquake-relevant tweets.
2. **Aid Recognition**: Identify if a tweet is about any aid activity.
3. **Aid Subcategory Classification**: Determine the specific kind of aid activity.

## Usage

Each model can be run independently using the following format:
```bash
python <model_name>.py
```

You need to modify the scripts to run it for your own purposes.