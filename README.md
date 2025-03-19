# TMK KEDV Models

A collection of machine learning models for analyzing earthquake-related social media content, developed for the Oxfam project.

## Setup

1. Install Anaconda from [https://www.anaconda.com/download](https://www.anaconda.com/download)
   > Note: Any recent version of Anaconda/Miniconda will work, as long as it can create environments with Python 3.10.4

2. Create and activate environment:
    ```bash
    # Create environment and install dependencies in one go
    conda create -n kedv python=3.10.4 pytorch torchvision torchaudio -c pytorch
    conda activate kedv
    pip install -r requirements.txt
    ```

## Model Pipeline

The system processes tweets through a sequence of models:

1. **Earthquake Detection**: Filters tweets related to earthquakes
2. **Aid Recognition**: Identifies aid-related content from earthquake tweets
3. **Aid Classification**: Categorizes the type of aid activity mentioned
4. **Organization Detection**: Analyzes user profiles to identify organizational accounts

## Usage

Each model can be run independently using the following format:
```bash
python <model_name>.py --input <input_file> --output <output_file>
```

Example:
```bash
python earthquake_model.py --input tweets.json --output earthquake_results.json
```

## Dependencies

Required packages and versions are listed in `requirements.txt`

## License

[Add License Information]
