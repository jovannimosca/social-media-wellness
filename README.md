# social-media-wellness
A social media and data mining project looking to identify and analyze wellness trends.

## Setup
1. Download the `bigboy.csv` dataset linked in the [sources list](./data/sources.md).
2. Install the requirements `pip install -r requirements.txt`.
3. Set up NLTK if you haven't already:
   ```sh
   bash> python3
   Python 3.12.0 ...
   >>> from nltk import download
   >>> download()
   ````
   Click the "Download" button and wait for everything to download.

4. Run `processData.py` to generate the processed data CSVs.
5. Run `getStatistics.py` to generate graphs, charts, and get stats.

## Layout
- `data/` - contains dataset (you should download the source data and put it here).
- `img/` - contains images (charts and graphs) rendered using `matplotlib` from the scripts.
- `processed/` - contains processed data.
   - You should use `relevant.csv` as your data source for subsequent processing. This includes all columns from the original dataset, but only those rows that contain relevant hashtags.
   - You may also use `cleaned.csv` as your data source for subsequent processing that needs all tweets. This just hase dates parsed in a format recognizable by `pandas`.