# Harry Potter: Text Mining Analysis

This repository contains a text mining analysis project conducted as the final work for the PhD course "Programming Methodologies for Data Analysis." The topic was not fixed; we chose to analyze the Harry Potter series with a focus on sentiment, book ordering, and clustering.

## Project Structure

- `Data/`: Contains raw texts and datasets.
- `Functions/`: Utility scripts for preprocessing and analysis.
- `Images/`: Visualizations (e.g., word clouds).
- `HP.ipynb`: Main Jupyter notebook with all analysis steps.
- `Slides.pdf`: Summary of results.

## Methodology

- **Preprocessing:** Raw text from the seven books is tokenized, cleaned, and stemmed, removing stopwords and punctuation.
- **Sentiment Analysis:** Uses VADER and NRC sentiment lexicons to quantify emotional tone across books.
- **Book Ordering:** The first and last books are fixed; the rest are ordered based on textual similarity to these anchors, using frequency-based scoring and log-likelihood ratios. T-tests assess significance between scores.
- **Clustering:** K-means and spectral clustering (using cosine similarity and graph Laplacian approaches) are used to find structure among the books.

## Results

- Sentiment analysis shows the books become progressively darker.
- The ordering process positions the series nearly perfectly in canonical order, except for books 4 (Goblet of Fire) and 5 (Order of the Phoenix), which are statistically indistinguishable.
- Both clustering methods consistently split the books into: 
  - first three (pre-Voldemort's return)
  - last four (post-return)
  This result is meaningful and consistent between methods.

## Disclaimer

This project was completed for academic purposes and is not published or peer-reviewed.
