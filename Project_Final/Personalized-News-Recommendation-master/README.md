## Bandits implemented
- UCB
- Thompson Sampling
- E-greedy
- LinUCB with disjoint linear models

## References
- ### A Contextual-Bandit Approach to Personalized News Article Recommendation https://arxiv.org/pdf/1003.0146.pdf
- ### Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms  https://arxiv.org/pdf/1003.5956.pdf
    Used algorithm 2 as a policy evaluator (for finite data stream)


## Dataset
### R6A - Yahoo! Front Page Today Module User Click Log Dataset, version 1.0 (1.1 GB)
The dataset contains 45,811,883 user visits to the Today Module. For each visit, both the user and each of the candidate articles are associated with a feature vector of dimension 6 (including a constant feature), constructed using a conjoint analysis with a bilinear model.
The dataset can be found [here](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r).



Online content recommendation represents an important example of interactive machine learning problems that require an efficient tradeoff between exploration and exploitation. Such problems, often formulated as various types of multi-armed bandits, have received extensive research in the machine learning and statistics literature. Due to the inherent interactive nature, creating a benchmark dataset for reliable algorithm evaluation is not as straightforward as in other fields of machine learning or recommendation, whose objects are often prediction. Our dataset contains a fraction of user click log for news articles displayed in the Featured Tab of the Today Module on Yahoo! Front Page during the first ten days in May 2009. The articles were chosen uniformly at random, which allows one to use a recently developed method of Li et al. [WSDM 2011] to obtain an unbiased evaluation of a bandit algorithm. To the best of our knowledge, this is the first real-world benchmark evaluating bandit algorithms reliably. The dataset contains 45,811,883 user visits to the Today Module. For each visit, both the user and each of the candidate articles are associated with a feature vector of dimension 6 (including a constant feature), constructed using a conjoint analysis with a bilinear model; see Chu et al. [KDD 2009] for more details. The size of this dataset is 1.1GB

Here are all the papers published on this Webscope Dataset:

- [Explore-Exploit in Top-N Recommender Systems via Gaussian Processes](https://yahoo-webscope-publications.s3.amazonaws.com/vanchinathan14explore.pdf)
- [Discovering Valuable Items from Massive Data](https://yahoo-webscope-publications.s3.amazonaws.com/vanchinathan15Discovering.pdf)

