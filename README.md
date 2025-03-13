# Beyond Stationarity in Time Series: Discovering Causal Structures and Latent Regimes via Markov Blankets

RCBNB-MB is a new causal discovery algorithm for discrete time series. Unlike traditional methods that assume a single, 
fixed causal structure, RCBNB-MB identifies different regimes, where the causal relationships remain consistent within each regime. 
The algorithm works in two steps: it first detects the regimes and then estimates the causal graph within each one. 
A key advantage of RCBNB-MB is that it relies on the Markov blanket, which helps reduce errors and keeps the most relevant information 
for prediction. By alternating between finding causal structures and segmenting the time series, 
the algorithm improves accuracy and minimizes errors.

## Required python packages

Python version: python=3.10

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Experiments


### Simulated data
* Please first set the correct system path in each script.
```bash
     import sys
     sys.path.append('/home/user/code_non_stationary/')
```

1. To test algorithms on simulated data with 2 regimes run: 
```bash
    cd \Experiments\Experiments_2_regime
    python test_[method].py

```
- [method]: can choose from ['rcbnb_mb', 'rcbnb_pa','rdynotears_mb','rdynotears_pa','rpcmci_mb','rpcmci_pa','rpcmciplus_mb', 'rpcmciplus_pa', 'test_rvarlingam_mb', 'test_rvarlingam_pa'];
- Results are saved in the folder 'results/2regimes'.

2. To test algorithms on simulated data with 3 regimes run:
 *  **Attention:** Before executing the code, please extract the contents of `3_regimes.zip` and place them in the `Dataset/n/` folder.
```bash
    cd \Experiments\Experiments_3_regime
    python test_[method]_3regimes.py
```
- [method]: can choose from ['rcbnb_mb', 'rcbnb_pa','rdynotears_mb','rdynotears_pa','rpcmci_mb','rpcmci_pa','rpcmciplus_mb', 'rpcmciplus_pa', 'test_rvarlingam_mb', 'test_rvarlingam_pa'];
- Results are saved in the folder 'results/3regimes'.

### Real data
1. To test algorithms on _real IT monitoring data_ run:
```bash
    cd \Experiments\Experiments_IT
    python test_rcbnb_mb_IT_monitoring.py
```
- Results are saved in the folder 'results/IT_monitoring'.

