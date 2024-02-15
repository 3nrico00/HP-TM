import numpy as np
from func.proc import CustomTokenizer, counter
from scipy import stats

#Counter = counter()

# this function create the counts and the freq for every book
def hp_count(idx, books_stem, most_comm_tot, df_book):
    #create the dict with the frequencies for the i-th book
    xx = {key: value for key, value in counter(books_stem[idx]).counts.items() if key in most_comm_tot}
    count = df_book['Stems'].map(xx)
    freq  = df_book['Stems'].map(xx)/df_book['Stems'].map(xx).sum()
    # df = pd.DataFrame
    return count, freq

def hp_scores(df, idx):
    N = df[f"Count_{idx}"].sum()
    n = df[f"Count_{idx}"]
    theta1 = df['Freq_1']
    theta7 = df['Freq_7']
    return round(1/N * sum(n*np.log(theta1/theta7)), 4)

def hp_var(df, idx):
    N = df[f"Count_{idx}"].sum()
    n = df[f"Count_{idx}"]
    theta1 = df['Freq_1']
    theta7 = df['Freq_7']
    return round(1/(N * (N - 1))*(sum(n * np.log(theta1/theta7)**2) - 1/N * (sum(n * np.log(theta1/theta7))**2)), 6)

def t_test(dict_1, dict_2):
    numerator = dict_1["score"] - dict_2["score"]
    denominator = np.sqrt(dict_1["Variance"] + dict_2["Variance"])

    gdl = dict_1["N"]+dict_2["N"]-2
    t_statistic = numerator / denominator
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=gdl))
    return t_statistic, p_value
