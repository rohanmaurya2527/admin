import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
# Medical Transactions
transactions = [
    ['Fever','Cough','Headache'],
    ['Fever','Cough'],
    ['Cough','Shortness of Breath'],
    ['Fever','Headache'],
    ['Cough','Chest Pain'],
    ['Fever','Cough','Chest Pain'],
    ['Headache','Nausea'],
    ['Fever','Cough','Shortness of Breath'],
    ['Cough','Headache'],
    ['Fever','Nausea'],
    ['Chest Pain','Shortness of Breath'],
    ['Fever','Cough','Headache'],
    ['Cough','Nausea'],
    ['Fever','Chest Pain'],
    ['Cough','Shortness of Breath','Chest Pain']
]
def run_market_basket(transactions):
    te = TransactionEncoder()
    df = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)    
    for name, algo in [("Apriori", apriori), ("FP-Growth", fpgrowth)]:
        print(f"\n=== {name} ===")
        # Frequent Itemsets
        freq_items = algo(df, min_support=0.3, use_colnames=True)
        print("\nFrequent Itemsets:")
        print(freq_items)       
        # Association Rules
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)
        print("\nAssociation Rules:")
        print(rules[['antecedents','consequents','support','confidence','lift']])
# Run
run_market_basket(transactions)
