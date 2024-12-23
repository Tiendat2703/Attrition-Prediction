import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules, apriori
from sklearn.preprocessing import LabelEncoder

def apriori_fpgrowth(input_file):
    df = pd.read_csv(input_file)
    le = LabelEncoder()
    df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

    df = df[df['Attrition'] == 1]

    df_dummies = pd.get_dummies(df)
    df_dummies = df_dummies.apply(lambda col: col > 0, axis=0).astype(bool)

    frequent_itemsets_fp = fpgrowth(df_dummies, min_support=0.2, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.8, num_itemsets=4)
    rules_with_attrition_fp = rules_fp[rules_fp['consequents'].apply(lambda x: 'Attrition' in str(x))]

    frequent_itemsets_ap = apriori(df_dummies, min_support=0.2, use_colnames=True)
    rules_ap = association_rules(frequent_itemsets_ap, metric="confidence", min_threshold=0.8)
    rules_with_attrition_ap = rules_ap[rules_ap['consequents'].apply(lambda x: 'Attrition' in str(x))]

    return rules_with_attrition_fp, rules_with_attrition_ap
