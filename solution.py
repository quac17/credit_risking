import itertools
from collections import defaultdict
import pprint

# --- Helper Functions ---
def findsubsets(s, n):
    subsets = list(sorted((itertools.combinations(s, n))))
    return subsets

def items_from_frequent_itemsets(frequent_itemset):
    items = list()
    for keys in frequent_itemset.keys():
        if isinstance(keys, tuple):
            for item in keys:
                items.append(item)
        else:
            items.append(keys)
    return sorted(list(set(items)))

# --- Problem 1: Apriori ---
def generate_frequent_itemsets(dataset, support, items, n=1, frequent_items={}):
    len_transactions = len(dataset)
    frequent_itemsets = {}
    
    if n == 1:
        item_counts = defaultdict(int)
        for transaction in dataset.values():
            for item in set(transaction):
                if item in items:
                    item_counts[item] += 1
        for item, count in item_counts.items():
            if count / len_transactions >= support:
                frequent_itemsets[item] = count
    else:
        if not frequent_items:
            return {}
        remaining_items = items_from_frequent_itemsets(frequent_items)
        all_subsets = findsubsets(remaining_items, n)
        for subset in all_subsets:
            n_subset = 0
            subset_set = set(subset)
            for transaction in dataset.values():
                if subset_set.issubset(set(transaction)):
                    n_subset += 1
            if n_subset / len_transactions >= support:
                frequent_itemsets[subset] = n_subset
    return frequent_itemsets

# --- Problem 2: FP-Growth ---
def item_support(dataset, min_support):
    len_transactions = len(dataset)
    support_dict = defaultdict(int)
    for transaction in dataset.values():
        for item in transaction:
            support_dict[item] += 1
    sorted_support = dict(sorted(support_dict.items(), key=lambda item: item[1], reverse=True))
    pruned_support = {key: val for key, val in sorted_support.items() if val / len_transactions >= min_support}
    return pruned_support

def reorder_transactions(dataset, min_support):
    pruned_support = item_support(dataset, min_support) 
    updated_dataset = dict()
    for key, value in dataset.items():
        # Filter and sort based on support count
        filtered_items = [item for item in value if item in pruned_support]
        # To match the expected homework output exactly, we sort by support count descending.
        # Tie-breaking in the homework seems to follow the order in support_dict_expected: C=7, D=9, E=5, B=6, A=6? 
        # Actually expected order is D(9), C(7), A(6), B(6), E(5).
        # Let's use the support count directly.
        updated_dataset[key] = sorted(filtered_items, key=lambda x: pruned_support[x], reverse=True)
    return updated_dataset

def build_fp_tree(updated_dataset):
    fp_tree = {}
    for transaction in updated_dataset.values():
        current_node = fp_tree
        for item in transaction:
            if item in current_node:
                current_node[item]['count'] += 1
            else:
                current_node[item] = {'count': 1, 'children': {}}
            current_node = current_node[item]['children']
    return fp_tree

# --- Manual Dataset for Testing (matching original homework) ---
manual_dataset = {
    'T1': ['D', 'C', 'E'],
    'T2': ['D', 'C', 'B'],
    'T3': ['D', 'C', 'A'],
    'T4': ['D', 'C', 'A', 'E'],
    'T5': ['D', 'C', 'A', 'B'],
    'T6': ['B'],
    'T7': ['D', 'E'],
    'T8': ['D', 'C', 'A', 'B'],
    'T9': ['D', 'A', 'B', 'E'],
    'T10': ['D', 'C', 'A', 'B', 'E']
}

# --- Execution & Verification ---
print("--- Problem 1: Apriori Test ---")
items = ['A', 'B', 'C', 'D', 'E']
freq1 = generate_frequent_itemsets(manual_dataset, 0.5, items, n=1)
print("Freq 1:", freq1)
freq2 = generate_frequent_itemsets(manual_dataset, 0.5, items, n=2, frequent_items=freq1)
print("Freq 2:", freq2)

print("\n--- Problem 2: FP-Growth Test ---")
sup = item_support(manual_dataset, 0.5)
print("Support Dict:", dict(sup))

reordered = reorder_transactions(manual_dataset, 0.5)
print("Reordered T10:", reordered['T10'])

tree = build_fp_tree(reordered)
pp = pprint.PrettyPrinter(depth=8)
print("FP Tree Structure:")
pp.pprint(tree)

# Logic check for the expected tree in homework
expected_tree_root_keys = set(['D', 'B'])
if set(tree.keys()) == expected_tree_root_keys:
    print("\nRoot keys match expected!")
else:
    print(f"\nRoot keys MISMATCH! Found: {list(tree.keys())}")
