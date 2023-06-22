# Given data
data = [
    ['share1', 'sell', 'sell', 'buy', 'yes'],
    ['share2', 'buy', 'buy', 'buy', 'yes'],
    ['share3', 'buy', 'sell', 'sell', 'yes'],
    ['share4', 'sell', 'sell', 'buy', 'yes'],
    ['share5', 'buy', 'sell', 'buy', 'yes'],
    ['share6', 'sell', 'sell', 'sell', 'yes'],
    ['share7', 'sell', 'buy', 'sell', 'no'],
    ['share8', 'sell', 'buy', 'buy', 'no'],
    ['share9', 'sell', 'buy', 'sell', 'no'],
    ['share10', 'buy', 'sell', 'sell', 'no'],
]

# Calculate the prior probability of each class (buy and sell)
total_shares = len(data)
buy_shares = sum(1 for d in data if d[4] == 'yes')
sell_shares = total_shares - buy_shares

p_buy = buy_shares / total_shares
p_sell = sell_shares / total_shares

# Calculate the conditional probabilities for each attribute
attributes = ['Hunter', 'Meyer', 'Smith']

p_buy_attributes = {}
p_sell_attributes = {}

for attribute in attributes:
    attribute_counts_buy = 0
    attribute_counts_sell = 0

    for d in data:
        if d[4] == 'yes':
            if d[attributes.index(attribute) + 1] == 'buy':
                attribute_counts_buy += 1
        else:
            if d[attributes.index(attribute) + 1] == 'buy':
                attribute_counts_sell += 1

    p_buy_attributes[attribute] = attribute_counts_buy / buy_shares
    p_sell_attributes[attribute] = attribute_counts_sell / sell_shares

# Case 1: share_A - Hunter: buy, Meyer: buy, Smith: sell
p_buy_share_A = p_buy * p_buy_attributes['Hunter'] * p_buy_attributes['Meyer'] * p_sell_attributes['Smith']

# Case 2: share_B - Hunter: buy, Meyer: sell, Smith: buy
p_buy_share_B = p_buy * p_buy_attributes['Hunter'] * p_sell_attributes['Meyer'] * p_buy_attributes['Smith']

p_buy_share_C = p_buy * p_sell_attributes['Hunter'] * p_sell_attributes['Meyer'] * p_buy_attributes['Smith']

p_buy_share_D = p_buy * p_sell_attributes['Hunter'] * p_buy_attributes['Meyer'] * p_buy_attributes['Smith']

print("Probability of buying for share_A:", p_buy_share_A)
print("Probability of buying for share_B:", p_buy_share_B)
print("Probability of buying for share_C:", p_buy_share_C)
print("Probability of buying for share_D:", p_buy_share_D)
