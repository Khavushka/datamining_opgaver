import pandas as pd


##################################################
#SE NEDERST I DOKUMENTET FOR AT INDSKRIVE NY DATA
##################################################

def naive_bayes(df, new_data):
    outcome_data = df.iloc[:, -1]
    classes = outcome_data.unique()

    # Find prior probabilities for each class (outcome)
    prior_probabilities = outcome_data.value_counts(normalize=True)

    # Initialize the likelihoods dictionary
    likelihoods = {}

    # For each feature:
    for column_name in df.iloc[:, :-1]:
        column_data = df[column_name]

        # Prior probability for the feature:
        prior_feature = column_data.value_counts(normalize=True)

        # Conditional probabilities for feature, given the class
        conditional_probabilities = pd.crosstab(column_data, outcome_data, normalize='index')

        # Add conditional probabilities to the likelihoods dictionary
        likelihoods[column_name] = conditional_probabilities

    # Make predictions for new data
    predictions = []
    for index, row in new_data.iterrows():
        posterior_probabilities = {}
        for outcome in classes:
            posterior_prob = prior_probabilities[outcome]
            for feature in new_data.columns[:-1]:
                value = row[feature]
                if value in likelihoods[feature].index:
                    posterior_prob *= likelihoods[feature].loc[value, outcome]
                else:
                    posterior_prob = 0.0
                    break
            posterior_probabilities[outcome] = posterior_prob

        # Normalize the posterior probabilities
        total_prob = sum(posterior_probabilities.values())
        normalized_probs = {outcome: prob / total_prob for outcome, prob in posterior_probabilities.items()}

        # Choose the class with the highest probability as the predicted class
        predicted_class = max(normalized_probs, key=normalized_probs.get)
        predictions.append(predicted_class)

    return predictions


########################################
#INDSKRIV DEN STORE TABEL MED DATA HER
#######################################
    # her er 1 = forbid/no, 2 = allow/yes
    
    # vigtigt at outcome står som "outcome", ellers virker koden ikke
    # resten af kolonnerne navngiver du bare som du har lyst
    # du tilføjer også bare nye kolonner
    
df = pd.DataFrame({'hunter':   [1,1,1,2,2,2,2,1,1,2],
                   'meyer': [1,1,1,1,1,2,2,2,2,1],
                   'smith':  [2,1,2,1,2,1,2,1,2,1],
                   'outcome': [2,2,2,2,2,2,1,1,1,1]
    })


##############################################
#INDSKRIV DE NYE DATA HER (DEM DU VIL TJEKKE)
################################################

# the new data you want to test:
new_data1 = pd.DataFrame({
    'hunter':   [2],
    'meyer': [2],
    'smith':  [1]
})

# Assuming you have new data for prediction
new_data2 = pd.DataFrame({
    'hunter': [2],
    'meyer': [1],
    'smith': [2]
})

# Assuming you have new data for prediction
new_data3 = pd.DataFrame({
    'hunter': [1],
    'meyer': [1],
    'smith': [2]
})

# Assuming you have new data for prediction
new_data4 = pd.DataFrame({
    'hunter': [1],
    'meyer': [2],
    'smith': [2]
})

#HER SKAL DU VÆRE SIKKER PÅ AT DU HAR ALLE DINE NYE DATA
predictions = [naive_bayes(df, new_data1),
               naive_bayes(df, new_data2),
               naive_bayes(df, new_data3),
               naive_bayes(df, new_data4)]

print(predictions)

