def calculate_precision_recall_f1score(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1_score

# Example usage
tp = 3
fp = 1
fn = 2

precision, recall, f1_score = calculate_precision_recall_f1score(tp, fp, fn)

print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
