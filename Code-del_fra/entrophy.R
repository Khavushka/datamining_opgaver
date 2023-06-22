# decision tree 

# information gain 

# calculate the entropy for a dataset
from math import log2
# proportion of examples in each class
risk_low = 4/8
risk_high = 4/8
# calculate entropy
entropy = -(risk_low * log2(risk_low) + risk_high * log2(risk_high))
# print the result
entropy
# = 1 

# calculate the entropy for the split in the dataset:

# area:
# proportion of examples in each class 
# urban 
urban_low = 3/3
urban_high = 0/3
# calculate entropy
urban_entropy = -(urban_low * log2(urban_low))
# print the result
urban_entropy = 1 
# = 0

# rural
rural_low = 1/5
rural_high = 4/5
# calculate entropy
rural_entropy = -(rural_low * log2(rural_low) + rural_high * log2(rural_high))
# print the result
rural_entropy
# = 0.721

# information gain: 
n = 8 
entropy - (urban_entropy/n)*urban_entropy - (rural_entropy/n)*rural_entropy
# = 0.8


# gender:
# proportion of examples in each class 
# female 
f_low = 2/3
f_high = 1/3
# calculate entropy
- (2/3)*log2(2/3)-1/3*log2(1/3)
# = 0.918

# men
m_low = 2/5
m_high = 3/5
# calculate entropy
- (2/5)*log2(2/5)-3/5*log2(3/5)
# = 0.971

# information gain: 
n = 8 
1 - (3/8)*0.918 - (5/8)*0.971
# = 0.048






################

# split via area
s1_class0 = 3/8
s1_class1 = 5/8
# calculate the entropy of the first group
s1_entropy = -(s1_class0 * log2(s1_class0) + s1_class1 * log2(s1_class1))
# print the result
s1_entropy
# = 0.95


# split via gender
s2_class0 = 5/8
s2_class1 = 3/8
# calculate the entropy of the second group
s2_entropy = -(s2_class0 * log2(s2_class0) + s2_class1 * log2(s2_class1))
# print the result
s2_entropy
# = 0.95


# split via time to license 
s3_class0 = 3/8
s3_class1 = 3/8
s3_class2 = 2/8
# calculate the entropy of the second group
s3_entropy = 1 - ((s3_class0 * log2(s3_class0)) + (s3_class1 * log2(s3_class1)) + (s3_class2 * log2(s3_class2))
# print the result
s3_entropy
# = 1.56

1 - ( 
  (s3_class0 * log2(s3_class0)) +
  (s3_class1 * log2(s3_class1)) +
  (s3_class2 * log2(s3_class2)) ) 

a = s3_class0 * log2(s3_class0)
b = s3_class1 * log2(s3_class1)
c = s3_class2 * log2(s3_class2)
1 - (a + b + c)
1 - (-1.56)


# Note that minimizing the entropy is equivalent to maximizing the information gain


# example of a decision tree trained with information gain
from sklearn.tree import DecisionTreeClassifier
model = sklearn.tree.DecisionTreeClassifier(criterion='entropy')


# entropy 
s3_class0 = 3/8
s3_class1 = 3/8
s3_class2 = 2/8
# calculate the entropy of the second group
s3_entropy = 1 - ((s3_class0 * log2(s3_class0)) + (s3_class1 * log2(s3_class1)) + (s3_class2 * log2(s3_class2))
                  # print the result
                  s3_entropy
                  # = 1.56

# calculate the information gain
gain = s_entropy - (8/20 * s1_entropy + 12/20 * s2_entropy)







# calculate the entropy for a dataset
from math import log2
# proportion of examples in each class
class0 = 10/100
class1 = 90/100
# calculate entropy
entropy = -(class0 * log2(class0) + class1 * log2(class1))
# print the result
print('entropy: %.3f bits' % entropy)