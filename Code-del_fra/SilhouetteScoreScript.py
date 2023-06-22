ScoreDict = {'A':2, 'B':4, 'C':6, 'D':10, 'E':14, 'F':16, 'G':18}
S1 = (('A','B','C'), ('D','E','F','G'))
S2 = (('A','B'),('C','D'),('E','F','G'))
S3 = (('A','B','C','D'),('E','F','G'))



def _mean(S):
    """Returns a list of the means of each subset of S"""
    list_of_means = []
    for subset in S:
        counter = 0
        for i in subset:
            counter += ScoreDict[str(i)]
        list_of_means.append(counter / len(subset))
    return list_of_means

def _complete_silhouette_score(liste):
    total = 0
    number_of_elements = 0
    for subset in liste: 
        for i in subset:
            total += i
        number_of_elements += len(subset)
    return(total/number_of_elements)

def silhouette_score(S):
    list_of_means = _mean(S)
    individual_silhouettes_per_subset =[[] for subset in S]
    index = -1
    for subset in S:
        index += 1
        for i in subset:
            a = abs(ScoreDict[str(i)]-list_of_means[index])
            modified_list_of_means = [x for x in list_of_means if x != list_of_means[index]]
            distance_from_i_to_other_clusters = [abs(ScoreDict[str(i)]-point) for point in modified_list_of_means]
            b = min(distance_from_i_to_other_clusters)
            silhouette_score = (b-a)/max(b,a)
            individual_silhouettes_per_subset[index].append(silhouette_score)
    silhouette_score_for_cluster = _complete_silhouette_score(individual_silhouettes_per_subset)
    return(silhouette_score_for_cluster)

s1_sil = round(silhouette_score(S1),4)
s2_sil = round(silhouette_score(S2),4)
s3_sil = round(silhouette_score(S3),4)
print("A higher silhouette score the better, i.e. bigger equals better cohesion")
print("Silhouette score for S1: ")
print(s1_sil)
print("Silhouette score for S2: ")
print(s2_sil)
print("Silhouette score for S3: ")
print(s3_sil)


