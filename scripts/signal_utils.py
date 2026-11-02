
def aggregate_by_counts(scores,counts):

    cursor = 0
    return_list=[]
    for i in range(len(counts)):
        maxs= max(scores[cursor:cursor + counts[i]])
        mean= sum(scores[cursor:cursor + counts[i]]) / counts[i]
        return_list.append((maxs,mean))
        cursor += counts[i]

    return return_list

def build_pair(question,passage,mode):

    if mode == "qnli":

        return (question,passage)

    if mode == "plain":

        return (passage,question)

    if mode == "template":

        return (passage, "This passage answers the question: "+question)