import nltk
nltk.download('brown')
import nltk
import sys
from nltk.corpus import brown
brown_tags_words = [ ]
for sent in brown.tagged_sents():
    brown_tags_words.append( ("START", "START") )
    brown_tags_words.extend([ (tag[:2], word) for (word, tag) in sent ])
    brown_tags_words.append( ("END", "END") )

cfd_tagwords = nltk.ConditionalFreqDist(brown_tags_words)
cpd_tagwords = nltk.ConditionalProbDist(cfd_tagwords, nltk.MLEProbDist)
print("The probability of an adjective (JJ) being 'new' is", cpd_tagwords["JJ"].prob("new"))
print("The probability of a verb (VB) being 'duck' is", cpd_tagwords["VB"].prob("duck"))
brown_tags = [tag for (tag, word) in brown_tags_words ]
cfd_tags= nltk.ConditionalFreqDist(nltk.bigrams(brown_tags))
cpd_tags = nltk.ConditionalProbDist(cfd_tags, nltk.MLEProbDist)
print("If we have just seen 'DT', the probability of 'NN' is", cpd_tags["DT"].prob("NN"))
print( "If we have just seen 'VB', the probability of 'JJ' is", cpd_tags["VB"].prob("DT"))
print( "If we have just seen 'VB', the probability of 'NN' is", cpd_tags["VB"].prob("NN"))
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("want") * \
    cpd_tags["VB"].prob("TO") * cpd_tagwords["TO"].prob("to") * \
    cpd_tags["TO"].prob("VB") * cpd_tagwords["VB"].prob("race") * \
    cpd_tags["VB"].prob("END")
print( "The probability of the tag sequence 'START PP VB TO VB END' for 'I want to race' is:",
prob_tagsequence)
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("saw") * \
    cpd_tags["VB"].prob("PP") * cpd_tagwords["PP"].prob("her") * \
    cpd_tags["PP"].prob("NN") * cpd_tagwords["NN"].prob("duck") * \
    cpd_tags["NN"].prob("END")
print( "The probability of the tag sequence 'START PP VB PP NN END' for 'I saw her duck' is:",
prob_tagsequence)
prob_tagsequence = cpd_tags["START"].prob("PP") * cpd_tagwords["PP"].prob("I") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("saw") * \
    cpd_tags["VB"].prob("PP") * cpd_tagwords["PP"].prob("her") * \
    cpd_tags["PP"].prob("VB") * cpd_tagwords["VB"].prob("duck") * \
    cpd_tags["VB"].prob("END")
print( "The probability of the tag sequence 'START PP VB PP VB END' for 'I saw her duck' is:",
prob_tagsequence)