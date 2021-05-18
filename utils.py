def decode_from_labels(word, labels, sep="/"):
    answer = ""
    for i, (label, letter) in enumerate(zip(labels, word)):
        if i > 0 and label[0] in "BS":
            answer += "/"
        answer += letter
    return answer