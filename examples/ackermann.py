def ackermann(m, n):
    if m == 0:
        return n + 1
    elif n == 0:
        return ackermann(m - 1, 1)
    else:
        return ackermann(m - 1, ackermann(m, n - 1))

for m in range(4):
    answers = []
    for n in range(5):
        answers.append(str(ackermann(m, n)))
    print(' '.join(answers))

