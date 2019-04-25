with open('training_set_VU_DM.csv', 'rb') as f:
    counter = 0
    for line in f:
        counter += 1
        print(line)
        if counter > 100:
            break