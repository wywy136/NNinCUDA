from dataset import MNISTDataset
from fnn import FNNForMNIST

import sys

def main(nl, nh, ne, nb):
    batch_size = nb
    epoch_num = ne
    training_set = MNISTDataset("train", batch_size)
    test_set = MNISTDataset("test", batch_size)
    test_num = test_set.num_figs
    num_batches_test = len(test_set)

    model = FNNForMNIST(nl, nh, 0.0005)
    losses = []

    for j in range(epoch_num):

        if j + 1 > epoch_num:
            return
        # [batch_num, 784], [batch_num,]
        train_img, train_lbl = training_set.get_batch_data()
        
        loss = model.step(train_img, train_lbl, len(train_img))

        if j % 500 == 0:
            print(f"Epoch {j+1}/{epoch_num} \tLoss {loss}")
            losses.append(loss)
    
    # Evaluation
    print("Evaluating on test dataset.")
    hit = 0
    for j in range(num_batches_test):
        test_img, test_lbl = test_set.get_batch_data()
        prd_lbl = model.predict(test_img, len(test_img))

        for k in range(len(prd_lbl)):
            # Predicted label is same as the true label
            if test_lbl[k][prd_lbl[k]] == 1:
                hit += 1
    
    print(f"Accuracy on test: {hit / test_num}")
    print(losses)

if __name__ == "__main__":
    [nl, nh, ne, nb] = sys.argv[1:]
    nl, nh, ne, nb = int(nl), int(nh), int(ne), int(nb)
    main(nl, nh, ne, nb)