from dataprocess import trainsetprocess, testsetprocess, generate_text
from model import main
import numpy as np
if __name__ == '__main__':
    """
    Test mode 
    """
    test_text = np.load('../data/test_text.npy')
    test_Y = "../data/test_Y.npy"
    output = "../data/test_otuput.txt"

    # trainsetprocess("../data/train.txt")
    testsetprocess('../data/test.txt')
    main(outputfile=test_Y)
    test_Y = np.load(test_Y)
    generate_text(test_text, test_Y, output)


