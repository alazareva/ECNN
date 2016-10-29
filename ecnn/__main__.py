import sys
sys.path.append(r"/Users/anastasiyalazareva/Documents/ECNN/")
import ecnn.datasets.dataset as ds
import ecnn.tournament as tmnt

import ecnn.functions as functions

def main():
    dataset = ds.get_cifar_10_dataset()
    dataset.limit_training_set_size(10000)
    tournament = tmnt.Tournament()
    fitness_function = functions.sort_on_accuracy
    tournament.run(dataset, fitness_function)


if __name__ == "__main__":
    main()


