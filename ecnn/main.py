import sys
sys.path.append(r"/Users/anastasiyalazareva/Documents/ECNN/")
import ecnn.datasets.dataset as ds
import ecnn.tournament as tmnt

def main():
    dataset = ds.get_cifar_10_dataset() #cifar10_dir = 'ecnn/datasets/cifar10/cifar-10-batches-py'
    dataset.limit_training_set_size(10000)
    tournament = tmnt.Tournament() # maybe pass in defaults here?
    tournament.run(dataset)


if __name__ == "__main__":
    main()


