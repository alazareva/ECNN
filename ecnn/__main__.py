import ecnn.datasets.dataset as ds
import ecnn.tournament as tmnt

def main():
    dataset = ds.get_cifar_10_dataset()
    dataset.limit_training_set_size(2000)
    tournament = tmnt.Tournament() # maybe pass in defaults here?
    tournament.run(dataset)


if __name__ == "__main__":
    main()


