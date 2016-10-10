import os
from collections import defaultdict
import pathos.multiprocessing as mp

from ecnn.tensorflow_model import TensorflowModel
from ecnn.class_defs import *
import ecnn.model_utils as model_utils
import ecnn.functions as functions


class Tournament(object):
    def run(self, dataset, fitness_function):
        tournament_report = self._load_tournament_report()
        selected = self._load_selected(tournament_report)
        self._restore_probabilities(tournament_report)

        for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
            self._create_data_dir(generation)
            print('Getting new Generation %d' % (generation))
            model_values = self._get_model_values(generation, selected)
            all_models, population = self._get_all_models_population(generation, tournament_report)

            while population < POPULATION:
                pool = mp.Pool()
                if GPUS > 0:
                    model_runs = [self.get_args(model_values, generation, dataset, gpu) for gpu in range(GPUS)]
                else:
                    model_runs = [self.get_args(model_values, generation, dataset) for _ in range(2)]
                results = pool.map(self.train_model_wrapper, model_runs)
                pool.close()
                pool.join()
                results = list(filter(None.__ne__, results))
                for result in results:
                    trained_model, values = result
                    model_name = '%d_%d' % (generation, population)
                    trained_model.name = model_name
                    print('generation %d: model %d' % (generation, population))
                    print('model, accuracy:', model_name, trained_model.validation_accuracy)
                    print('model, params:', model_name, trained_model.trainable_parameters)
                    all_models[model_name] = trained_model
                    model_utils.save(trained_model, os.path.join(DIR, str(generation), '%d_model.p') % population)
                    model_utils.save(values, os.path.join(DIR, str(generation), '%d_values.p') % population)
                    tournament_report[generation]['all'] = all_models
                    model_utils.save(tournament_report, os.path.join(DIR, 'report.p'))
                    population += 1
            # regenrate val set
            dataset.update_validation_set()

            key_function = fitness_function(all_models.values())

            selected = model_utils.select_models(all_models, key_function, SELECT)
            tournament_report[generation]['selected'] = selected
            if generation > 0:
                model_utils.update_probabilities(all_models, key_function)
                mutation_alpha_beta = {m.__name__: (m.alpha, m.beta) for m in model_utils.Mutation.__subclasses__()}
                layer_alpha_beta = {lu.__name__: (lu.alpha, lu.beta) for lu in model_utils.LayerUtils.__subclasses__()}
                tournament_report[generation]['mutation_alpha_beta'] = mutation_alpha_beta
                tournament_report[generation]['layer_alpha_beta'] = layer_alpha_beta

            model_utils.save(all_models, os.path.join(DIR, str(generation), 'summary.p'))
            model_utils.save(tournament_report, os.path.join(DIR, 'report.p'))
            if REMOVE_VALUES:
                model_utils.remove_values(all_models.keys(), selected)


    def test(self, dataset, generation):
        tournament_report = model_utils.load(os.path.join(DIR, 'report.p'))
        selected = tournament_report[generation]['selected']
        pool = mp.Pool()
        model_values = model_utils.generate_mutated_models(selected)
        model_runs = [self.get_test_args(model_values, dataset, gpu) for gpu in range(GPUS)]
        results = pool.map(self.test_model_wrapper, model_runs)
        output = {}
        for result in results:
            if result is not None:
                model, test_accuracy = result
                output[model.name] = test_accuracy
                print('modedl, test_accuracy', model_utils.model_to_string(model), test_accuracy)
        model_utils.save(output, os.path.join(DIR, 'test_results'+str(generation)+'.p'))

    def test_model(self, tf_model, dataset, saved_values, gpu=None):
        if gpu is not None:
            os.environ.update({'CUDA_VISIBLE_DEVICES': str(gpu)})
            print('AVAILABLE GPU', os.environ['CUDA_VISIBLE_DEVICES'])

        print('testing', model_utils.model_to_string(tf_model.model))
        try:
            model, test_accuracy = tf_model.run(dataset, saved_values, gpu=gpu)
            return model, test_accuracy
        except ValueError as ve:
            return None
            print('value error')
        except Exception as e:
            print('runtime error')
            print(str(e))
            return None

    def test_model_wrapper(self, args):
        return self.test_model(*args)

    def get_test_args(self, model_values, dataset, gpu=None):
        model, saved_values = model_values.__next__()
        print('restored', model_utils.model_to_string(model))
        tf_model = TensorflowModel(model)
        return (tf_model, dataset, saved_values, gpu)

    def get_args(self, model_values, generation, dataset, gpu=None):
        training_functions = TrainingFunctions(iterations=functions.iterations,
                                               learning_rate=functions.learning_rate(),
                                               batch_size=functions.batch_size,
                                               regularization=functions.regularization,
                                               stopping_rule=functions.stopping_rule,
                                               keep_prob_conv=functions.keep_prob_conv,
                                               keep_prob_dense=functions.keep_prob_dense)
        model, saved_values = model_values.__next__()
        model.generation = generation
        print('restored', model_utils.model_to_string(model))
        if model.mutation == 'Keep':
            training_functions.learning_rate = functions.learning_rate(model.learning_rate)
        tf_model = TensorflowModel(model)
        return (tf_model, dataset, saved_values, training_functions, gpu)



    def train_model_wrapper(self, args):
        return self.train_model(*args)

    def train_model(self, tf_model, dataset, saved_values, training_functions, gpu):
        if gpu is not None:
            os.environ.update({'CUDA_VISIBLE_DEVICES': str(gpu)})
            print('AVAILABLE GPU', os.environ['CUDA_VISIBLE_DEVICES'])
        try:
            trained_model, new_values = tf_model.run(dataset, saved_values, training_functions=training_functions,
                                                     gpu=gpu)
            return (trained_model, new_values)
        except ValueError as ve:
            print(str(ve))
            return None
        except Exception as e:
            print(str(e))
            print('runtime error')
            return None

    def _load_tournament_report(self):
        if CURRENT_GENERATION == 0 and CURRENT_POPULATION == 0:
            return defaultdict(dict)
        else:
            return model_utils.load(os.path.join(DIR, 'report.p'))

    def _load_selected(self, tournament_report):
        if CURRENT_GENERATION - 1 in tournament_report:
            return tournament_report[CURRENT_GENERATION - 1]['selected']
        else:
            return {}

    def _create_data_dir(self, generation):
        if generation == CURRENT_GENERATION:
            if CURRENT_POPULATION == 0:
                os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)
        else:
            os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)

    def _get_model_values(self, generation, selected):
        if generation > 0:
            return  model_utils.generate_mutated_models(selected)
        else:
            return model_utils.generate_initial_population()

    def _get_all_models_population(self, generation, tournament_report):
        if generation == CURRENT_GENERATION and CURRENT_POPULATION != 0:
            return tournament_report[generation]['all'], CURRENT_POPULATION
        else:
            return {}, 0
    def _restore_probabilities(self, tournament_report):
        if CURRENT_GENERATION > 1:
            model_utils.restore_probabilities(tournament_report[CURRENT_GENERATION - 1]['mutation_alpha_beta'],
                                              tournament_report[CURRENT_GENERATION - 1]['layer_alpha_beta'])
