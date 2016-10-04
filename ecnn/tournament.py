import os
from collections import defaultdict
import pathos.multiprocessing as mp

from ecnn.tensorflow_model import TensorflowModel
from ecnn.class_defs import *
import ecnn.model_utils as model_utils
import ecnn.functions as functions

# TODO multi GPu, documentation, testing, orgainze project
# run 2 models, work on report, presentation, datavis

# TODO maybe use flask for display?
# TODO use coverage testing

# TODO flag to remove unsused values
# http://www.bitfusion.io/2016/05/09/easy-tensorflow-model-training-aws/

class Tournament(object):
    def run(self, dataset):
        pool = mp.ProcessingPool()
        if CURRENT_GENERATION == 0:
            if CURRENT_POPULATION == 0:
                tournament_report = defaultdict(dict)
            else:
                tournament_report = model_utils.load(os.path.join(DIR, 'report.p'))
        else:
            tournament_report = model_utils.load(os.path.join(DIR, 'report.p'))
            selected = tournament_report[CURRENT_GENERATION-1]['selected']
            if CURRENT_GENERATION > 1:
                model_utils.restore_probabilities(tournament_report[CURRENT_GENERATION-1]['mutation_alpha_beta'],
                                              tournament_report[CURRENT_GENERATION - 1]['layer_alpha_beta'])
        error_logs = []

        for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
            if generation == CURRENT_GENERATION:
                if CURRENT_POPULATION == 0:
                    os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)
            else:
                    os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)

            print('Getting new Generation %d' % (generation))
            if generation > 0:
                if generation == CURRENT_GENERATION and CURRENT_POPULATION > SELECT:
                    model_values = model_utils.generate_mutated_models(selected, keep = False)
                else:
                    model_values = model_utils.generate_mutated_models(selected)
            else:
                model_values = model_utils.generate_initial_population()

            if generation == CURRENT_GENERATION and CURRENT_POPULATION != 0:
                all_models = tournament_report[generation]['all']
                population = CURRENT_POPULATION
            else:
                all_models = {}
                population = 0

            while population < POPULATION:
                if GPUS > 0:
                    model_runs = [self.get_args(model_values, generation, dataset, gpu) for gpu in range(GPUS)]
                else:
                    model_runs = [self.get_args(model_values, generation, dataset) for _ in range(2)]
                results = pool.map(self.train_model_wrapper, model_runs)
                #pool.close()
                #pool.join()
                for result in results:
                    if result:
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

            key_function = functions.sort_on_accuracy_params(all_models.values())

            selected = model_utils.select_models(all_models, key_function, SELECT)
            tournament_report[generation]['selected'] = selected


            if generation > 0:
                model_utils.update_probabilities(all_models, key_function)
                mutation_alpha_beta = {m.__name__: (m.alpha, m.beta) for m in model_utils.Mutation.__subclasses__()}
                layer_alpha_beta = {lu.__name__: (lu.alpha, lu.beta) for lu in model_utils.LayerUtils.__subclasses__()}
                tournament_report[generation]['mutation_alpha_beta'] = mutation_alpha_beta
                tournament_report[generation]['layer_alpha_beta'] = layer_alpha_beta

            model_utils.save(tournament_report, os.path.join(DIR,'report.p'))
            model_utils.save(error_logs, os.path.join(DIR, 'errors.p'))
            if REMOVE_VALUES:
                model_utils.remove_values(DIR, all_models.keys(), selected)


    def retrain(self, models, dir_name,dataset,keep_values = True):
        pool = mp.Pool()
        os.makedirs(os.path.join(DIR, dir_name), exist_ok=True)
        model_values = model_utils.generate_mutated_models(models)
        model_runs = [self.get_args(model_values, dataset, gpu) for gpu in range(GPUS)]
        if not keep_values:
            model_runs = [self.remove_values(run) for run in model_runs]
        results = pool.map(self.train_model_wrapper, model_runs)
        for result in results:
            if result:
                trained_model, values = result
                print('model, accuracy:', str(trained_model.name), trained_model.validation_accuracy)
                model_utils.save(trained_model, os.path.join(DIR, dir_name, '%s_model.p') % str(trained_model.name))
                model_utils.save(values, os.path.join(DIR, dir_name, '%s_values.p') % str(trained_model.name))

    def remove_values(self, run):
        tf_model, dataset, saved_values, training_functions, gpu = run
        return (tf_model, dataset, SavedValues(), training_functions, gpu)


    def test(self, dataset, selected):
        pool = mp.Pool()
        model_values = model_utils.generate_mutated_models(selected)
        model_runs = [self.get_test_args(model_values, dataset, gpu) for gpu in range(GPUS)]
        results = pool.map(self.test_model_wrapper, model_runs)
        for result in results:
            if result is not None:
                model, test_accuracy = result
                print('modedl, test_accuracy', model_utils.model_to_string(model), test_accuracy)



    def test_model(self, tf_model, dataset, saved_values, gpu=None):
        print('testing', model_utils.model_to_string(tf_model.model))
        try:
            model, test_accuracy = tf_model.run(dataset, saved_values, gpu = gpu)
            return model, test_accuracy
        except ValueError as ve:
            return None
        except Exception as e:
            print('runtime error')
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
            print('restored learning rate', model.learning_rate)
        tf_model = TensorflowModel(model)
        return (tf_model, dataset, saved_values, training_functions, gpu)

    def train_model_wrapper(self, args):
        return self.train_model(*args)

    def train_model(self, tf_model, dataset, saved_values, training_functions, gpu=None):
        print('testing', model_utils.model_to_string(tf_model.model))
        try:
            trained_model, new_values = tf_model.run(dataset, saved_values, training_functions=training_functions,
                                                     gpu = gpu)
            return (trained_model, new_values)
        except ValueError as ve:
            print(str(ve))
        except Exception as e:
            print('runtime error')
            return None



