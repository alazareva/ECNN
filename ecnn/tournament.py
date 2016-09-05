# script to run tourtnament

import os
import time
from collections import defaultdict
import importlib

from ecnn.tensorflow_model import TensorflowModel
from ecnn.class_defs import *
# from ecnn import mock_functions as functions
import ecnn.model_utils as model_utils
import ecnn.functions as functions

# TODO multi GPu, documentation, testing, orgainze project

# TODO maybe use flask for display?
# TODO use coverage testing

# TODO flag to remove unsused values
# http://www.bitfusion.io/2016/05/09/easy-tensorflow-model-training-aws/

class Tournament(object):
    def run(self, dataset):
        if CURRENT_GENERATION == 0:
            tournament_report = defaultdict(dict)
        else:
            tournament_report = model_utils.load(os.path.join(DIR, 'report.p'))
            selected = tournament_report[CURRENT_GENERATION-1]['selected']
            if CURRENT_GENERATION > 1:
                model_utils.restore_probabilities(tournament_report[CURRENT_GENERATION-1]['mutation_alpha_beta'],
                                              tournament_report[CURRENT_GENERATION - 1]['layer_alpha_beta'])
        error_logs = []

        for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
            os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)
            training_functions = TrainingFunctions(iterations=functions.iterations,
                                                   learning_rate=functions.learning_rate(),
                                                   batch_size=functions.batch_size,
                                                   regularization=functions.regularization,
                                                   stopping_rule=functions.stopping_rule,
                                                   keep_prob_conv=functions.keep_prob_conv,
                                                   keep_prob_dense=functions.keep_prob_dense)
            print('Getting new Generation %d' % (generation))
            if generation > 0:
                model_values = model_utils.generate_mutated_models(selected)
            else:
                model_values = model_utils.generate_initial_population()
            all_models = {}
            population = 0
            while population < POPULATION:
                try:
                    model, saved_values = model_values.__next__()
                    model.generation = generation
                    if model.mutation == 'Keep':
                        training_functions.learning_rate = functions.learning_rate(model.learning_rate)
                        print('restored learning rate', model.learning_rate)
                    print('training Model:', model_utils.model_to_string(model))
                    tf_model = TensorflowModel(model)
                    start_time = time.time()
                    trained_model, new_values  = tf_model.run(dataset, saved_values,
                                                              training_functions=training_functions)

                    duration = time.time() - start_time
                    model_name = '%d_%d' % (generation, population)
                    trained_model.name = model_name

                    print('generation %d: model %d, time = %.2f' % (generation, population, float(duration)))
                    print('model, accuracy:', model_name, trained_model.validation_accuracy)
                    all_models[model_name] = trained_model
                    model_utils.save(trained_model, os.path.join(DIR, str(generation), '%d_model.p') % population)
                    model_utils.save(new_values, os.path.join(DIR, str(generation), '%d_values.p') % population)
                    population += 1
                except ValueError as ve:
                    print(str(ve))
                    continue
                except Exception as e:
                    error_logs.append({'%d_%d' % (generation, population): str(e)})
                    raise e

            # regenrate val set
            dataset.update_validation_set()

            key_function = functions.selection_function_val_accuracy

            selected = model_utils.select_models(all_models, key_function, SELECT)
            tournament_report[generation]['selected'] = selected

            if generation > 0:
                model_utils.update_probabilities(all_models, key_function)
                mutation_alpha_beta = {m.__name__: (m.alpha, m.beta) for m in model_utils.Mutation.__subclasses__()}
                layer_alpha_beta = {lu.__name__: (lu.alpha, lu.beta) for lu in model_utils.LayerUtils.__subclasses__()}
                tournament_report[generation]['mutation_alpha_beta'] = mutation_alpha_beta
                tournament_report[generation]['layer_alpha_beta'] = layer_alpha_beta

                # TODO keep learning rate when mutation is keep

            model_utils.save(all_models, os.path.join(DIR, str(generation), 'summary.p'))
            model_utils.save(tournament_report, os.path.join(DIR,'report.p'))
            model_utils.save(error_logs, os.path.join(DIR, 'errors.p'))
            if REMOVE_VALUES:
                model_utils.remove_values(DIR, all_models.keys(), selected)
        # TODO write test code to run test stats on final 5 models

    def test(self, dataset, number_of_additional_training_iterations):
        # get the best model
        # possibly train more
        # test
        pass


