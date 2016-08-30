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

# TODO next week, unit tests, multi gpu, adaptive research (talk rose)

# TODO maybe use flask for display?
# TODO https://gist.github.com/Mistobaan/dd32287eeb6859c6668d GPU on mac
# TODO use coverage testing
# maybe have a predefined inital network that a user can put in
# # TODO larning rate is a tensor so it can be adjusted during training (can pass in function)
# sess.run(train_step,learning_rate = tf.placeholder(tf.float32, shape=[]) feed_dict={learning_rate: 0.1})
#  TODO have all train related variables as functions to pass in, reg strength, learnin rate, dropout, these can be
# closures and can estimate internal params for functions based on feedback durning training
# TODO maybe have restor function or restor from generation

# TODO flag to remove unsused values
# TODO maybe use average loss and weights
# TODO keep track of mutation codes

class Tournament(object):
    def run(self, dataset):
        tournament_report = defaultdict(dict)  # or load previous
        error_logs = []

        for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
            os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)
            training_functions = TrainingFunctions(iterations=functions.iterations,
                                                   learning_rate=functions.learning_rate,
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
                    print('training Model:', model_utils.model_to_string(model))
                    tf_model = TensorflowModel(model)
                    start_time = time.time()
                    trained_model, new_values  = tf_model.run(dataset, saved_values,
                                                              training_functions=training_functions)

                    duration = time.time() - start_time
                    #for testing
                    '''
                    for l in model.convolutional_layers:
                        l.output_shape = IMAGE_SHAPE
                    trained_model, new_values = model, model_utils.values_for_testing(model)

                    trained_model.validation_accuracy  = 5
                    trained_model.validation_x_entropy = 8
                    '''

                    model_name = '%d_%d' % (generation, population)
                    trained_model.name = model_name
                    trained_model.ancestor = model.name
                    new_values.name = model_name
                    print('generation %d: model %d, time = %.2f' % (generation, population, float(duration)))
                    print('model, accuracy:', model_name, trained_model.validation_accuracy)
                    print('model, x_entropy:', trained_model.validation_x_entropy)
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
            # TODO save report periodically so that it can be loaded if crash happens
            # save summaries for the generation
            selected = model_utils.select_models(all_models)
            tournament_report[generation]['selected'] = selected
            model_utils.save(all_models, os.path.join(DIR, str(generation), 'summary.p'))

            model_utils.save(tournament_report, os.path.join(DIR,'report.p'))
            model_utils.save(error_logs, os.path.join(DIR, 'errors.p'))

        # TODO write test code to run test stats on final 5 models



