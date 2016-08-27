# script to run tourtnament

import os
import pickle
import itertools
import functools
import copy
from collections import defaultdict


from ecnn.tensorflow_model import TensorflowModel
from ecnn.class_defs import *
from ecnn import mock_functions as functions

# TODO SAT finish mutation defs and tests, do regularization


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
# TODO test this withouth tf model first
class Tournament(object):
# How to get all subclasses for sc in Mutation.__subclasses__(): get_prob()
    def run(self):
        tournament_report = defaultdict(dict)  # or load previous
        error_logs = []

        for generation in range(CURRENT_GENERATION, MAX_GENERATIONS):
            os.makedirs(os.path.join(DIR, str(generation)), exist_ok=True)
            training_functions = TrainingFunctions(iterations=functions.iterations,
                                                   learning_rate=functions.learning_rate,
                                                   batch_size=functions.batch_size)
            print('Getting new Generation %d' % (generation))
            if generation > 0:
                model_values = self.generate_mutated_models(tournament_report[generation - 1]['selected'])
            else:
                model_values = self.generate_initial_population()
            model_summaries = {}
            population = 0
            while population < POPULATION:
                try:
                    model, saved_values = model_values.__next__()
                    model.generation = generation
                    #tf_model = TensorflowModel(model)
                    #trained_model, new_values, model_summary = tf_model.run(DATASET, saved_values,
                    #training_functions=training_functions)
                    #for testing
                    trained_model, new_values, model_summary = model, SavedValues(), ModelSummary(validation_accuracy
                                                                                                  = 5,
                                                                                                  validation_x_entropy = 8)
                    model_name = '%d_%d' % (generation, population)
                    trained_model.name = model_name
                    new_values.name = model_name
                    model_summary.name = model_name
                    print('model, accuracy:', model_name, model_summary.validation_accuracy)
                    print('model structure:', self.model_to_string(model))
                    model_summaries[model_name] = model_summary
                    self.save(trained_model, os.path.join(DIR, str(generation), '%d_model.p') % population)
                    self.save(new_values, os.path.join(DIR, str(generation), '%d_values.p') % population)
                    population += 1
                except Exception as e:
                    #print(str(e))
                    error_logs.append({'%d_%d' % (generation, population): str(e)})
                    raise e
            # TODO save report periodically so that it can be loaded if crash happens
            # save summaries for the generation
            selected = self.select_models(model_summaries)
            tournament_report[generation]['selected'] = selected
            print('t0', tournament_report[0])
            self.save(model_summaries, os.path.join(DIR, str(generation), 'summary.p'))

        self.save(tournament_report, os.path.join(DIR,'report.p'))
        self.save(error_logs, os.path.join(DIR, 'errors.p'))

        # TODO write test code to run test stats on final 5 models


    def save(self, obj, filepath):
        with open(filepath, 'wb') as ofile:
            pickle.dump(obj, ofile)

    def select_models(self, model_summaries):
            sorted_models = sorted(model_summaries.values(), key=functions.selection_function)
            return {model_summary.name: model_summary for model_summary in sorted_models[:SELECT]}

    # TODO refactor this
    def generate_initial_population(self):
        ''' Returns a generator that yeilds random models with
        one convolutional layer followed by a fully connected output layer

        '''
        while True:
            model = Model()
            number_of_initial_convo_layers = np.random.randint(1, INITIAL_CONVOLUTIONAL_LAYER)
            for i in range(number_of_initial_convo_layers):
                filter_size, _ = functions.get_filter_size(IMAGE_SHAPE[0], IMAGE_SHAPE[1]) #this will need to change if
                # pooling applied
                filters = functions.get_number_of_filters()
                layer = ConvolutionalLayer(filter_size=filter_size, filters=filters, name='c%d' % i)
                model.convolutional_layers.append(layer)
            logits = OutputLayer()
            model.logits = logits
            yield model, SavedValues()

    # TODO problem, this will momoize all generations, need separae object to destroy this one
    # maybe don't need cuz time more impacted by tf runs and not loading of data, memory more important
    def memoize(self, obj):
        cache = obj.cache = {}
        functools.wraps(obj)
        def memoizer(*args, **kwargs):
            if args not in cache:
                cache[args] = obj(*args, **kwargs)
            return copy.deepcopy(cache[args])
        return memoizer


    # TODO refactor one load function

    def load_model(self, model_name):
        generation, number = model_name.split('_')
        model_path = os.path.join(DIR, str(generation), '%d_model.p' % number)
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return copy.deepcopy(model)


    def load_saved_values(self, model_name):
        generation, number = model_name.split('_')
        params_path = os.path.join(DIR, str(generation), '%d_values.p' % number)
        with open(params_path, 'rb') as params_file:
            values = pickle.load(params_file)
        return copy.deepcopy(values)


    def generate_mutated_models(self, summaries):
        seen = set()
        while True:
            print(summaries.keys())
            models_names = list(summaries.keys())
            random_model_name = np.random.choice(models_names)
            mutations = Mutation.__subclasses__()
            cross_overs = CrossOver.__subclasses__()
            mutation_probabilities = [mutation.get_probability(summaries[random_model_name]) for mutation in mutations]
            max_mutation_prob = max(mutation_probabilities)

            pairs  = itertools.combinations(models_names, 2)
            random_pair = np.random.choice(pairs)
            cross_over_probabilities = [cross_over.get_probability(summaries[random_pair[0]], summaries[random_pair[1]]) for
                                        cross_over in cross_overs]
            max_co_prob = max(cross_over_probabilities)
            if max_co_prob > max_mutation_prob and random_pair not in seen:
                saved_model1, saved_values1 = self.load_model(random_pair[0]), self.load_saved_values(random_pair[0])
                saved_model2, saved_values2 = self.load_model(random_pair[1]), self.load_saved_values(random_pair[1])
                yield cross_overs[cross_over_probabilities.index(max_co_prob)].cross((saved_model1, saved_values1),(saved_model2, saved_values2))

            else:
                saved_model, saved_values = self.load_model(random_model_name), self.load_saved_values(random_model_name)
                yield mutations[mutations.index(max_mutation_prob)].mutate(saved_model, saved_values)


    def load_summaries(generation):
        summary_path = os.path.join(DIR, generation, 'summary.p')
        with open(summary_path, 'rb') as summary_file:
            summaries = pickle.load(summary_file)
        return summaries





    def model_to_string(self, model):
        conv_layers_string = ' --> '.join('[%s, fs: %d, f: %d]' %
                                          (layer.name, layer.filter_size, layer.filters) for layer in
                                          model.convolutional_layers)
        dense_layers_string = ' --> '.join('[%s, h: %d]' %
                                           (layer.name, layer.hidden_units) for layer in
                                           model.dense_layers)
        return ' --> '.join([conv_layers_string, dense_layers_string, model.logits.name])

