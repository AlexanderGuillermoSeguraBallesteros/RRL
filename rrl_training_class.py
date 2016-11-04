

class rrl_training:
    def __init__(self, initial_theta, inputs_window, training_window, training_data, iterations,
                 independent_training=True, optimization_function="sharpe", transaction_costs=0.001,
                 number_of_shares=1):
        training_data = np.asarray(training_data)
        maximum_data_length = training_window * iterations + inputs_window
        training_data_length = training_data.size
        if maximum_data_length > training_data_length:
            raise ValueError("More data is required for the training, try to reduce training window or iterations")
        self.theta = initial_theta
        self.inputs_window = inputs_window

