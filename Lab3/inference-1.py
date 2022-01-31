import numpy as np
import graphics
import rover

def forward_backward(all_possible_hidden_states,
                     all_possible_observed_states,
                     prior_distribution,
                     transition_model,
                     observation_model,
                     observations):
    """
    Inputs
    ------
    all_possible_hidden_states: a list of possible hidden states
    all_possible_observed_states: a list of possible observed states
    prior_distribution: a distribution over states

    transition_model: a function that takes a hidden state and returns a
        Distribution for the next state
    observation_model: a function that takes a hidden state and returns a
        Distribution for the observation from that hidden state
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    rover.py), and the i-th Distribution should correspond to time
    step i
    """

    
    num_time_steps = len(observations)
    # Declaring
    forward_messages = [None] * num_time_steps
    backward_messages = [None] * num_time_steps
    marginals = [None] * num_time_steps
     
    # TODO: Compute the forward messages
    # Initializing Forward Message
    alpha_z0 = rover.Distribution()
    for zo in all_possible_hidden_states:
        alpha_z0[zo] = prior_distribution[zo] * observation_model(zo)[observations[0]]
    alpha_z0.renormalize()
    forward_messages[0] = alpha_z0
    # Compute Forward Message using Recursion:
    for i in range(1, num_time_steps):
        forward_messages[i] = rover.Distribution()
        for zn in all_possible_hidden_states:
            forward_messages[i][zn] = 0

            # no observation:
            if observations[i] == None:
                for zn_prev in all_possible_hidden_states:
                    alpha_z_1 = forward_messages[i-1][zn_prev]
                    p_cond = transition_model(zn_prev)[zn]
                    forward_messages[i][zn] += alpha_z_1 * p_cond
            # with observations:
            else: 
                for zn_prev in all_possible_hidden_states:
                    alpha_z_1 = forward_messages[i-1][zn_prev]
                    p_cond = transition_model(zn_prev)[zn]
                    forward_messages[i][zn] += alpha_z_1 * p_cond  
                p_obs = observation_model(zn)[observations[i]]
                forward_messages[i][zn] = p_obs * forward_messages[i][zn] 
            

        # Renormalizing    
        forward_messages[i].renormalize()
        
    # TODO: Compute the backward messages
    # Initializing Backward Message
    beta_zN1 = rover.Distribution()
    for i in all_possible_hidden_states:
        beta_zN1[i] = 1  
    backward_messages[num_time_steps - 1] = beta_zN1
    
    # Compute Backward Message using Recursion:
    for i in range(num_time_steps-2, -1, -1):
        backward_messages[i] = rover.Distribution()
        for zn in all_possible_hidden_states:
            backward_messages[i][zn] = 0
            for zn_next in all_possible_hidden_states:
                # no observation:
                if observations[i+1] == None:
                    beta_z_1 = backward_messages[i+1][zn_next]
                    p_cond = transition_model(zn)[zn_next]
                    backward_messages[i][zn] += beta_z_1 * p_cond
                # with observations:
                else: 
                    beta_z_1 = backward_messages[i + 1][zn_next]
                    p_obs = observation_model(zn_next)[observations[i+1]]
                    p_cond = transition_model(zn)[zn_next]

                    backward_messages[i][zn] += beta_z_1 * p_obs * p_cond
        # Renormalizing    
        backward_messages[i].renormalize()

    # TODO: Compute the marginals 
    for i in range(num_time_steps):
        marginals[i] = rover.Distribution()
        normalize_factor = 0
        for zn in all_possible_hidden_states:
            alpha = forward_messages[i][zn]
            beta = backward_messages[i][zn]
            normalize_factor += alpha * beta
            if alpha * beta != 0:
                marginals[i][zn] = alpha * beta
        
        if normalize_factor != 0:
            for zn in marginals[i].keys():
                marginals[i][zn] = marginals[i][zn] / normalize_factor

        print(i, marginals[i])
    
    return marginals

def Viterbi(all_possible_hidden_states,
            all_possible_observed_states,
            prior_distribution,
            transition_model,
            observation_model,
            observations):
    """
    Inputs
    ------
    See the list inputs for the function forward_backward() above.

    Output
    ------
    A list of esitmated hidden states, each state is encoded as a tuple
    (<x>, <y>, <action>)
    """

    # TODO: Write your code here
    num_time_steps = len(observations)
    w = [None] * num_time_steps
    
    w[0] = rover.Distribution()
    for z0 in all_possible_hidden_states:
        p_z0 = prior_distribution[z0]
        if observations[0] == None:
            p_cond = 1
        else:
            p_cond = observation_model(z0)[observations[0]]
        if (p_z0 * p_cond) != 0:
            w[0][z0] = np.log(p_z0 * p_cond)

    best_est_zn = [None] * num_time_steps

    for i in range(1, num_time_steps):
        w[i] = rover.Distribution()
        best_est_zn[i] = dict()
        for zn in all_possible_hidden_states:
            max_prev = -10000.00
            for zn_prev in w[i - 1]:
                p_cond_prev = transition_model(zn_prev)[zn]
                if p_cond_prev != 0:
                    current_val = w[i - 1][zn_prev] + np.log(p_cond_prev)
                    if current_val > max_prev:
                        max_prev = current_val
                        best_est_zn[i][zn] = zn_prev   
           
            if observations[i] == None:
                p_cond = 1
            else:
                p_cond = observation_model(zn)[observations[i]]

            if p_cond != 0:

                w[i][zn] = np.log(p_cond) + max_prev


    estimated_hidden_states = [None] * num_time_steps
    best_z0 = - 10000.00
    for z0 in w[num_time_steps-1]:
        current = w[num_time_steps-1][z0]
        if current > best_z0:
            best_z0 = current
            estimated_hidden_states[num_time_steps-1] = z0

    for i in range(1, num_time_steps):
        zi = num_time_steps-i
        zi_prev = estimated_hidden_states[zi]
        estimated_hidden_states[zi-1] = best_est_zn[zi][zi_prev]

    return estimated_hidden_states


if __name__ == '__main__':
   
    enable_graphics = True
    
    missing_observations = True
    if missing_observations:
        filename = 'test_missing.txt'
    else:
        filename = 'test.txt'
            
    # load data    
    hidden_states, observations = rover.load_data(filename)
    num_time_steps = len(hidden_states)

    all_possible_hidden_states   = rover.get_all_hidden_states()
    all_possible_observed_states = rover.get_all_observed_states()
    prior_distribution           = rover.initial_distribution()
    
    print('Running forward-backward...')
    marginals = forward_backward(all_possible_hidden_states,
                                 all_possible_observed_states,
                                 prior_distribution,
                                 rover.transition_model,
                                 rover.observation_model,
                                 observations)
    print('\n')


   
    timestep = num_time_steps - 1
    print("Most likely parts of marginal at time %d:" % (timestep))
    print('\n')
    print(sorted(marginals[timestep].items(), key=lambda x: x[1], reverse=True)[:10])
    print('\n')

  
    print('Running Viterbi...')
    estimated_states = Viterbi(all_possible_hidden_states,
                               all_possible_observed_states,
                               prior_distribution,
                               rover.transition_model,
                               rover.observation_model,
                               observations)
    print('\n')
    
    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10, num_time_steps):
        print(estimated_states[time_step])


    # if you haven't complete the algorithms, to use the visualization tool
    # let estimated_states = [None]*num_time_steps, marginals = [None]*num_time_steps
    #estimated_states = [None]*num_time_steps
    #marginals = [None]*num_time_steps
    
    
    error_v_sum = 0
    error_fb_sum = 0
    estimated_fb_states = [None] * num_time_steps
    for i in range(num_time_steps):
        # Calculating Error for Forward-Backward: 
        p_max = 0
        for zn in marginals[i]:
            p_current = marginals[i][zn]
            if p_current > p_max:
                p_max = p_current
                estimated_fb_states[i] = zn
    
        if estimated_fb_states[i] == hidden_states[i]:
            error_fb_sum += 1
        # Calculating Error for Viterbi: 
        if estimated_states[i] == hidden_states[i]:
            error_v_sum += 1
    error_v = 1-error_v_sum/100
    error_fb = 1-error_fb_sum/100
    
    print('Viterbi Error: ', error_v)
    print('Forward-Backward Error: ', error_fb)
    
    if enable_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()

    

    
        
